#!/usr/bin/env python3
"""
Autonomous ML research loop driven by Claude on AWS Bedrock.
Iteratively modifies nanochat/gpt.py to optimize AttnRes, trains for a few minutes,
keeps improvements, discards regressions. Runs indefinitely.

Configuration:
    - .env                      — secrets (AWS_BEARER_TOKEN_BEDROCK, AWS_REGION_NAME)
    - autoresearch_config.json  — all tunable parameters (model, training, experiment)

Usage:
    python autoresearch.py                 # run with config defaults
    python autoresearch.py --tag apr8       # override experiment tag

Requires:
    - anthropic + boto3: uv add anthropic boto3
    - nanochat dependencies: uv sync --extra gpu
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
GPT_FILE = REPO_ROOT / "nanochat" / "gpt.py"
TRAINING_ARGS_FILE = REPO_ROOT / "autoresearch_training_args.json"
PROGRAM_FILE = REPO_ROOT / "autoresearch_program.md"
RESULTS_FILE = REPO_ROOT / "autoresearch_results.tsv"
CONFIG_FILE = REPO_ROOT / "autoresearch_config.json"
ENV_FILE = REPO_ROOT / ".env"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_dotenv():
    """Load .env file into os.environ (simple key=value, no quotes handling needed)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_config(cli_overrides: dict) -> dict:
    """Load config from JSON, apply CLI overrides."""
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
    else:
        config = {}

    # Defaults
    defaults = {
        "bedrock": {
            "model_id": "us.anthropic.claude-sonnet-4-6",
            "region": "us-east-1",
            "max_tokens": 16384,
        },
        "training": {
            "depth": 8,
            "aspect_ratio": 64,
            "device_batch_size": 4,
            "total_batch_size": 4096,
            "num_iterations": 500,
            "attn_res_block_size": 8,
            "max_seq_len": 512,
            "window_pattern": "L",
        },
        "experiment": {
            "max_experiments": -1,
            "timeout_multiplier": 3,
            "tag": None,
        },
    }

    # Merge defaults < config file < CLI overrides
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = {}
        for key, default_val in section_defaults.items():
            config[section].setdefault(key, default_val)

    # Apply CLI overrides
    for key, value in cli_overrides.items():
        if value is None:
            continue
        for section in config.values():
            if key in section:
                section[key] = value

    # Region from env
    config["bedrock"]["region"] = os.environ.get(
        "AWS_REGION_NAME", config["bedrock"]["region"]
    )

    return config


# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------

def make_client(config: dict):
    """Create Anthropic Bedrock client. Supports ABSK tokens and AWS profiles."""
    try:
        from anthropic import AnthropicBedrock
    except ImportError:
        print("ERROR: pip install anthropic boto3")
        sys.exit(1)

    region = config["bedrock"]["region"]

    # ABSK bearer token (Anthropic Bedrock API Key)
    absk_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if absk_token:
        for key in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]:
            os.environ.pop(key, None)
        return AnthropicBedrock(api_key=absk_token, aws_region=region)

    return AnthropicBedrock(aws_region=region)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=REPO_ROOT
    )
    return result.stdout.strip()

def git_short_hash() -> str:
    return git("rev-parse", "--short", "HEAD")

def git_commit(message: str):
    git("add", str(GPT_FILE))
    git("add", str(TRAINING_ARGS_FILE))
    git("commit", "-m", message)

def git_reset_hard():
    git("checkout", "--", str(GPT_FILE))
    git("checkout", "--", str(TRAINING_ARGS_FILE))
    git("reset", "HEAD~1", "--hard")

def git_branch_exists(branch: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", branch],
        capture_output=True, cwd=REPO_ROOT
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_training_args() -> dict:
    """Load training args that Claude can modify between experiments."""
    if TRAINING_ARGS_FILE.exists():
        return json.loads(TRAINING_ARGS_FILE.read_text())
    return {}


def build_train_cmd(config: dict) -> str:
    t = config["training"]
    # Claude-modifiable args (from autoresearch_training_args.json)
    args = load_training_args()

    cmd = (
        f"uv run python -m scripts.base_train"
        # Fixed by user — Claude cannot change these
        f" --depth={t['depth']} --aspect-ratio={t['aspect_ratio']}"
        f" --max-seq-len={t['max_seq_len']} --device-batch-size={t['device_batch_size']}"
        f" --num-iterations={t['num_iterations']}"
        f" --use-attn-res"
        f" --window-pattern={t['window_pattern']} --run=dummy"
        f" --eval-every={t['num_iterations']}"
        f" --core-metric-every=-1 --sample-every=-1 --save-every=-1"
    )
    # Append Claude-modifiable args
    arg_map = {
        "total_batch_size": "--total-batch-size",
        "weight_decay": "--weight-decay",
        "embedding_lr": "--embedding-lr",
        "unembedding_lr": "--unembedding-lr",
        "matrix_lr": "--matrix-lr",
        "scalar_lr": "--scalar-lr",
        "warmup_steps": "--warmup-steps",
        "warmdown_ratio": "--warmdown-ratio",
        "final_lr_frac": "--final-lr-frac",
        "attn_res_block_size": "--attn-res-block-size",
    }
    for key, flag in arg_map.items():
        if key in args:
            cmd += f" {flag}={args[key]}"
        elif key in t:
            cmd += f" {flag}={t[key]}"

    return cmd


def run_training(config: dict) -> dict:
    """Run a training experiment. Returns dict with val_bpb, peak_vram_mb, status."""
    cmd = build_train_cmd(config)
    log_path = REPO_ROOT / "run.log"
    timeout = config["training"]["num_iterations"] * config["experiment"]["timeout_multiplier"]

    print(f"  Running: {cmd}")

    try:
        with open(log_path, "w") as log_file:
            subprocess.run(
                cmd.split(),
                stdout=log_file, stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                timeout=max(timeout, 600),
            )
    except subprocess.TimeoutExpired:
        return {"val_bpb": 0.0, "peak_vram_mb": 0.0, "status": "crash", "error": "timeout"}

    log_text = log_path.read_text()
    bpb_matches = re.findall(r"Validation bpb:\s+([\d.]+)", log_text)
    mem_matches = re.findall(r"Peak memory usage:\s+([\d.]+)MiB", log_text)

    if not bpb_matches:
        lines = log_text.strip().split("\n")
        error = "\n".join(lines[-30:]) if len(lines) > 30 else log_text
        return {"val_bpb": 0.0, "peak_vram_mb": 0.0, "status": "crash", "error": error}

    return {
        "val_bpb": float(bpb_matches[-1]),
        "peak_vram_mb": float(mem_matches[-1]) if mem_matches else 0.0,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

def init_results():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("commit\tval_bpb\tmemory_mb\tstatus\tdescription\n")

def log_result(commit, val_bpb, memory_mb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_mb:.1f}\t{status}\t{description}\n")

def get_results_history() -> str:
    return RESULTS_FILE.read_text() if RESULTS_FILE.exists() else ""

def get_best_bpb() -> float:
    best = float("inf")
    for line in get_results_history().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            bpb = float(parts[1])
            if 0 < bpb < best:
                best = bpb
    return best


# ---------------------------------------------------------------------------
# Claude interaction
# ---------------------------------------------------------------------------

def build_prompt(current_gpt_py: str, results_history: str, last_outcome: str) -> str:
    training_args = TRAINING_ARGS_FILE.read_text() if TRAINING_ARGS_FILE.exists() else "{}"
    return f"""## Current state

### Results so far
```
{results_history}
```

### Last experiment outcome
{last_outcome}

### Current autoresearch_training_args.json
```json
{training_args}
```

### Current nanochat/gpt.py
```python
{current_gpt_py}
```

## Your task

Propose the next experiment. Follow the priorities in the program.

You can modify TWO files:
1. `nanochat/gpt.py` — model architecture, AttnRes implementation, optimizer groups
2. `autoresearch_training_args.json` — training hyperparameters (LRs, batch size, weight decay, warmup, block_size)

DO NOT change: depth, aspect_ratio, max_seq_len, device_batch_size (these are fixed).

Respond with:
1. **Idea**: One sentence describing what you're trying
2. **Reasoning**: Why this might help (1-2 sentences)
3. **Code**: The COMPLETE `nanochat/gpt.py` in a ```python code block
4. **Args**: The COMPLETE `autoresearch_training_args.json` in a ```json code block (even if unchanged)

IMPORTANT: Return FULL file contents, not diffs.
"""


def call_claude(client, config: dict, system_prompt: str, user_message: str) -> str:
    response = client.messages.create(
        model=config["bedrock"]["model_id"],
        max_tokens=config["bedrock"]["max_tokens"],
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def parse_response(response: str) -> tuple[str, str, str | None]:
    """Extract description, gpt.py code, and training args from Claude's response.
    Returns (description, gpt_code, training_args_json_or_None)."""
    desc_match = re.search(r"\*\*Idea\*\*:\s*(.+?)(?:\n|$)", response)
    description = desc_match.group(1).strip() if desc_match else "experiment"

    # Extract python code block (gpt.py)
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
        raise ValueError("No ```python code block found in response")
    code = code_match.group(1)
    for check in ["class GPT", "def forward", "block_attn_res"]:
        if check not in code:
            raise ValueError(f"Code missing '{check}' — likely truncated or wrong")

    # Extract JSON block (training args) — optional
    args_json = None
    json_match = re.search(r"```json\n(.*?)```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1).strip()
        try:
            json.loads(raw)  # validate
            args_json = raw
        except json.JSONDecodeError:
            pass  # ignore malformed JSON, keep previous args

    return description, code, args_json


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous AttnRes research via Claude on Bedrock")
    parser.add_argument("--tag", type=str, default=None, help="Experiment tag (overrides config)")
    parser.add_argument("--max-experiments", type=int, default=None, help="Max experiments (overrides config)")
    args = parser.parse_args()

    # Load env and config
    load_dotenv()
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config = load_config(cli_overrides)

    tag = config["experiment"]["tag"] or datetime.now().strftime("%b%d").lower()
    branch = f"autoresearch/{tag}"

    print("=== Autoresearch: AttnRes Optimization ===")
    print(f"Branch:  {branch}")
    print(f"Model:   {config['bedrock']['model_id']}")
    print(f"Region:  {config['bedrock']['region']}")
    print(f"Config:  depth={config['training']['depth']}, "
          f"ar={config['training']['aspect_ratio']}, "
          f"bs={config['training']['device_batch_size']}, "
          f"iters={config['training']['num_iterations']}")
    print()

    # Git branch
    if not git_branch_exists(branch):
        git("checkout", "-b", branch)
        print(f"Created branch: {branch}")
    else:
        git("checkout", branch)
        print(f"Resumed branch: {branch}")

    init_results()
    system_prompt = PROGRAM_FILE.read_text()
    client = make_client(config)
    print("Bedrock client ready.\n")

    # --- Baseline ---
    best_bpb = get_best_bpb()
    if best_bpb == float("inf"):
        print("=" * 60)
        print("BASELINE RUN")
        print("=" * 60)
        result = run_training(config)
        commit = git_short_hash()
        if result["status"] == "ok":
            best_bpb = result["val_bpb"]
            log_result(commit, best_bpb, result["peak_vram_mb"], "keep", "baseline")
            print(f"  Baseline val_bpb: {best_bpb:.6f}")
            print(f"  Peak VRAM: {result['peak_vram_mb']:.1f} MiB")
        else:
            print(f"  BASELINE CRASHED: {result.get('error', 'unknown')[:200]}")
            sys.exit(1)
    else:
        print(f"Resuming from best_bpb: {best_bpb:.6f}")

    # --- Experiment loop ---
    max_exp = config["experiment"]["max_experiments"]
    experiment_num = 0
    last_outcome = "Baseline established."

    while True:
        experiment_num += 1
        if 0 < max_exp < experiment_num:
            print(f"\nReached max experiments ({max_exp}). Stopping.")
            break

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT {experiment_num}")
        print("=" * 60)

        current_code = GPT_FILE.read_text()
        results_history = get_results_history()

        print("  Asking Claude for next experiment...")
        user_msg = build_prompt(current_code, results_history, last_outcome)

        try:
            response = call_claude(client, config, system_prompt, user_msg)
        except Exception as e:
            print(f"  Bedrock API error: {e}")
            print("  Waiting 30s before retry...")
            time.sleep(30)
            continue

        try:
            description, new_code, new_args = parse_response(response)
        except ValueError as e:
            print(f"  Failed to parse response: {e}")
            last_outcome = f"PARSE ERROR: {e}. Return the FULL gpt.py in a ```python block."
            continue

        print(f"  Idea: {description}")

        GPT_FILE.write_text(new_code)
        if new_args:
            TRAINING_ARGS_FILE.write_text(new_args)
        git_commit(f"autoresearch: {description}")
        commit = git_short_hash()

        print(f"  Training (commit {commit})...")
        t0 = time.time()
        result = run_training(config)
        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.0f}s")

        if result["status"] == "crash":
            error_short = result.get("error", "")[:150].replace("\n", " ")
            print(f"  CRASH: {error_short}")
            log_result(commit, 0.0, 0.0, "crash", description)
            git_reset_hard()
            last_outcome = f"CRASHED: {description}. Error: {error_short}"

        elif result["val_bpb"] < best_bpb:
            improvement = best_bpb - result["val_bpb"]
            best_bpb = result["val_bpb"]
            log_result(commit, best_bpb, result["peak_vram_mb"], "keep", description)
            print(f"  KEEP! val_bpb: {best_bpb:.6f} (improved by {improvement:.6f})")
            last_outcome = (
                f"KEPT: {description}. "
                f"val_bpb: {best_bpb + improvement:.6f} -> {best_bpb:.6f} "
                f"(delta: -{improvement:.6f}). VRAM: {result['peak_vram_mb']:.0f} MiB."
            )
        else:
            delta = result["val_bpb"] - best_bpb
            log_result(commit, result["val_bpb"], result["peak_vram_mb"], "discard", description)
            print(f"  DISCARD. val_bpb: {result['val_bpb']:.6f} (worse by {delta:.6f})")
            git_reset_hard()
            last_outcome = (
                f"DISCARDED: {description}. "
                f"val_bpb {result['val_bpb']:.6f} vs best {best_bpb:.6f} "
                f"(+{delta:.6f}). Reverted."
            )

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print("=" * 60)
    print(get_results_history())
    print(f"Best val_bpb: {best_bpb:.6f}")


if __name__ == "__main__":
    main()
