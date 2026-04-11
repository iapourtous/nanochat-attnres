#!/usr/bin/env python3
"""
Standalone dataset downloader — no torch dependency.
Usage: python3 scripts/download_data.py --fr 130 --en 170 --wiki-fr -1 --wiki-en -1 --books-fr -1 --diverse-fr -1 --europarl -1 --arxiv -1 --mlsum-fr -1 --mlsum-en -1 -w 8
"""

import os
import argparse
import time
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool

# Same as nanochat.common.get_base_dir() but without torch import
def get_base_dir():
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        nanochat_dir = os.path.join(home_dir, ".cache", "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

# Conversion functions for HF streaming datasets
def _convert_reasoning_core(example):
    prompt = example.get("prompt", "")
    answer = example.get("answer", "")
    if not prompt:
        return None
    return prompt + "\n" + answer

def _convert_synlogic(example):
    """Convert SynLogic chat messages to plain text."""
    prompt = example.get("prompt", [])
    if not prompt:
        return None
    parts = []
    for msg in prompt:
        content = msg.get("content", "")
        if content:
            parts.append(content)
    reward = example.get("reward_model", {})
    answer = reward.get("answer", "") if isinstance(reward, dict) else ""
    if answer:
        parts.append(answer)
    return "\n".join(parts)

DATASETS = {
    "fr": {
        "name": "FineWeb2-HQ French",
        "base_url": "https://huggingface.co/datasets/epfml/FineWeb2-HQ/resolve/refs%2Fconvert%2Fparquet/fra_Latn/train",
        "max_shard": 434,
        "filename": lambda i: f"fr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i:04d}.parquet",
        "repack": True,
        "keep_columns": ["text", "id", "url", "date", "quality_score", "language_score"],
    },
    "en": {
        "name": "ClimbMix-400B English",
        "base_url": "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        "max_shard": 6542,
        "filename": lambda i: f"en_{i:05d}.parquet",
        "remote_filename": lambda i: f"shard_{i:05d}.parquet",
        "repack": False,
        "keep_columns": None,
    },
    "wiki-fr": {
        "name": "Wikipedia French",
        "base_url": "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.fr/train",
        "max_shard": 16,
        "filename": lambda i: f"wikifr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i}.parquet",
        "repack": True,
        "keep_columns": ["text", "title", "id"],
    },
    "wiki-en": {
        "name": "Wikipedia English",
        "base_url": "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.en/train",
        "max_shard": 40,
        "filename": lambda i: f"wikien_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i}.parquet",
        "repack": True,
        "keep_columns": ["text", "title", "id"],
    },
    "books-fr": {
        "name": "PleIAs French-PD-Books",
        "base_url": "https://huggingface.co/api/datasets/PleIAs/French-PD-Books/parquet/default/train",
        "max_shard": 7,
        "filename": lambda i: f"booksfr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i}.parquet",
        "repack": True,
        "keep_columns": ["complete_text", "title"],
        "rename_text": "complete_text",
    },
    "diverse-fr": {
        "name": "PleIAs French-PD-diverse",
        "base_url": "https://huggingface.co/api/datasets/PleIAs/French-PD-diverse/parquet/default/train",
        "max_shard": 3,
        "filename": lambda i: f"divfr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i}.parquet",
        "repack": True,
        "keep_columns": ["complete_text", "title"],
        "rename_text": "complete_text",
    },
    "europarl": {
        "name": "Europarl FR↔EN",
        "base_url": "https://huggingface.co/datasets/Helsinki-NLP/europarl/resolve/refs%2Fconvert%2Fparquet/en-fr/train",
        "max_shard": 1,
        "filename": lambda i: f"europarl_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i:04d}.parquet",
        "repack": True,
        "keep_columns": ["translation"],
        "flatten_translation": True,
    },
    "arxiv": {
        "name": "RedPajama arXiv",
        "base_url": "https://huggingface.co/datasets/zxzy/redpajama_arxiv_hf/resolve/refs%2Fconvert%2Fparquet/default/train",
        "max_shard": 11,
        "filename": lambda i: f"arxiv_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i:04d}.parquet",
        "repack": True,
        "keep_columns": ["text"],
    },
    "mlsum-fr": {
        "name": "MLSUM French summaries",
        "base_url": "https://huggingface.co/datasets/mlsum/resolve/refs%2Fconvert%2Fparquet/fr/train",
        "max_shard": 0,
        "filename": lambda i: f"mlsumfr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i:04d}.parquet",
        "repack": True,
        "keep_columns": ["text", "summary"],
        "concat_columns": ["text", "summary"],
    },
    "mlsum-en": {
        "name": "MLSUM English summaries",
        "base_url": "https://huggingface.co/datasets/mlsum/resolve/refs%2Fconvert%2Fparquet/en/train",
        "max_shard": 0,
        "filename": lambda i: f"mlsumen_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i:04d}.parquet",
        "repack": True,
        "keep_columns": ["text", "summary"],
        "concat_columns": ["text", "summary"],
    },
    "nemmath": {
        "name": "Nemotron-CC-Math-v1 4plus",
        "base_url": "https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1/resolve/refs%2Fconvert%2Fparquet/4plus/train",
        "max_shard": 349,
        "filename": lambda i: f"nemmath_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i:04d}.parquet",
        "repack": True,
        "keep_columns": ["text"],
        "hf_gated": True,
    },
    "owm": {
        "name": "OpenWebMath",
        "base_url": "https://huggingface.co/datasets/open-web-math/open-web-math/resolve/refs%2Fconvert%2Fparquet/default/train",
        "max_shard": 149,
        "filename": lambda i: f"owm_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i:04d}.parquet",
        "repack": True,
        "keep_columns": ["text"],
    },
    "rcore": {
        "name": "Reasoning-Core SPT",
        "hf_dataset": "reasoning-core/symbolic-pretraining-pile",
        "hf_config": None,
        "convert_fn": _convert_reasoning_core,
        "filename": lambda i: f"rcore_{i:04d}.parquet",
        "rows_per_shard": 50000,
        "hf_download": True,
    },
    "synlog-easy": {
        "name": "SynLogic Easy",
        "hf_dataset": "MiniMaxAI/SynLogic",
        "hf_config": "easy",
        "convert_fn": _convert_synlogic,
        "filename": lambda i: f"synloge_{i:04d}.parquet",
        "rows_per_shard": 20000,
        "hf_download": True,
    },
    "synlog-hard": {
        "name": "SynLogic Hard",
        "hf_dataset": "MiniMaxAI/SynLogic",
        "hf_config": "hard",
        "convert_fn": _convert_synlogic,
        "filename": lambda i: f"synlogh_{i:04d}.parquet",
        "rows_per_shard": 35000,
        "hf_download": True,
    },
}

DATA_DIR = os.path.join(get_base_dir(), "base_data_bilingual")


def _download_task(task):
    lang, index = task
    ds = DATASETS[lang]

    local_name = ds["filename"](index)
    filepath = os.path.join(DATA_DIR, local_name)
    if os.path.exists(filepath):
        print(f"Skipping {local_name} (already exists)")
        return True

    remote_name = ds["remote_filename"](index)
    url = f"{ds['base_url']}/{remote_name}"
    raw_path = filepath + ".raw.tmp"
    print(f"Downloading {local_name} from {ds['name']}...")

    headers = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        # Try loading from .env file at project root
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        if key.strip() == "HF_TOKEN":
                            hf_token = value.strip()
                            break
    if ds.get("hf_gated") and not hf_token:
        print(f"WARNING: {ds['name']} requires HF_TOKEN. Set HF_TOKEN env var or add to .env")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60, allow_redirects=True, headers=headers)
            response.raise_for_status()

            dl_path = raw_path if ds["repack"] else filepath + ".tmp"
            with open(dl_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            if ds["repack"]:
                table = pq.read_table(raw_path)

                if ds.get("flatten_translation"):
                    trans = table.column("translation")
                    texts = []
                    for row in trans.to_pylist():
                        fr, en = row.get("fr", ""), row.get("en", "")
                        texts.append(f"{fr}\n\n{en}")
                    table = pa.table({"text": texts})
                elif ds.get("concat_columns"):
                    cols = ds["concat_columns"]
                    available = [c for c in cols if c in table.column_names]
                    parts = [table.column(c).to_pylist() for c in available]
                    texts = ["\n\n".join(p for p in row_parts if p) for row_parts in zip(*parts)]
                    table = pa.table({"text": texts})
                else:
                    available_cols = [c for c in ds["keep_columns"] if c in table.column_names]
                    table = table.select(available_cols)
                    rename_col = ds.get("rename_text")
                    if rename_col and rename_col in table.column_names:
                        table = table.rename_columns(
                            ["text" if c == rename_col else c for c in table.column_names]
                        )

                pq.write_table(table, filepath + ".tmp")
                del table
                os.remove(raw_path)

            os.rename(filepath + ".tmp", filepath)
            size_mb = os.path.getsize(filepath) / 1e6
            extra = ", repacked" if ds["repack"] else ""
            print(f"Saved {local_name} ({size_mb:.0f} MB{extra})")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {local_name}: {e}")
            for path in [raw_path, filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {local_name} after {max_attempts} attempts")
                return False

    return False


def _download_hf_dataset(lang, max_shards):
    """Download a dataset via HuggingFace streaming and convert to parquet shards."""
    ds = DATASETS[lang]
    convert_fn = ds["convert_fn"]
    rows_per_shard = ds["rows_per_shard"]
    hf_dataset = ds["hf_dataset"]
    hf_config = ds.get("hf_config")

    print(f"Streaming {ds['name']} from HuggingFace ({hf_dataset})...")

    from datasets import load_dataset
    hf_ds = load_dataset(hf_dataset, hf_config, split="train", streaming=True)

    shard_idx = 0
    buffer = []

    for example in hf_ds:
        # Stop if we've written enough shards
        if max_shards != -1 and shard_idx >= max_shards:
            break

        text = convert_fn(example)
        if text is None:
            continue
        buffer.append(text)

        if len(buffer) >= rows_per_shard:
            filepath = os.path.join(DATA_DIR, ds["filename"](shard_idx))
            if os.path.exists(filepath):
                print(f"Skipping {ds['filename'](shard_idx)} (already exists)")
            else:
                table = pa.table({"text": buffer})
                tmp_path = filepath + ".tmp"
                pq.write_table(table, tmp_path)
                os.rename(tmp_path, filepath)
                size_mb = os.path.getsize(filepath) / 1e6
                print(f"Saved {ds['filename'](shard_idx)} ({size_mb:.0f} MB, {len(buffer)} rows)")
                del table
            buffer = []
            shard_idx += 1

    # Write remaining rows as a final partial shard
    if buffer and (max_shards == -1 or shard_idx < max_shards):
        filepath = os.path.join(DATA_DIR, ds["filename"](shard_idx))
        if os.path.exists(filepath):
            print(f"Skipping {ds['filename'](shard_idx)} (already exists)")
        else:
            table = pa.table({"text": buffer})
            tmp_path = filepath + ".tmp"
            pq.write_table(table, tmp_path)
            os.rename(tmp_path, filepath)
            size_mb = os.path.getsize(filepath) / 1e6
            print(f"Saved {ds['filename'](shard_idx)} ({size_mb:.0f} MB, {len(buffer)} rows)")
            del table
        shard_idx += 1

    print(f"Done streaming {ds['name']}: {shard_idx} shards written.")
    return shard_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download bilingual pretraining data (standalone, no torch)")
    parser.add_argument("--fr", type=int, default=0, help="FineWeb2-HQ French shards. -1 = all 435.")
    parser.add_argument("--en", type=int, default=0, help="ClimbMix English shards. -1 = all 6543.")
    parser.add_argument("--wiki-fr", type=int, default=0, help="Wikipedia French shards. -1 = all 17.")
    parser.add_argument("--wiki-en", type=int, default=0, help="Wikipedia English shards. -1 = all 41.")
    parser.add_argument("--books-fr", type=int, default=0, help="PleIAs French-PD-Books shards. -1 = all 8.")
    parser.add_argument("--diverse-fr", type=int, default=0, help="PleIAs French-PD-diverse shards. -1 = all 4.")
    parser.add_argument("--europarl", type=int, default=0, help="Europarl FR↔EN. -1 = all 2.")
    parser.add_argument("--arxiv", type=int, default=0, help="RedPajama arXiv. -1 = all 12. (~43GB)")
    parser.add_argument("--mlsum-fr", type=int, default=0, help="MLSUM French summaries. -1 = all 1.")
    parser.add_argument("--mlsum-en", type=int, default=0, help="MLSUM English summaries. -1 = all 1.")
    parser.add_argument("--nemmath", type=int, default=0, help="Nemotron-CC-Math shards. -1 = all 350. (~25GB, gated)")
    parser.add_argument("--owm", type=int, default=0, help="OpenWebMath shards. -1 = all 150. (~10GB)")
    # Reasoning datasets (HF streaming)
    parser.add_argument("--rcore", type=int, default=0, help="Reasoning-Core SPT shards (50K rows each). -1 = all.")
    parser.add_argument("--synlog", type=int, default=0, help="SynLogic shards (easy+hard). -1 = all.")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Parallel download workers")
    args = parser.parse_args()

    source_counts = {
        "fr": args.fr, "en": args.en,
        "wiki-fr": args.wiki_fr, "wiki-en": args.wiki_en,
        "books-fr": args.books_fr, "diverse-fr": args.diverse_fr,
        "europarl": args.europarl, "arxiv": args.arxiv,
        "mlsum-fr": args.mlsum_fr, "mlsum-en": args.mlsum_en,
        "nemmath": args.nemmath, "owm": args.owm,
        "rcore": args.rcore,
        "synlog-easy": args.synlog, "synlog-hard": args.synlog,
    }

    if all(v == 0 for v in source_counts.values()):
        parser.error("Specify at least one source. Example: --fr 130 --en 170 --wiki-fr -1")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Separate HF streaming datasets from direct-download datasets
    hf_sources = {}
    direct_sources = {}
    for source, count in source_counts.items():
        if count == 0:
            continue
        if DATASETS[source].get("hf_download"):
            hf_sources[source] = count
        else:
            direct_sources[source] = count

    # Build download task list for direct-download datasets
    tasks = []
    for source, count in direct_sources.items():
        ds = DATASETS[source]
        max_shard = ds["max_shard"]
        n = max_shard + 1 if count == -1 else min(count, max_shard + 1)
        for i in range(n):
            tasks.append((source, i))

    # Summary
    all_parts = []
    if tasks:
        summary = {s: sum(1 for k, _ in tasks if k == s) for s in direct_sources}
        all_parts += [f"{v} {DATASETS[k]['name']}" for k, v in summary.items()]
    for source, count in hf_sources.items():
        label = "all" if count == -1 else str(count)
        all_parts.append(f"{label} {DATASETS[source]['name']} (streaming)")
    print(f"Downloading: {', '.join(all_parts)}")
    print(f"Target directory: {DATA_DIR}")
    print()

    # Download direct-download datasets via parallel Pool
    if tasks:
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(_download_task, tasks)
        successful = sum(1 for s in results if s)
        print(f"Direct downloads: {successful}/{len(tasks)} shards")

    # Download HF streaming datasets sequentially
    for source, count in hf_sources.items():
        _download_hf_dataset(source, count)

    print(f"Done! All downloads saved to {DATA_DIR}")
