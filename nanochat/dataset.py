"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

Bilingual French+English pretraining:
- French: FineWeb2-HQ (fra_Latn) — top 10% quality-filtered French web text
- English: ClimbMix-400B — curated English web text (original nanochat dataset)
Both are downloaded into the same directory and mixed by the dataloader.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Dataset sources

DATASETS = {
    "fr": {
        "name": "FineWeb2-HQ French",
        "base_url": "https://huggingface.co/datasets/epfml/FineWeb2-HQ/resolve/refs%2Fconvert%2Fparquet/fra_Latn/train",
        "max_shard": 434,           # 435 shards total, ~34B tokens
        "filename": lambda i: f"fr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i:04d}.parquet",
        "repack": True,             # strip embeddings column (~2GB → ~200MB)
        "keep_columns": ["text", "id", "url", "date", "quality_score", "language_score"],
    },
    "en": {
        "name": "ClimbMix-400B English",
        "base_url": "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        "max_shard": 6542,          # 6543 shards total, ~400B tokens
        "filename": lambda i: f"en_{i:05d}.parquet",
        "remote_filename": lambda i: f"shard_{i:05d}.parquet",
        "repack": False,
        "keep_columns": None,
    },
    "wiki-fr": {
        "name": "Wikipedia French",
        "base_url": "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.fr/train",
        "max_shard": 16,            # 17 shards, ~3B tokens, encyclopedic quality
        "filename": lambda i: f"wikifr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i}.parquet",
        "repack": True,
        "keep_columns": ["text", "title", "id"],
    },
    "wiki-en": {
        "name": "Wikipedia English",
        "base_url": "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.en/train",
        "max_shard": 40,            # 41 shards, ~4B tokens, encyclopedic quality
        "filename": lambda i: f"wikien_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i}.parquet",
        "repack": True,
        "keep_columns": ["text", "title", "id"],
    },
    "books-fr": {
        "name": "PleIAs French-PD-Books",
        "base_url": "https://huggingface.co/api/datasets/PleIAs/French-PD-Books/parquet/default/train",
        "max_shard": 7,             # 8 shards, ~16B words, classic French literature
        "filename": lambda i: f"booksfr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i}.parquet",
        "repack": True,
        "keep_columns": ["complete_text", "title"],
        "rename_text": "complete_text",  # rename to "text" for dataloader compatibility
    },
    "diverse-fr": {
        "name": "PleIAs French-PD-diverse",
        "base_url": "https://huggingface.co/api/datasets/PleIAs/French-PD-diverse/parquet/default/train",
        "max_shard": 3,             # 4 shards, ~43B words, archives & Google Books
        "filename": lambda i: f"divfr_{i:04d}.parquet",
        "remote_filename": lambda i: f"{i}.parquet",
        "repack": True,
        "keep_columns": ["complete_text", "title"],
        "rename_text": "complete_text",  # rename to "text" for dataloader compatibility
    },
}

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_bilingual")

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir

    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  DATASET NOT FOUND")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  To download the bilingual dataset, run:")
            print()
            print("    python -m nanochat.dataset --fr 65 --en 65   # ~10B tokens, 50/50 mix")
            print()
            print("=" * 80)
            print()

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx, columns=['text'])
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def _download_task(task):
    """Downloads a single shard. Task is (lang, index) tuple."""
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

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()

            dl_path = raw_path if ds["repack"] else filepath + ".tmp"
            with open(dl_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            # Repack if needed (strip heavy columns, rename text column)
            if ds["repack"]:
                table = pq.read_table(raw_path)
                available_cols = [c for c in ds["keep_columns"] if c in table.column_names]
                table = table.select(available_cols)
                # Rename text column if needed (e.g. "complete_text" → "text")
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
            extra = ", embeddings stripped" if ds["repack"] else ""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download bilingual pretraining data")
    # Core datasets
    parser.add_argument("--fr", type=int, default=0,
                        help="FineWeb2-HQ French shards. Each ≈ 78M tokens. -1 = all 435.")
    parser.add_argument("--en", type=int, default=0,
                        help="ClimbMix English shards. Each ≈ 61M tokens. -1 = all 6543.")
    # High-quality supplements
    parser.add_argument("--wiki-fr", type=int, default=0,
                        help="Wikipedia French shards. -1 = all 17. ~3B tokens.")
    parser.add_argument("--wiki-en", type=int, default=0,
                        help="Wikipedia English shards. -1 = all 41. ~4B tokens.")
    parser.add_argument("--books-fr", type=int, default=0,
                        help="PleIAs French-PD-Books shards. -1 = all 8. Classic literature.")
    parser.add_argument("--diverse-fr", type=int, default=0,
                        help="PleIAs French-PD-diverse shards. -1 = all 4. Archives & books.")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    args = parser.parse_args()

    # Map CLI args to dataset keys
    source_counts = {
        "fr": args.fr, "en": args.en,
        "wiki-fr": args.wiki_fr, "wiki-en": args.wiki_en,
        "books-fr": args.books_fr, "diverse-fr": args.diverse_fr,
    }

    if all(v == 0 for v in source_counts.values()):
        parser.error("Specify at least one source. Examples:\n"
                     "  --fr 130 --en 170                    # core bilingual (~20B tokens)\n"
                     "  --wiki-fr -1 --wiki-en -1            # all Wikipedia\n"
                     "  --books-fr -1 --diverse-fr -1        # PleIAs French heritage\n"
                     "  --fr 130 --en 170 --wiki-fr -1 --wiki-en -1 --books-fr -1  # everything")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Build download task list
    tasks = []
    for source, count in source_counts.items():
        if count == 0:
            continue
        ds = DATASETS[source]
        max_shard = ds["max_shard"]
        n = max_shard + 1 if count == -1 else min(count, max_shard + 1)
        for i in range(n):
            tasks.append((source, i))

    # Summary
    summary = {s: sum(1 for k, _ in tasks if k == s) for s in source_counts if source_counts[s] != 0}
    parts = [f"{v} {DATASETS[k]['name']}" for k, v in summary.items()]
    print(f"Downloading {len(tasks)} shards: {', '.join(parts)}")
    print(f"Target directory: {DATA_DIR}")
    print()

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(_download_task, tasks)

    successful = sum(1 for s in results if s)
    print(f"Done! {successful}/{len(tasks)} shards downloaded to {DATA_DIR}")
