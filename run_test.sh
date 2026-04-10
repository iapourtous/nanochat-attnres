cd /home/dess4ever/workspace/recherche/nanochat-attnres
OMP_NUM_THREADS=1 uv run python -m scripts.base_train \
  --depth=4 --max-seq-len=512 \
  --device-batch-size=1 --total-batch-size=512 \
  --num-iterations=5 --use-attn-res --attn-res-block-size=4 \
  --window-pattern=L --run=dummy --save-every=-1
