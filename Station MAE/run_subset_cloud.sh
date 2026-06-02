DATA_ROOT="/home/renku/work/PeakWeatherDataset"   # ← update this
SAVE_DIR="checkpoints/run_subset_cloud"

python main.py \
    --data_root "$DATA_ROOT" \
    --cache_dir "$SAVE_DIR" \
    --window 72 \
    --max_delta 36 \
    --delta_mode fixed_grid \
    --delta_grid_stride 3 \
    --d_model 64 \
    --enc_layers 2 \
    --dec_layers 2 \
    --batch_size 8 \
    --num_workers 0 \
    --epochs 1 \
    --limit_train_batches 50 \
    --lr 1e-4 \
    --warmup_epochs 0 \
    --factorised_encoder \
    --cross_attn_decoder \
    --exclude_stations 110 \
    --save_dir checkpoints/smoke_test