"""
scripts/extract_masked_attention.py

Diagnostic extraction: for a trained checkpoint with a plain cross-attention
decoder (cross_attn_decoder=True, station_local_decoder=False — v27, v30-nll,
v31), run inference over a sample of test windows at mask_ratio=0.5 and a
SINGLE fixed lead-time, and record the decoder's cross-attention weights.

Masked stations contribute no token to the encoder at all (see
StationMAEEncoder._mask_stations), so a masked station's prediction depends
on its visible neighbours ONLY through the decoder's cross-attention read of
the encoder context (StationMAEDecoder.forward_with_attn /
StationMAE.predict_with_decoder_attn — both added alongside this script,
see model/decoder.py and model/mae.py). This script is the "run once, cache
the result" half of that; notebooks/analysis/14_ablation_loss_and_attention.
ipynb loads the .npz this produces and never needs torch itself.

NOT executed or tested in the authoring environment (no torch / no GPU
available there — repeated `pip install torch` failures from disk quota).
Written to mirror src/test.py's dataset/model construction as closely as
possible (same StationMAE.from_cfg factory, same StationMAEDataset args, same
fixed-eval-mask seeding) to minimise the chance of silent drift, but treat
this as a first draft: run it locally, read the printed shapes/counts, and
fix anything that doesn't match before trusting the notebook's plots.

Usage:
    python src/scripts/extract_masked_attention.py \
        --checkpoint checkpoints/v27/best.ckpt \
        --run_name v27 \
        --data_root /path/to/PeakWeatherDataset \
        --lead_hours 2.0 \
        --n_windows 400 \
        --out test_results/v27/attn_mr0.50_lead2h.npz

Runtime: dominated by the train-split dataset build for obs_stats (same
~4-minute cost test.py pays), then a lightweight per-window encoder+decoder
pass for n_windows samples — no multi-delta grid, no gradient, single lead
only, so this should be a small fraction of a full evaluation dump's cost.
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import load_peakweather, StationMAEDataset          # noqa: E402
from model.mae import StationMAE                                       # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a Lightning .ckpt (v27 / v30-nll / v31 only "
                        "— station_local decoders have nothing to extract).")
    p.add_argument("--run_name", type=str, required=True,
                   help="Label stored in the output .npz, e.g. 'v27'.")
    p.add_argument("--data_root", type=str,
                   default=os.environ.get(
                       "DATA_ROOT",
                       os.path.join(os.path.dirname(os.path.dirname(
                           os.path.dirname(os.path.abspath(__file__)))),
                           "PeakWeatherDataset")),
                   help="Defaults to <project root>/PeakWeatherDataset, same "
                        "as src/download.py's own default.")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--exclude_stations", type=str, nargs="*", default=None,
                   help="Defaults to the checkpoint's own exclude_stations "
                        "(e.g. PFA) if not given.")
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the seed used for the paired predictions.pt "
                        "dumps (test.py default 42) for the masked/visible "
                        "split to line up with the rest of the analysis.")
    p.add_argument("--lead_hours", type=float, default=2.0,
                   help="Single lead-time to extract attention for.")
    p.add_argument("--n_windows", type=int, default=400,
                   help="Number of test windows to sample (capped, not the "
                        "full test set — this is a diagnostic, not a metric).")
    p.add_argument("--stride", type=int, default=9,
                   help="Sliding-window stride in 10-min steps (matches "
                        "run_test_cloud.sh's hardcoded STRIDE=9).")
    p.add_argument("--index_mode", type=str, default="sliding")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def _epoch_hours_to_month_season_tod(hours: np.ndarray):
    """hours-since-epoch (float array) -> (month[1-12], season[0-3 DJF/MAM/JJA/SON], tod_bin[0-3])."""
    ep = datetime.datetime(1970, 1, 1)
    dts = [ep + datetime.timedelta(hours=float(h)) for h in hours.ravel()]
    month = np.array([d.month for d in dts]).reshape(hours.shape)
    season = np.select(
        [np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]),
         np.isin(month, [6, 7, 8])], [0, 1, 2], 3,
    )
    tod_bin = np.array([d.hour // 6 for d in dts]).reshape(hours.shape)
    return month, season, tod_bin


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ── Checkpoint / cfg ────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt, "expected a Lightning checkpoint (state_dict key)"
    saved_cfg = ckpt.get("hyper_parameters", {}).get("cfg", {})

    # Same resolution order as test.py: cfg first, CLI --exclude_stations wins.
    window            = saved_cfg.get("window", 288)
    max_delta         = saved_cfg.get("max_delta", 18)
    delta_mode        = saved_cfg.get("delta_mode", "fixed_grid")
    delta_grid_stride = saved_cfg.get("delta_grid_stride", 3)
    exclude_stations  = args.exclude_stations or saved_cfg.get("exclude_stations") or None

    if not (saved_cfg.get("cross_attn_decoder", False)
            and not saved_cfg.get("station_local_decoder", False)):
        raise SystemExit(
            f"[abort] {args.checkpoint} does not look like a plain "
            f"cross-attention decoder (cross_attn_decoder="
            f"{saved_cfg.get('cross_attn_decoder')}, station_local_decoder="
            f"{saved_cfg.get('station_local_decoder')}). "
            "This script only supports v27 / v30-nll / v31-style checkpoints; "
            "station_local decoders (v32-blind) have no cross-station "
            "attention to extract by design."
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  window={window}  max_delta={max_delta}  delta_mode={delta_mode}  "
          f"delta_grid_stride={delta_grid_stride}  exclude={exclude_stations}")

    # ── Data ────────────────────────────────────────────────────────────
    cache_dir = args.cache_dir or args.data_root
    print("Loading PeakWeather dataset …")
    ds = load_peakweather(root=args.data_root)

    print("Building train dataset for normalisation statistics (~few min) …")
    train_ds = StationMAEDataset(
        ds, window_size=window, delta_steps=max_delta, split="train",
        num_delta_per_sample=1, max_delta_steps=max_delta,
        cache_dir=cache_dir, exclude_stations=exclude_stations,
        delta_mode=delta_mode, delta_grid_stride=delta_grid_stride,
    )
    obs_stats = train_ds.obs_stats

    print("Building test dataset …")
    test_ds = StationMAEDataset(
        ds, window_size=window, delta_steps=max_delta, split="test",
        obs_stats=obs_stats, num_delta_per_sample=1, max_delta_steps=max_delta,
        cache_dir=cache_dir, exclude_stations=exclude_stations,
        delta_mode=delta_mode, delta_grid_stride=delta_grid_stride,
        index_mode=args.index_mode, train_stride=args.stride,
    )
    print(f"  test windows available: {len(test_ds):,} "
          f"(sampling up to {args.n_windows:,})")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # ── Model ───────────────────────────────────────────────────────────
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if not k.startswith("model."):
            continue
        k = k[len("model."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        state_dict[k] = v
    use_nll = any(k.endswith("decoder.log_var_head.weight")
                  or k.endswith("decoder.log_var_head.bias") for k in state_dict)

    model = StationMAE.from_cfg(saved_cfg, dropout=0.0, use_nll_loss=use_nll,
                                 mask_ratio=args.mask_ratio)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [state_dict] missing={len(missing)} unexpected={len(unexpected)} "
              f"— inspect before trusting results if either is non-trivial.")
    model = model.to(device).eval()

    if not model.encoder.factorised:
        raise SystemExit(
            "[abort] this checkpoint's encoder is not factorised; the "
            "attention-extraction methods added to encoder.py/decoder.py "
            "for this analysis assume the factorised path every surviving "
            "checkpoint actually uses."
        )

    # ── Fixed, seeded station mask (same recipe as test.py's --fixed_eval_mask) ──
    N_stations = test_ds.spatial.shape[0]
    num_masked = int(N_stations * args.mask_ratio)
    torch.manual_seed(args.seed)
    noise = torch.rand(N_stations)
    order = torch.argsort(noise)
    fixed_masked  = order[:num_masked].sort().values
    fixed_visible = order[num_masked:].sort().values
    model.encoder.set_fixed_eval_mask(fixed_masked, fixed_visible)
    print(f"  [mask] fixed eval mask from seed {args.seed}: "
          f"{num_masked} masked / {N_stations - num_masked} visible "
          f"(identical for every window)")

    visible_np = fixed_visible.numpy()
    masked_np  = fixed_masked.numpy()
    N_vis      = len(visible_np)
    N_msk      = len(masked_np)

    # ── Which K-grid column is our target lead time? ──────────────────────
    lead_steps = round(args.lead_hours * 6)   # 10-min steps per hour = 6
    grid = np.asarray(getattr(test_ds, "delta_grid", None))
    if grid is None or grid.size == 0:
        raise SystemExit("[abort] test_ds.delta_grid not found — this script "
                          "assumes delta_mode='fixed_grid'.")
    k_idx = int(np.argmin(np.abs(grid - lead_steps)))
    actual_steps = int(grid[k_idx])
    if actual_steps != lead_steps:
        print(f"  [warn] requested {args.lead_hours}h ({lead_steps} steps) not "
              f"exactly on the grid; using the nearest entry: {actual_steps} "
              f"steps = {actual_steps / 6:.2f}h")

    # ── Accumulators ────────────────────────────────────────────────────
    # contrib_sum[m, v]: total (over sampled windows) cross-attention mass a
    # masked station m's Δ=lead prediction placed on visible station v,
    # summed over encoder timesteps W. Indexed by POSITION in masked_np /
    # visible_np, not raw station id — mapped back to station ids on save.
    contrib_sum   = np.zeros((N_msk, N_vis), dtype=np.float64)
    contrib_count = 0
    records = []   # one row per (window, masked-station): dict of scalars

    n_seen = 0
    with torch.no_grad():
        for batch in test_loader:
            if n_seen >= args.n_windows:
                break
            x           = batch["x"].to(device)
            x_mask      = batch["x_mask"].to(device)
            spatial     = batch["spatial"].to(device)
            x_hours     = batch["x_hours"].to(device)
            y_raw       = batch["y"].to(device)          # (B, K, N, V)
            y_mask_raw  = batch["y_mask"].to(device)
            y_hours_all = batch["y_hours"].to(device)     # (B, K)
            delta_all   = batch["delta_steps"].to(device)  # (B, K)

            y_hours_k = y_hours_all[:, k_idx]              # (B,)
            delta_k   = delta_all[:, k_idx]                # (B,)
            y_k       = y_raw[:, k_idx]                     # (B, N, V)
            ymask_k   = y_mask_raw[:, k_idx]                # (B, N, V)

            preds, midx, vidx, attn = model.predict_with_decoder_attn(
                x, x_mask, spatial, x_hours, y_hours_k, delta_k,
            )
            # attn: (L, B, N, W*N_vis) -> average over layers and heads
            # (heads already averaged inside forward_with_attn) -> (B, N, W*N_vis)
            attn_mean = attn.mean(dim=0)
            B, N, WNv = attn_mean.shape
            W = WNv // N_vis
            attn_ws = attn_mean.view(B, N, W, N_vis).sum(dim=2)   # (B, N, N_vis)
            attn_ws = attn_ws.cpu().numpy()

            Vt = model.num_target_vars
            abs_err = (preds[..., :Vt] - y_k[..., :Vt]).abs().cpu().numpy()   # (B, N, Vt)
            ok      = ymask_k[..., :Vt].bool().cpu().numpy()

            hrs = y_hours_k.cpu().numpy()
            month, season, tod = _epoch_hours_to_month_season_tod(hrs)

            for b in range(B):
                if n_seen >= args.n_windows:
                    break
                n_seen += 1
                for mi, m_station in enumerate(masked_np):
                    row_ok = ok[b, m_station]
                    if not row_ok.any():
                        continue
                    err = float(np.nanmean(np.where(row_ok, abs_err[b, m_station], np.nan)))
                    contrib_sum[mi] += attn_ws[b, m_station]
                    records.append((
                        n_seen, int(m_station), float(hrs[b]),
                        int(month[b]), int(season[b]), int(tod[b]), err,
                    ))
                contrib_count += 1

    contrib_mean = contrib_sum / max(contrib_count, 1)

    rec_dtype = np.dtype([
        ("window", "i4"), ("masked_station_pos", "i4"), ("target_hour", "f8"),
        ("month", "i2"), ("season", "i2"), ("tod_bin", "i2"), ("abs_err", "f8"),
    ])
    records_arr = np.array(records, dtype=rec_dtype)

    # No station-id list lives on StationMAEDataset itself. common.py's own
    # loaders recover station identity for a dump by matching its saved
    # "spatial" tensor against build_spatial_features(ds) (see
    # notebooks/analysis/common.py's `assert dev < 1e-3` station-order check)
    # — do the same here instead of inventing an id list that might not
    # match: save test_ds.spatial and let the notebook join it exactly the
    # way it already joins predictions.pt's "spatial" field to the station
    # table via common.py's station_table().
    spatial_np = test_ds.spatial.cpu().numpy()   # (N_stations, 15)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        run_name=args.run_name,
        lead_hours=actual_steps / 6.0,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
        n_windows=n_seen,
        masked_station_pos=masked_np,     # row index into `spatial`
        visible_station_pos=visible_np,   # row index into `spatial`
        spatial=spatial_np,               # (N_stations, 15) — join key, same
                                           # convention as predictions.pt's
                                           # "spatial" field (see common.py)
        contrib_mean=contrib_mean,        # (N_masked, N_visible)
        records=records_arr,
    )
    print(f"Saved {args.out}  (n_windows={n_seen}, "
          f"masked={N_msk}, visible={N_vis}, lead={actual_steps/6:.2f}h)")


if __name__ == "__main__":
    main()
