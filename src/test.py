"""
test.py

Test-set inference for Station-MAE.

Loads a trained checkpoint, runs the model over the held-out test split once per
mask ratio, and writes the raw tensors to

    <save_dir>/<ckpt-stem>_mr<R>/predictions.pt

That file is the only output. No metric is computed here: MAE/RMSE per lead
time, per variable, per station, masked vs visible, and persistence skill are
all derived downstream in notebooks/Test_Results_Exploration.ipynb, which owns
the per-station inverse normalisation. Keeping the numbers in one place stops
the script and the notebook from disagreeing.

predictions.pt keys
-------------------
    preds        (M, K, N, 5)   normalised predictions
    targets      (M, K, N, 6)   normalised targets
    masks        (M, K, N, 6)   sensor availability
    masked_idx   (M, n_masked)  stations hidden from the encoder (empty at MR 0)
    delta_steps  (M, K)         lead times, in 10-min steps
    window_hours (M,)           window start, hours since epoch
    target_hours (M, K)         target time per lead
    spatial      (N, 15)        static station descriptors
    log_var      (M, K, N, 5)   log sigma^2 — only if the checkpoint has a sigma head

Usage
-----
    # Architecture is auto-read from the Lightning checkpoint
    python test.py --data_root /path/to/peakweather \\
                   --checkpoint checkpoints/full_run_cloud_v27/best.ckpt \\
                   --test_mask_ratios 0.0 0.5 --seed 42 \\
                   --save_dir test_results/v27

    Normally invoked through src/scripts/run_test_cloud.sh.

Arguments
---------
  Data
    --data_root        STR   Path to PeakWeather data directory (required)
    --cache_dir        STR   Pre-built tensor cache (defaults to data_root)
    --window           INT   Input window steps (auto-read from checkpoint)
    --max_delta        INT   Max lead-time steps (auto-read from checkpoint)
    --batch_size       INT   Inference batch size (default 32)
    --num_workers      INT   DataLoader workers (default 4)
    --index_mode       STR   "sliding" or "random" window index
    --stride           INT   Window stride for sliding mode
    --global_norm            Global instead of per-station normalisation; must
                             match how the checkpoint was trained

  Model
    --checkpoint       STR   Path to .ckpt (required; repeatable). Every
                             architecture setting is auto-read from the saved cfg.

  Evaluation
    --test_mask_ratios FLT   Mask ratios to sweep (default: the trained ratio)
    --seed             INT   Seeds the station mask, once per mask ratio
    --save_predictions INT   Cap on windows dumped (0 = all)
    --save_dir         STR   Output directory (default "test_results")
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Renku session containers cap /dev/shm (often 64 MB), which the default sharing
# strategy can exhaust when DataLoader workers hand batches back to the main
# process on the large sliding test set → "No space left on device". Route the
# handoff through temp files instead. Same guard as main.py / train_lstm.py /
# test_lstm.py.
torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Station-MAE test-set evaluation")

    # Data
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    # default=None → auto-read from checkpoint; explicit CLI value always wins
    p.add_argument("--window",      type=int, default=None,
                   help="Input window in 10-min steps (default: read from checkpoint, "
                        "fallback 72 = 12 h)")
    p.add_argument("--max_delta",   type=int, default=None,
                   help="Max lead-time in 10-min steps (default: read from checkpoint, "
                        "fallback 36 = 6 h)")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # Model architecture — all flags are auto-detected from Lightning checkpoints.
    # Pass any of these explicitly to override the saved value.
    p.add_argument("--checkpoint",  type=str, nargs="+", required=True,
                   help="One or more .pt / .ckpt checkpoint paths to evaluate. "
                        "For Lightning checkpoints all arch and data settings are "
                        "read automatically; only --data_root is required alongside.")
    p.add_argument("--d_model",     type=int,   default=None)
    p.add_argument("--enc_heads",   type=int,   default=None)
    p.add_argument("--enc_layers",  type=int,   default=None)
    p.add_argument("--dec_heads",   type=int,   default=None)
    p.add_argument("--dec_layers",  type=int,   default=None)
    p.add_argument("--mlp_ratio",   type=float, default=None)
    p.add_argument("--mask_ratio",  type=float, default=None)
    p.add_argument("--factorised_encoder", action="store_true", default=None,
                   help="Axial attention encoder (auto-detected from Lightning checkpoints)")
    p.add_argument("--temporal_patch", type=int, default=None,
                   help="Auto-read from the checkpoint; override only to debug.")
    p.add_argument("--residual_head", action="store_true", default=None,
                   help="v15 residual head; auto-read from the checkpoint cfg.")
    p.add_argument("--temporal_window", type=int, default=None,
                   help="Local temporal attention window (auto-detected from Lightning checkpoints)")
    p.add_argument("--cross_attn_decoder", action="store_true", default=None,
                   help="Cross-attention decoder (auto-detected from Lightning checkpoints)")
    p.add_argument("--device",      type=str, default=None)

    # Output
    p.add_argument("--exclude_stations",   type=str, nargs="+", default=None,
                   help="Override checkpoint exclude_stations. Use abbreviation e.g. PFA.")
    p.add_argument("--global_norm",       action="store_true",
                   help="Use global per-variable normalisation (one mean/std per variable). "
                        "Required when testing checkpoints trained before per-station "
                        "normalisation was introduced. Without this flag, per-station "
                        "normalisation is used, which is inconsistent with old checkpoints "
                        "and produces astronomical RMSE values.")
    p.add_argument("--test_mask_ratios",  type=float, nargs="+", default=None,
                   help="One or more encoder mask ratios to sweep at inference time. "
                        "Default: use the trained mask_ratio from the checkpoint. "
                        "Example: --test_mask_ratios 0.0 0.5 to compare no-masking vs trained. "
                        "  0.0 → all stations visible (pure temporal forecasting). "
                        "  0.5 → 50%% masked (trained setting, gap-filling + forecasting).")
    p.add_argument("--index_mode",        type=str, default="sliding",
                   choices=["sliding", "blocks"],
                   help="Window selection for the test split.\n"
                        "  sliding (default): all contiguous windows, stride=1 (~105k samples).\n"
                        "    Most stable metrics; slow.\n"
                        "  blocks: non-overlapping windows only (~1,460 samples).\n"
                        "    Fast evaluation; use for paper metrics and quick checks.")
    p.add_argument("--stride",            type=int, default=1,
                   help="Sliding forecast-origin spacing in 10-min steps (9 = 90 min = "
                        "1h30 → rolling-origin evaluation). Only used with --index_mode "
                        "sliding; ignored for blocks. Default 1 = every window.")
    p.add_argument("--save_dir",          type=str, default="test_results")
    p.add_argument("--save_predictions",  type=int, default=0,
                   help="Cap the number of test windows dumped to predictions.pt. "
                        "0 = ALL windows (the normal setting). Use a small value "
                        "for a quick smoke test of a new checkpoint. Each window "
                        "holds preds, targets, masks, masked_idx, delta_steps, "
                        "timestamps and station spatial features.")
    p.add_argument("--seed",            type=int, default=42,
                   help="Seed for the EVALUATION-time station mask. The mask at "
                        "mask_ratio>0 is drawn from the global RNG inside "
                        "StationMAEEncoder._mask_stations(); without a fixed seed "
                        "two evaluation runs hide different stations, which makes "
                        "masked-station comparisons across models unpaired. The "
                        "RNG is re-seeded once per mask ratio (see the evaluation "
                        "loop), so a given seed yields the same masked set "
                        "regardless of which other ratios are evaluated in the "
                        "same invocation. Reproducibility additionally requires "
                        "the same --batch_size, --index_mode and --stride, since "
                        "those change how much randomness each pass consumes.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint cfg helper
# ---------------------------------------------------------------------------

def _read_lightning_cfg(path: str) -> dict:
    """
    Load a Lightning .ckpt and return its saved hyper_parameters["cfg"] dict.
    Returns {} if the file is not a Lightning checkpoint or the key is absent.
    Does NOT load model weights — only the cfg sub-dict is read.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("hyper_parameters", {}).get("cfg", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _prediction_sanity_check(res: dict, label: str = "") -> None:
    """
    Fast post-dump sanity check on the freshly collected predictions.

    Purpose: catch a broken model rebuild AT DUMP TIME, not days later in the
    notebook. The v13 incident (input-context block silently dropped) would
    have tripped three of the four checks below.

    Checks (on a ≤500-window subsample, normalised space, ~1 s):
      1. finiteness            — NaN/Inf in predictions
      2. dispersion            — pred std vs target std per variable
                                 (conditional-mean collapse → ratio ≪ 1)
      3. persistence collapse  — Δ>0 overall RMSE vs the last-observation
                                 baseline (same definition as test_lstm.py)
      4. context pathway       — at Δ=0, VISIBLE stations must beat MASKED
                                 ones: the decoder can read visible stations
                                 directly from input_context. visible ≈ masked
                                 is the signature of a severed context path.
    Warnings are printed, never raised — the dump is already on disk and may
    still be useful for diagnosis.
    """
    import numpy as np
    Vt   = res["preds"].shape[-1]
    Mw   = res["preds"].shape[0]
    sub  = torch.arange(0, Mw, max(1, Mw // 500))
    P    = res["preds"][sub].numpy()
    T    = res["targets"][sub][..., :Vt].numpy()
    M    = res["masks"][sub][..., :Vt].numpy() > 0.5
    MI   = res.get("masked_idx")
    vnames = list(res.get("var_names", [f"var{v}" for v in range(Vt)]))
    print(f"\n  ── Sanity check ({label}, {len(sub)} windows) ──")

    # 1. finiteness
    n_bad = int(np.count_nonzero(~np.isfinite(P)))
    print(f"  {'✓' if n_bad == 0 else '⚠'} finiteness: "
          f"{'all finite' if n_bad == 0 else f'{n_bad} NaN/Inf values!'}")

    # 2. dispersion (conditional-mean collapse)
    ratios = []
    for v in range(Vt):
        m = M[..., v]
        r = float(P[..., v][m].std() / max(T[..., v][m].std(), 1e-6))
        ratios.append(r)
    worst = min(ratios)
    flag  = "✓" if worst > 0.5 else "⚠"
    print(f"  {flag} dispersion (pred std / target std): "
          + "  ".join(f"{vnames[v][:4]}={ratios[v]:.2f}" for v in range(Vt))
          + ("" if worst > 0.5 else "   ← ≪1 = mean-collapse"))

    # 3. persistence collapse on Δ>0 (identical definition to test_lstm.py)
    K  = P.shape[1]
    fc = slice(1, K)
    persist = np.repeat(T[:, :1], K, axis=1)
    both    = M & M[:, :1]
    e  = (P[:, fc] - T[:, fc])[both[:, fc]]
    ep = (persist[:, fc] - T[:, fc])[both[:, fc]]
    rm, rp = float(np.sqrt(np.mean(e**2))), float(np.sqrt(np.mean(ep**2)))
    skill  = 1.0 - rm / rp if rp > 0 else float("nan")
    flag   = "✓" if skill > -0.5 else "⚠"
    print(f"  {flag} vs persistence (Δ>0 norm RMSE): model {rm:.4f} "
          f"vs persist {rp:.4f} → skill {skill:+.3f}"
          + ("" if skill > -0.5 else "   ← far below persistence: check the rebuild"))

    # 4. context pathway: visible vs masked stations at Δ=0
    if MI is not None and MI.shape[1] > 0:
        mi  = MI[sub].numpy()
        N   = P.shape[2]
        selm = np.zeros((len(sub), N), bool)
        np.put_along_axis(selm, mi, True, axis=1)
        m0   = M[:, 0]                                    # (m, N, Vt) at Δ=0
        e0   = np.abs(P[:, 0] - T[:, 0])
        mae_mask = float(e0[m0 & selm[:, :, None]].mean())
        mae_vis  = float(e0[m0 & ~selm[:, :, None]].mean())
        ratio    = mae_vis / max(mae_mask, 1e-6)
        flag     = "✓" if ratio < 0.9 else "⚠"
        print(f"  {flag} Δ=0 context path: visible MAE {mae_vis:.4f} vs "
              f"masked {mae_mask:.4f} (ratio {ratio:.2f})"
              + ("" if ratio < 0.9 else
                 "   ← visible ≈ masked: input_context may be severed"))
    print()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # ── Device — priority: CUDA (cloud GPU) > MPS (Apple Silicon) > CPU ──
    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() and
                      torch.backends.mps.is_built() else
            "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cuda.enable_flash_sdp(True)

    print(f"\n[test.py]  device={device}")

    # ── Fail fast on missing checkpoints ──────────────────────────────────
    # Building the datasets takes minutes. Validate first, and refuse to run at
    # all if NOTHING can be evaluated — otherwise the config below silently
    # falls back to library defaults (d_model=128, window=288, max_delta=18),
    # the run "succeeds", and writes nothing. That failure mode cost a full
    # evaluation cycle once; it must be loud.
    _missing = [p for p in args.checkpoint if not os.path.exists(p)]
    _present = [p for p in args.checkpoint if os.path.exists(p)]
    if _missing:
        print("\n[ERROR] checkpoint(s) not found:")
        for p in _missing:
            print(f"    {p}")
            _d = os.path.dirname(p)
            if os.path.isdir(_d):
                _sib = sorted(f for f in os.listdir(_d) if f.endswith(".ckpt"))
                print(f"      dir exists; .ckpt files here: {_sib or '(none)'}")
            elif os.path.isdir(os.path.dirname(_d)):
                _runs = sorted(os.listdir(os.path.dirname(_d)))
                print(f"      no such dir. Available runs in "
                      f"{os.path.dirname(_d)}: {_runs or '(none)'}")
            else:
                print(f"      parent directory does not exist either: {_d}")
    if not _present:
        raise SystemExit(
            "\n[ABORT] none of the requested checkpoints exist — nothing to evaluate.\n"
            "        Check CHECKPOINT in src/scripts/run_test_cloud.sh, and that it\n"
            "        matches RUN_NAME (the run being written to test_results/)."
        )
    if _missing:
        print(f"\n  continuing with {len(_present)} of {len(args.checkpoint)} checkpoint(s)")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Auto-read settings from first Lightning checkpoint ────────────────
    # Loads only the cfg sub-dict (no weights); fast even for large checkpoints.
    # CLI flags always win when explicitly provided (non-None).
    _ckpt_cfg: dict = {}
    for _p in _present:
        _ckpt_cfg = _read_lightning_cfg(_p)
        if _ckpt_cfg:
            print(f"Auto-detected settings from checkpoint: {_p}")
            break
    if not _ckpt_cfg:
        # The checkpoint exists but carries no hyper_parameters. Every setting
        # below then falls back to a library default that almost certainly does
        # not match how the model was trained — say so rather than printing a
        # confident banner full of wrong numbers.
        print("\n[WARN] no cfg found inside the checkpoint(s).")
        print("       Architecture and window settings will fall back to DEFAULTS")
        print("       (window=288, max_delta=18, d_model=128) unless passed on the CLI.")
        print("       If the banner below does not match your training run, stop now.\n")

    def _resolve(cli_val, cfg_key: str, fallback):
        """Return CLI value if set (not None), else checkpoint cfg, else fallback."""
        if cli_val is not None:
            return cli_val
        return _ckpt_cfg.get(cfg_key, fallback)

    # Data settings — resolved before datasets are built
    window             = _resolve(args.window,    "window",    288)
    max_delta          = _resolve(args.max_delta, "max_delta", 18)
    # CLI --exclude_stations overrides checkpoint value; use abbreviation e.g. PFA
    exclude_stations   = args.exclude_stations or _ckpt_cfg.get("exclude_stations", None) or None
    delta_mode         = _ckpt_cfg.get("delta_mode",         "fixed_grid")
    delta_grid_stride  = _ckpt_cfg.get("delta_grid_stride",  3)

    # Arch settings — resolved per-checkpoint below, but define defaults here
    _d_model    = _resolve(args.d_model,    "d_model",    128)
    _enc_heads  = _resolve(args.enc_heads,  "enc_heads",  4)
    _enc_layers = _resolve(args.enc_layers, "enc_layers", 4)
    _dec_heads  = _resolve(args.dec_heads,  "dec_heads",  4)
    _dec_layers = _resolve(args.dec_layers, "dec_layers", 2)
    _mlp_ratio  = _resolve(args.mlp_ratio,  "mlp_ratio",  4.0)
    _mask_ratio = _resolve(args.mask_ratio, "mask_ratio", 0.5)

    # ── Evaluation mode ───────────────────────────────────────────────────
    # max_delta == 0 → pure spatial inpainting (gap-filling):
    #   • no lead-times to sweep, no persistence baseline, no skill scores
    #   • gap-filling (masked stations only) is the primary metric
    # max_delta  > 0 → temporal forecasting:
    #   • full delta sweep, lead-1, persistence baseline + skill scores
    #   • gap-filling is a secondary metric
    _is_inpainting = (max_delta == 0)
    _mode_str = "inpainting (max_delta=0)" if _is_inpainting else f"forecasting (max_delta={max_delta})"
    print(f"  window={window}  max_delta={max_delta}  mode={_mode_str}  "
          f"d_model={_d_model}  enc_layers={_enc_layers}  dec_layers={_dec_layers}")

    # ── Data ─────────────────────────────────────────────────────────────
    from data.dataset import load_peakweather, StationMAEDataset

    cache_dir = args.cache_dir or args.data_root

    print("Loading PeakWeather dataset …")
    ds = load_peakweather(root=args.data_root)

    # ── Station table diagnostic ──────────────────────────────────────────────
    # Printed once so you can verify station IDs and find the correct exclusion key.
    _stns = ds.stations_table
    _idx_sample = list(_stns.index[:5])
    _name_col   = next((c for c in ["name", "abbr", "station_name"] if c in _stns.columns), None)
    print(f"  stations_table: {len(_stns)} stations  "
          f"index dtype={_stns.index.dtype}  "
          f"sample indices={_idx_sample}")
    if _name_col:
        print(f"  sample {_name_col}s: {list(_stns[_name_col].head())}")
    if exclude_stations:
        _excl_upper = {str(s).upper() for s in exclude_stations}
        print(f"  looking for: {exclude_stations}")
        _matched = []
        for _idx in _stns.index:
            _candidates = {str(_idx).upper()}
            try:
                _candidates.add(str(int(float(str(_idx)))).upper())
            except (ValueError, TypeError):
                pass
            if _candidates & _excl_upper:
                _matched.append(str(_idx))
        print(f"  matched indices: {_matched if _matched else 'NONE — use abbreviation, e.g. PFA not 110'}")

    # Build train_ds only to get obs_stats (do not iterate over it).
    # Exclude the same stations as training so normalisation stats match exactly.
    print("Building train dataset for normalisation statistics …")
    if args.global_norm:
        print("  Normalisation: GLOBAL (per-variable) — use for old checkpoints")
    else:
        print("  Normalisation: per-station (per-station × per-variable)")

    train_ds = StationMAEDataset(
        ds, window_size=window, delta_steps=max_delta, split="train",
        num_delta_per_sample=1, max_delta_steps=max_delta,
        cache_dir=cache_dir,
        exclude_stations=exclude_stations,
        delta_mode=delta_mode,
        delta_grid_stride=delta_grid_stride,
        global_norm=args.global_norm,
    )
    obs_stats = train_ds.obs_stats
    print("  obs_stats ready (train split)")

    print("Building test dataset …")
    test_ds = StationMAEDataset(
        ds, window_size=window, delta_steps=max_delta, split="test",
        obs_stats=obs_stats,
        num_delta_per_sample=1,
        max_delta_steps=max_delta,
        cache_dir=cache_dir,
        exclude_stations=exclude_stations,
        delta_mode=delta_mode,
        delta_grid_stride=delta_grid_stride,
        index_mode=args.index_mode,
        train_stride=args.stride,
        global_norm=args.global_norm,
    )
    _window_mode_str = (
        f"non-overlapping blocks (~{len(test_ds):,} windows)"
        if args.index_mode == "blocks"
        else f"sliding stride={args.stride} ({len(test_ds):,} windows)"
    )
    print(f"  test samples: {_window_mode_str}  "
          f"(delta_mode={delta_mode}"
          + (f", grid=[0..{max_delta} step {delta_grid_stride}] K={len(test_ds.delta_grid)}"
             if delta_mode == "fixed_grid" else "") + ")")

    _use_persistent = (args.num_workers > 0)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_use_persistent,
        prefetch_factor=(4 if _use_persistent else None),
    )


    # ── Per-checkpoint evaluation ─────────────────────────────────────────

    for ckpt_path in args.checkpoint:
        if not os.path.exists(ckpt_path):
            print(f"\n[SKIP] checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_path}")

        # ── Load checkpoint (Lightning .ckpt or legacy .pt) ─────────────
        # map_location="cpu", NOT `device`. torch.load's deserializer restores
        # each tensor's storage on `map_location` DIRECTLY during unpickling
        # (serialization.py: default_restore_location -> obj.to(device=...)),
        # which is a different code path from an ordinary `.to(device)` call
        # on an already-constructed tensor. On some virtualised CUDA profiles
        # (observed here: NVIDIA A10-8Q, a GRID vGPU slice) that direct-to-CUDA
        # storage path raises "CUDA driver error: operation not supported"
        # even though CUDA otherwise works fine in the same process — the model
        # is moved to `device` two lines below via `.to(device)`, an ordinary
        # tensor op, which does not hit this restriction.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            # Lightning format — weights stored under "state_dict" with "model." prefix
            ckpt_epoch    = ckpt.get("epoch", "?")
            ckpt_val_loss = float("nan")

            # All arch settings: checkpoint cfg wins, CLI flags override when set
            saved_cfg = ckpt.get("hyper_parameters", {}).get("cfg", {})

            def _arch(cli_val, cfg_key: str, fallback):
                if cli_val is not None:
                    return cli_val
                return saved_cfg.get(cfg_key, fallback)

            c_d_model    = _arch(args.d_model,    "d_model",           _d_model)
            c_enc_heads  = _arch(args.enc_heads,  "enc_heads",         _enc_heads)
            c_enc_layers = _arch(args.enc_layers, "enc_layers",        _enc_layers)
            c_dec_heads  = _arch(args.dec_heads,  "dec_heads",         _dec_heads)
            c_dec_layers = _arch(args.dec_layers, "dec_layers",        _dec_layers)
            c_mlp_ratio  = _arch(args.mlp_ratio,  "mlp_ratio",         _mlp_ratio)
            c_mask_ratio = _arch(args.mask_ratio, "mask_ratio",        _mask_ratio)
            factorised   = _arch(args.factorised_encoder, "factorised_encoder", False)
            temp_window  = _arch(args.temporal_window,   "temporal_window",    0)
            temp_patch   = _arch(args.temporal_patch,    "temporal_patch",     1)
            cross_attn   = _arch(args.cross_attn_decoder,"cross_attn_decoder", False)
            # v15 structural settings — recorded in cfg by main.py
            residual     = bool(saved_cfg.get("residual_head", False))

            # ── v18–v22 structural settings ──────────────────────────────
            # These were recorded by main.py from the start but never READ
            # here, so every one of them silently reverted to its default
            # when rebuilding the model. That is the same defect class as
            # input_cross_attn below, and it is not survivable:
            #
            #   value_embedding  "mlp"/"fourier" -> "linear" swaps the whole
            #                    per-variable map (mlp_w1/w2 vs var_weights)
            #   static_in_token  True -> False drops var_proj.static_maps AND
            #                    re-adds pos_emb/station_emb to the token
            #   direct_head      True -> False rebuilds a decoder that the
            #                    trained model does not have
            #
            # The defaults below are the pre-v18 behaviour, so v15 and
            # earlier checkpoints keep loading exactly as they did.
            value_emb    = saved_cfg.get("value_embedding", "linear") or "linear"
            static_tok   = bool(saved_cfg.get("static_in_token", False))

            # Strip "model." prefix (Lightning) and then "_orig_mod." prefix
            # (torch.compile wraps parameters under OptimizedModule).
            state_dict = {}
            for k, v in ckpt["state_dict"].items():
                if not k.startswith("model."):
                    continue
                k = k[len("model."):]
                if k.startswith("_orig_mod."):
                    k = k[len("_orig_mod."):]
                state_dict[k] = v
        else:
            # Legacy format saved by the old engine/train.py
            ckpt_epoch    = ckpt.get("epoch",    "?")
            ckpt_val_loss = ckpt.get("val_loss", float("nan"))
            c_d_model    = _d_model
            c_enc_heads  = _enc_heads
            c_enc_layers = _enc_layers
            c_dec_heads  = _dec_heads
            c_dec_layers = _dec_layers
            c_mlp_ratio  = _mlp_ratio
            c_mask_ratio = _mask_ratio
            factorised   = bool(args.factorised_encoder)
            temp_window  = args.temporal_window or 0
            temp_patch   = args.temporal_patch or 1
            cross_attn   = bool(args.cross_attn_decoder)
            residual     = bool(args.residual_head)          # v15
            # Legacy .pt checkpoints all predate v18, so the pre-v18 defaults
            # are the correct reconstruction here.
            value_emb    = "linear"
            static_tok   = False
            state_dict   = ckpt["model_state_dict"]
            saved_cfg    = {}

        print(f"  Saved at epoch {ckpt_epoch}  "
              + (f"val_loss={ckpt_val_loss:.5f}" if ckpt_val_loss == ckpt_val_loss else ""))
        print(f"  d_model={c_d_model}  enc_layers={c_enc_layers}  dec_layers={c_dec_layers}  "
              f"mlp_ratio={c_mlp_ratio}")
        print(f"  factorised_encoder={factorised}  "
              f"temporal_window={temp_window}  temporal_patch={temp_patch}  "
              f"cross_attn_decoder={cross_attn}")
        print(f"  [v15] residual_head={residual}")
        print(f"  [v18+] value_embedding={value_emb}  "
              f"static_in_token={static_tok}")

        # ── Detect an NLL (heteroscedastic) checkpoint ────────────────────────
        # The cfg key "nll_loss" is unreliable (absent/None on older runs such as
        # v9), so infer it from the weights: an NLL model has a second decoder
        # head predicting log σ².  Without this the head would be silently
        # dropped by strict=False and the predicted uncertainty lost.
        # ── Structural flags inferred from the WEIGHTS, not the cfg ───────
        # The cfg dict is not a reliable record: `input_context_cross_attn` was
        # never written into it, and test.py never passed it, so every
        # evaluation ever run rebuilt the decoder WITHOUT its input-context
        # cross-attention block. strict=False then dropped
        # decoder.input_cross_attn.* silently, severing the direct path from the
        # stations' current observations to the decoder. Symptom: the model
        # predicts a hidden station exactly as well as one it was shown.
        #
        # The state_dict cannot lie about which modules were trained, so infer
        # from it. This also fixes v9/v11/v12/v13 retroactively, whose cfgs
        # predate the key.
        _has_input_ctx = any("decoder.input_cross_attn." in k for k in state_dict)
        if _has_input_ctx:
            raise SystemExit(
                "\n[ABORT] This checkpoint contains decoder.input_cross_attn.* "
                "weights — it predates v15, where the input-context pathway was "
                "removed. Evaluate pre-v15 checkpoints with the code that "
                "trained them (git checkout main / the training-time commit). "
                "Rebuilding it here would silently drop those trained weights — "
                "the exact class of bug the v13 post-mortem uncovered."
            )

        # ── Shared pos_emb / station_emb / temporal_emb guard ───────────────
        # mae.py now builds ONE PositionalEmbedding/StationEmbedding/
        # TemporalEmbedding instance and registers it under BOTH
        # `encoder.<name>` and `decoder.<name>`, so a station's query and its
        # matching key carry a bit-identical positional fingerprint (see
        # tests/test_shared_embeddings.py). A checkpoint trained BEFORE that
        # change has two INDEPENDENTLY-trained modules at those two paths,
        # with different weight values.
        #
        # This is silent, unlike a missing/unexpected key: both key names
        # exist in the checkpoint AND in the current model, so
        # load_state_dict(strict=False) reports nothing wrong. What actually
        # happens is a double write to the SAME underlying nn.Parameter:
        # Module.load_state_dict recurses through the model's OWN module tree
        # in registration order (`self.encoder` is assigned before
        # `self.decoder` in StationMAE.__init__), so `encoder.<name>.*` is
        # copied in first and `decoder.<name>.*` is copied in second,
        # SILENTLY OVERWRITING it. The encoder's trained position/topography/
        # time understanding is discarded and replaced with the decoder's —
        # for every window, at every station — with no error, no warning, and
        # shapes that match perfectly.
        #
        # Detected by comparing the CHECKPOINT's own two copies: a
        # current-code checkpoint has bit-identical values at both paths (one
        # shared tensor was saved via two attribute names); a pre-fix
        # checkpoint does not, to the precision of two independently
        # optimised parameter sets.
        _shared_mismatch = sorted({
            _name
            for _name in ("pos_emb", "station_emb", "temporal_emb")
            for _ek in state_dict
            if _ek.startswith(f"encoder.{_name}.")
            and (_dk := "decoder." + _ek[len("encoder."):]) in state_dict
            and not torch.equal(state_dict[_ek], state_dict[_dk])
        })

        saved_cfg_for_build = saved_cfg

        _use_nll = any(k.endswith("decoder.log_var_head.weight") or
                       k.endswith("decoder.log_var_head.bias")
                       for k in state_dict)
        print(f"  NLL uncertainty head: {'FOUND — σ will be predicted and saved' if _use_nll else 'absent (point predictions only)'}")

        # Build model
        from model.mae import StationMAE
        # ── Build from the checkpoint's own cfg ──────────────────────────
        # StationMAE._CFG_TO_ARG is the single table mapping saved cfg keys to
        # constructor arguments, shared with main.py. This file used to carry a
        # second, independently maintained list; it drifted, and every v18+
        # checkpoint silently rebuilt with pre-v18 defaults as a result.
        #
        # Overrides win over the checkpoint: dropout is always 0 at inference,
        # use_nll_loss is inferred from the WEIGHTS (the cfg key is unreliable
        # on v9-era runs), and any CLI flag the user set explicitly wins over
        # what the checkpoint recorded.
        _overrides = {"dropout": 0.0, "use_nll_loss": _use_nll}
        for _cli, _arg in (
            (args.d_model,            "d_model"),
            (args.enc_heads,          "enc_heads"),
            (args.enc_layers,         "enc_layers"),
            (args.dec_heads,          "dec_heads"),
            (args.dec_layers,         "dec_layers"),
            (args.mlp_ratio,          "mlp_ratio"),
            (args.mask_ratio,         "mask_ratio"),
            (args.factorised_encoder, "factorised_encoder"),
            (args.temporal_window,    "temporal_window"),
            (args.temporal_patch,     "temporal_patch"),
            (args.cross_attn_decoder, "cross_attn_decoder"),
        ):
            if _cli is not None:
                _overrides[StationMAE._CFG_TO_ARG.get(_arg, _arg)] = _cli
        model = StationMAE.from_cfg(saved_cfg_for_build, **_overrides)
        try:
            model = model.to(device)
        except RuntimeError as _e:
            # "CUDA driver error: operation not supported" here is almost always
            # the allocator, not the model: expandable_segments:True puts the
            # caching allocator on the CUDA virtual-memory API, which several
            # vGPU profiles (e.g. A10-8Q) do not implement. It surfaces on the
            # first host->device copy, well after torch.cuda.is_available()
            # returned True. Name the fix rather than leaving a bare driver error.
            if "operation not supported" in str(_e) and device.type == "cuda":
                _alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
                raise RuntimeError(
                    f"Moving the model to {device} failed with: {_e}\n\n"
                    f"PYTORCH_CUDA_ALLOC_CONF={_alloc}\n"
                    "If that contains expandable_segments, this GPU very likely "
                    "does not support the CUDA virtual-memory API it relies on "
                    "(common on vGPU profiles). Re-run with:\n"
                    "    EXPANDABLE_SEGMENTS=0 bash src/scripts/run_test_cloud.sh\n"
                    "or unset PYTORCH_CUDA_ALLOC_CONF. Training is unaffected "
                    "because run_full_cloud.sh never sets it."
                ) from _e
            raise

        # ── Legacy-checkpoint compatibility: un-share what this checkpoint
        # never shared ────────────────────────────────────────────────────
        # Rather than refusing to load a pre-shared-embedding checkpoint,
        # give the DECODER its own fresh, separate copy of each mismatched
        # module. Both `encoder.<name>.*` and `decoder.<name>.*` keys then
        # map to genuinely distinct nn.Parameters, so load_state_dict below
        # populates each from the checkpoint's own two independently-trained
        # copies instead of colliding. Exact shapes are guaranteed to match:
        # d_model comes from this same checkpoint's cfg, and the Fourier
        # dims (position/station/temporal) are architecture constants that
        # have not changed — only TemporalEmbedding's WAVELENGTH VALUES have,
        # and those live in its `lambdas` buffer, which load_state_dict
        # overwrites from the checkpoint just like any other tensor, so the
        # freshly-built instance ends up with the checkpoint's original
        # wavelengths, not today's defaults.
        if _shared_mismatch:
            from model.embeddings import (
                PositionalEmbedding, StationEmbedding, TemporalEmbedding,
                POSITION_FOURIER_DIM, STATION_CHAR_DIM, TEMPORAL_FOURIER_DIM,
            )
            print(f"  [compat] legacy checkpoint predates shared embeddings "
                  f"{_shared_mismatch} — rebuilding the decoder's copies as "
                  f"UNSHARED so both trained versions load intact.")
            if "pos_emb" in _shared_mismatch:
                model.decoder.pos_emb = PositionalEmbedding(
                    d_model=c_d_model, fourier_dim=POSITION_FOURIER_DIM).to(device)
            if "station_emb" in _shared_mismatch:
                model.decoder.station_emb = StationEmbedding(
                    d_model=c_d_model, input_dim=STATION_CHAR_DIM).to(device)
            if "temporal_emb" in _shared_mismatch:
                model.decoder.temporal_emb = TemporalEmbedding(
                    d_model=c_d_model, fourier_dim=TEMPORAL_FOURIER_DIM).to(device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (using default init): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")

        # ── Architecture mismatch guard ───────────────────────────────────
        # strict=False is deliberate (it lets an NLL checkpoint load into a
        # point-prediction model and vice versa), but it will just as happily
        # drop a whole STRUCTURAL module. If the checkpoint was trained with
        # temporal patching and cfg failed to carry temporal_patch through,
        # the rebuilt model has no patch_merge: every patch_merge weight lands
        # in `unexpected`, the encoder silently runs unpatched on 6x the
        # tokens, and the predictions are garbage produced very slowly.
        # That is a wrong result, not just a slow one, so refuse to continue.
        _structural = ("patch_merge", "patch_norm", "var_proj",
                       "input_cross_attn",   # <-- the one that was silently dropped
                       "encoder.blocks", "decoder.blocks")
        # decoder.anchor_norm.* appears in every checkpoint trained before the
        # v23 query-anchor option was removed. The module no longer exists, so
        # the keys land in `unexpected` and are ignored — expected, not a fault.
        _known_stale = ("decoder.anchor_norm",)
        unexpected = [k for k in unexpected
                      if not any(s in k for s in _known_stale)]
        _bad = [k for k in list(missing) + list(unexpected)
                if any(s in k for s in _structural)]
        if _bad:
            raise SystemExit(
                "\n[ABORT] structural weights did not match the checkpoint:\n"
                + "".join(f"    {k}\n" for k in _bad[:12])
                + (f"    ... and {len(_bad)-12} more\n" if len(_bad) > 12 else "")
                + "\n  The rebuilt architecture differs from the trained one — most\n"
                  "  likely temporal_patch / temporal_window / d_model / layer counts\n"
                  "  were not read from the checkpoint cfg. Check the banner above\n"
                  "  against the training run, or pass the values explicitly.\n"
                  "  Evaluating anyway would produce meaningless predictions."
            )
        print(f"  Parameters: {model.count_parameters():,}")

        import time as _time

        base_label = os.path.splitext(os.path.basename(ckpt_path))[0]

        # ── Determine which mask ratios to sweep ─────────────────────────
        # Default: use the trained mask_ratio from the checkpoint.
        # CLI --test_mask_ratios overrides, enabling e.g. 0.0 vs 0.5 comparison.
        _test_mrs = (args.test_mask_ratios if args.test_mask_ratios
                     else [model.mask_ratio])

        # Same restriction, different cause: --station_local_decoder folds the
        # station axis into the batch, so the decoder needs the encoder to
        # return tokens for EVERY station. StationMAE asserts this at
        # construction, but that assertion cannot fire here — the sweep sets
        # model.encoder.mask_ratio after the model is built. Without this drop
        # the run reaches the forward-time guard and dies mid-dump, after the
        # dataset build and possibly after an earlier ratio has been written.
        if getattr(model, "station_local_decoder", False):
            _dropped = [mr for mr in _test_mrs if mr > 0.0]
            _test_mrs = [mr for mr in _test_mrs if mr == 0.0] or [0.0]
            if _dropped:
                print(f"  [station_local_decoder] skipping mask ratios "
                      f"{_dropped} — every station must contribute encoder "
                      f"tokens. Evaluating {_test_mrs} only.")

        for _test_mr in _test_mrs:
            # Override the encoder mask ratio for this evaluation pass
            model.encoder.mask_ratio = _test_mr
            model.eval()

            # ── Fix the station mask ────────────────────────────────────────
            # _mask_stations() draws torch.rand(B, N) from the GLOBAL RNG on
            # every forward pass, INCLUDING at mask_ratio 0 (the draw happens
            # before num_masked is applied). Two consequences, both handled by
            # re-seeding here rather than once at start-up:
            #
            #   1. Without a seed, two evaluation runs hide different stations,
            #      so masked-station errors cannot be compared pairwise across
            #      models. Measured previously: 0 of 11,684 windows shared a
            #      masked set between two runs, mean overlap 49.5% (chance).
            #   2. Because the mask_ratio 0 pass still consumes randomness,
            #      seeding only once would make the mr0.50 mask depend on
            #      whether mr0.00 ran first. Re-seeding per ratio makes each
            #      pass independent of the others, so `--test_mask_ratios 0.5`
            #      and `--test_mask_ratios 0.0 0.5` produce the same mr0.50
            #      masked set.
            #
            # Reproducibility still requires the same --batch_size,
            # --index_mode and --stride: those change the shape and number of
            # draws, hence the sequence of masks.
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            print(f"  [seed] station mask seeded with {args.seed} "
                  f"(reproducible at batch_size={args.batch_size}, "
                  f"index_mode={args.index_mode}, stride={args.stride})")

            _mr_tag   = f"mr{_test_mr:.2f}"
            label     = f"{base_label}_{_mr_tag}"
            mr_save   = os.path.join(args.save_dir, label)
            os.makedirs(mr_save, exist_ok=True)

            print(f"\n{'─'*60}")
            print(f"  mask_ratio={_test_mr:.2f}  "
                  f"({'trained setting' if _test_mr == c_mask_ratio else 'override'})  "
                  f"→ save_dir: {mr_save}")

            # ── One forward pass over the test windows → predictions.pt ──
            from engine.evaluate import collect_predictions
            t0 = _time.time()
            _n = args.save_predictions if args.save_predictions > 0 else len(test_ds)
            print(f"  [dump] Writing {_n:,} windows "
                  f"(index_mode={args.index_mode}, stride={args.stride}) …")
            _pred_path = os.path.join(mr_save, "predictions.pt")
            _res = collect_predictions(
                model, test_loader, device,
                n_windows=_n,
                save_path=_pred_path,
            )
            print(f"  ⏱  {_time.time()-t0:.0f}s")
            # Report the uncertainty explicitly. It is stored only when the
            # checkpoint carries decoder.log_var_head, so a silent absence
            # would otherwise be indistinguishable from a point-prediction
            # model — and the NLL run exists precisely for this output.
            if "log_var" in _res:
                _lv = _res["log_var"]
                print(f"  [uncertainty] log_var saved: {tuple(_lv.shape)} "
                      f"(M, K, N, V)")
                # These are display-only statistics; predictions.pt is already
                # on disk by this point. Never let them abort the mask-ratio
                # sweep — a full dump costs ~10 min of GPU time, and losing the
                # next ratio to a formatting bug is not an acceptable trade.
                try:
                    _sig = torch.exp(0.5 * _lv.float()).flatten()
                    # torch.quantile() rejects inputs above 2**24 elements
                    # ("input tensor is too large"); a full dump is
                    # M*K*N*V ~ 1.2e8. median() has no such cap, which is why
                    # only the percentiles used to fail. Subsample first.
                    _MAXQ = 1 << 20
                    _note = ""
                    if _sig.numel() > _MAXQ:
                        # Random, not strided: the tensor is laid out
                        # (M, K, N, V), so a fixed stride can alias with the
                        # variable or station period and sample one variable
                        # far more often than the others. Draw from a LOCAL
                        # generator — touching the global RNG here would shift
                        # the station mask and break cross-model pairing.
                        _g = torch.Generator().manual_seed(0)
                        _idx = torch.randint(0, _sig.numel(), (_MAXQ,),
                                             generator=_g)
                        _sig = _sig[_idx]
                        _note = f"   [{_MAXQ:,}-value subsample]"
                    print(f"                sigma = exp(0.5*log_var), "
                          f"normalised units: median {_sig.median():.4f}, "
                          f"p05 {_sig.quantile(0.05):.4f}, "
                          f"p95 {_sig.quantile(0.95):.4f}{_note}")
                except Exception as _e:
                    print(f"                [warn] sigma summary failed "
                          f"({type(_e).__name__}: {_e}). predictions.pt is "
                          f"already written — compute sigma downstream.")
            elif _use_nll:
                print("  [uncertainty] ⚠ checkpoint has a sigma head but no "
                      "log_var was returned — investigate before using it.")
            else:
                print("  [uncertainty] none (point-prediction checkpoint)")
            _prediction_sanity_check(_res, label)
            del _res

    print(f"\nDone — predictions.pt written per mask ratio under: "
          f"{os.path.abspath(args.save_dir)}")
    print("Compute metrics downstream in Test_Results_Exploration.ipynb.")


if __name__ == "__main__":
    main()
