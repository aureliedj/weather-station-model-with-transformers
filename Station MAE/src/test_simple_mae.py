"""
test_simple_mae.py — dump simple-MAE test predictions in the shared schema.

Two evaluation scenarios, mirroring the transformer's directory layout so the
run drops straight into Test_Results_Exploration.ipynb:

    best_mr0.00/predictions.pt   FORECAST: all stations visible for the 12 h
                                 context, the last 6 h fully masked.
    best_mr0.50/predictions.pt   OUTAGE:  50% of stations hidden for the WHOLE
                                 window (their 18 h) + the future block masked
                                 for everyone. masked_idx records the hidden
                                 stations (gap-filling analysis downstream).

Schema (identical to test.py / test_lstm.py):
    preds (M, K, N, 5) · targets (M, K, N, 6) · masks (M, K, N, 6)
    masked_idx (M, N_masked) · delta_steps (M, K) · window_hours (M,)
    target_hours (M, K) · spatial (N, 15) · var_names · n_windows
with K = 13 leads [0, 3, …, 36] steps (0 … 6 h at 30-min spacing), selected
from the model's per-10-min reconstruction of the future block.

Δ=0 caveat: the Δ=0 slot is the LAST CONTEXT step. For VISIBLE stations that
token is unmasked, and the loss never supervises visible context tokens — so
the model's Δ=0 output is untrained there. It is dumped as-is (do not read
Δ=0 as a gap-filling score for visible stations; for the mr0.50 scenario's
HIDDEN stations, Δ=0 IS the trained gap-filling output).

Usage:
    python test_simple_mae.py --data_root /path --checkpoint .../best.ckpt
See run_simple_mae_test_cloud.sh for the cloud configuration.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple-MAE test prediction dump")
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--cache_dir",   type=str, default=None)
    p.add_argument("--local_cache_dir", type=str, default="/tmp/station_mae_cache")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--exclude_stations", type=str, nargs="+", default=None)
    p.add_argument("--index_mode",  type=str, default="sliding",
                   choices=["sliding", "blocks"])
    p.add_argument("--stride",      type=int, default=9,
                   help="Sliding stride in 10-min steps (9 = 90 min, rolling origin)")
    p.add_argument("--max_windows", type=int, default=0, help="0 = ALL windows")
    p.add_argument("--station_ratio", type=float, default=0.5,
                   help="Hidden-station fraction for the mr0.50 scenario")
    p.add_argument("--seed",        type=int, default=42,
                   help="Seed for the mr0.50 station masks (reproducible)")
    p.add_argument("--save_dir",    type=str, default="test_results")
    p.add_argument("--run_name",    type=str, default="simple-mae-v1")
    p.add_argument("--global_norm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Checkpoint → model (strict load: any mismatch must abort) ────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt["hyper_parameters"]["cfg"]
    from model.simple_mae import SimpleMAE
    model = SimpleMAE(
        window_steps=cfg["window"], patch_steps=cfg["patch_steps"],
        d_model=cfg["d_model"], enc_layers=cfg["enc_layers"],
        enc_heads=cfg["enc_heads"], dec_dim=cfg["dec_dim"],
        dec_layers=cfg["dec_layers"], dec_heads=cfg["dec_heads"],
        dropout=0.0, mask_ratio=cfg["mask_ratio"],
        station_ratio=cfg["station_ratio"], future_slots=cfg["future_slots"],
        strategy_probs=tuple(cfg["strategy_probs"]),
        fuse_layers=cfg.get("fuse_layers", 0),
    )
    sd = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)   # simple model → strict, no surprises
    model = model.to(device).eval()
    print(f"Loaded SimpleMAE: {model.count_parameters():,} params "
          f"(epoch {ckpt.get('epoch', '?')})")

    # ── Data (same conventions as everywhere else) ───────────────────────
    from data.dataset import load_peakweather, StationMAEDataset
    ds = load_peakweather(root=args.data_root)
    common = dict(window_size=cfg["window"], delta_steps=0, max_delta_steps=0,
                  cache_dir=args.cache_dir or args.data_root,
                  shared_memory=False,
                  fast_cache_dir=args.local_cache_dir.strip() or None,
                  exclude_stations=args.exclude_stations,
                  delta_mode="fixed_grid", delta_grid_stride=1,
                  global_norm=args.global_norm)
    train_ds = StationMAEDataset(ds, split="train", **common)      # stats only
    test_ds  = StationMAEDataset(ds, split="test",
                                 obs_stats=train_ds.obs_stats,
                                 index_mode=args.index_mode,
                                 train_stride=args.stride, **common)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"Test windows: {len(test_ds):,} "
          f"({args.index_mode}, stride={args.stride})")

    S, P = model.S, model.P
    W    = cfg["window"]
    f0   = W - model.future_slots * P            # first future step
    LEADS = list(range(0, model.future_slots * P + 1, 3))   # [0,3,…,36]
    steps = [f0 - 1] + [f0 - 1 + d for d in LEADS[1:]]      # step indices per lead
    print(f"Leads (10-min steps): {LEADS} → window steps {steps}")

    limit = args.max_windows if args.max_windows > 0 else float("inf")
    gen   = torch.Generator().manual_seed(args.seed)

    def run_scenario(tag: str, hide_stations: bool) -> None:
        keep = {k: [] for k in ("preds", "targets", "masks", "masked_idx",
                                "delta_steps", "window_hours", "target_hours")}
        spatial_saved = None
        collected = 0
        try:
            from tqdm import tqdm
            _iter = tqdm(loader, desc=f"predict {tag}", unit="batch")
        except ImportError:
            _iter = loader
        with torch.no_grad():
            for batch in _iter:
                if collected >= limit:
                    break
                x  = batch["x"].to(device)
                xm = batch["x_mask"].to(device)
                sp = batch["spatial"]
                sp = (sp[0] if sp.dim() == 3 else sp).to(device)
                xh = batch["x_hours"].to(device)
                B, _, N, _ = x.shape

                # scenario mask: future block always; + hidden stations
                tmask = torch.zeros(B, S, N, dtype=torch.bool, device=device)
                tmask[:, S - model.future_slots:, :] = True
                if hide_stations:
                    n_hide = int(N * args.station_ratio)
                    order  = torch.rand(B, N, generator=gen).argsort(dim=1)
                    midx   = order[:, :n_hide].to(device)          # (B, n_hide)
                    sel    = torch.zeros(B, N, dtype=torch.bool, device=device)
                    sel.scatter_(1, midx, True)
                    tmask |= sel.unsqueeze(1)
                else:
                    midx = torch.empty(B, 0, dtype=torch.long, device=device)

                _amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" else torch.autocast("cpu", enabled=False))
                with _amp:
                    _, preds, _, _ = model(x, xm, sp, xh, token_mask=tmask)
                preds = preds.float()                              # (B, W, N, 5)

                take = B if limit == float("inf") else min(B, int(limit - collected))
                # .clone(): break views into worker shared-memory tensors
                keep["preds"].append(preds[:take][:, steps].cpu().clone())
                keep["targets"].append(x[:take][:, steps].cpu().clone())
                keep["masks"].append(xm[:take][:, steps].cpu().clone())
                keep["masked_idx"].append(midx[:take].cpu().clone())
                keep["delta_steps"].append(
                    torch.tensor(LEADS).unsqueeze(0).expand(take, -1).clone())
                keep["window_hours"].append(xh[:take, 0].cpu().clone())
                keep["target_hours"].append(xh[:take][:, steps].cpu().clone())
                if spatial_saved is None:
                    spatial_saved = sp.cpu().clone()
                collected += take

        result = {k: torch.cat(v, dim=0) for k, v in keep.items()}
        result["spatial"]   = spatial_saved
        result["var_names"] = ["temperature", "pressure", "humidity",
                               "wind_u", "wind_v"]
        result["n_windows"] = collected
        out_dir = os.path.join(args.save_dir, args.run_name, tag)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "predictions.pt")
        torch.save(result, path)
        print(f"[{tag}] saved {collected:,} windows → {path} "
              f"({os.path.getsize(path)/1e6:.0f} MB)")

        # ── sanity: persistence skill on Δ>0 + dispersion ────────────────
        Pn = result["preds"].numpy()
        Tn = result["targets"].numpy()[..., :5]
        Mn = result["masks"].numpy()[..., :5] > 0.5
        K  = Pn.shape[1]
        persist = np.repeat(Tn[:, :1], K, axis=1)
        both = Mn & Mn[:, :1]
        e  = (Pn[:, 1:] - Tn[:, 1:])[both[:, 1:]]
        ep = (persist[:, 1:] - Tn[:, 1:])[both[:, 1:]]
        rm, rp = float(np.sqrt(np.mean(e**2))), float(np.sqrt(np.mean(ep**2)))
        skill = 1 - rm / rp if rp > 0 else float("nan")
        disp  = min(float(Pn[:, 1:, :, v][Mn[:, 1:, :, v]].std()
                          / max(Tn[:, 1:, :, v][Mn[:, 1:, :, v]].std(), 1e-6))
                    for v in range(5))
        print(f"[{tag}] sanity — Δ>0 norm RMSE {rm:.4f} vs persistence {rp:.4f} "
              f"→ skill {skill:+.3f} | dispersion_min {disp:.2f}")
        print(("[{}] ✓ looks sane" if skill > -0.5 and disp > 0.3 else
               "[{}] ⚠ check the model/rebuild before trusting this dump")
              .format(tag))

    run_scenario("best_mr0.00", hide_stations=False)
    run_scenario("best_mr0.50", hide_stations=True)
    print("\nDone. Add the run to CMP_RUNS in Test_Results_Exploration.ipynb.")
    print("Reminder: Δ=0 predictions are untrained for VISIBLE stations "
          "(see module docstring).")


if __name__ == "__main__":
    main()
