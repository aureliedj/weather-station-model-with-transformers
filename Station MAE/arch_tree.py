#!/usr/bin/env python3
"""
arch_tree.py — Station-MAE architecture tree with live tensor shapes.

Run from the "Station MAE/" directory (no GPU needed):

  # Current production config (tw12-d1024-v2, run_full_cloud.sh):
    python arch_tree.py

  # Override specific settings:
    python arch_tree.py --window 72 --stations 20 --d_model 64
    python arch_tree.py --joint                  # joint spatiotemporal encoder
    python arch_tree.py --factorised             # factorised axial encoder
    python arch_tree.py --flat --temporal_window 0  # full flat (no windowing)

Defaults reflect the current run_full_cloud.sh configuration:
    flat encoder + temporal_window=12, d_model=1024, enc_layers=6, dec_layers=4,
    cross-attn decoder, window=72, stations=155, mask_ratio=0.5
"""
import argparse
import torch

# ── CLI ────────────────────────────────────────────────────────────────────────

p = argparse.ArgumentParser()
p.add_argument("--joint",            action="store_true", help="Joint spatiotemporal encoder (RoPE)")
p.add_argument("--factorised",       action="store_true", help="Factorised axial encoder")
p.add_argument("--flat",             action="store_true", help="Full flat encoder (no windowing)")
p.add_argument("--concat_dec",       action="store_true", help="Concatenated self-attn decoder (default: cross-attn)")
p.add_argument("--window",           type=int,   default=72,   help="W: input timesteps (default 72 = 12h)")
p.add_argument("--stations",         type=int,   default=155,  help="N: stations after PFA exclusion (default 155)")
p.add_argument("--d_model",          type=int,   default=1024, help="d_model (default 1024)")
p.add_argument("--enc_layers",       type=int,   default=6)
p.add_argument("--dec_layers",       type=int,   default=4)
p.add_argument("--mask_ratio",       type=float, default=0.5)
p.add_argument("--temporal_window",  type=int,   default=12,   help="Temporal chunk size (default 12 = 2h chunks)")
args = p.parse_args()

B, W, N, V = 2, args.window, args.stations, 6
D          = args.d_model
N_vis      = N - round(N * args.mask_ratio)
N_masked   = N - N_vis

# ── Model ──────────────────────────────────────────────────────────────────────

from model.mae import StationMAE

# Derive encoder mode from flags:
# --joint      → joint spatiotemporal + RoPE
# --factorised → factorised axial (temporal + spatial sub-layers)
# --flat       → full flat (no windowing, overrides temporal_window)
# default      → flat + temporal_window=12 (windowed flat, current production config)
_is_flat        = not args.joint and not args.factorised
_temporal_win   = 0 if args.flat else args.temporal_window
_num_heads      = max(2, D // 64)   # scale heads with d_model (1 head per 64 dims)

model = StationMAE(
    d_model                 = D,
    enc_heads               = _num_heads,
    enc_layers              = args.enc_layers,
    dec_heads               = _num_heads,
    dec_layers              = args.dec_layers,
    mlp_ratio               = 4.0,
    dropout                 = 0.0,
    mask_ratio              = args.mask_ratio,
    factorised_encoder      = args.factorised,
    joint_encoder           = args.joint,
    temporal_window         = _temporal_win,
    cross_attention_decoder = not args.concat_dec,
).eval()

if args.joint:
    enc_mode = "joint (RoPE)"
elif args.factorised:
    enc_mode = f"factorised" + (f" tw={_temporal_win}" if _temporal_win else "")
elif _temporal_win:
    enc_mode = f"flat windowed (tw={_temporal_win})"
else:
    enc_mode = "flat (full attention)"
dec_mode = "concat self-attn" if args.concat_dec else "cross-attn"

# ── Forward hooks ──────────────────────────────────────────────────────────────

_records: dict = {}

def _shape(t):
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    if isinstance(t, (tuple, list)):
        s = [_shape(x) for x in t if isinstance(x, torch.Tensor)]
        return s[0] if len(s) == 1 else s
    return None

def _hook(key):
    def fn(mod, inp, out):
        _records[key] = (
            _shape(inp[0]) if inp else None,
            _shape(inp[1]) if len(inp) > 1 and isinstance(inp[1], torch.Tensor) else None,
            _shape(out),
        )
    return fn

enc, dec = model.encoder, model.decoder
_hooks = []
for key, mod in (
    [("var_proj",  enc.var_proj),
     ("pos_emb",   enc.pos_emb),
     ("stn_emb",   enc.station_emb),
     ("temp_emb",  enc.temporal_emb),
     ("tok_norm",  enc.token_norm),
     ("enc_norm",  enc.norm)]
    + [(f"enc[{i}]", b) for i, b in enumerate(enc.blocks)]
    + [(f"dec[{i}]", b) for i, b in enumerate(dec.blocks)]
    + [("dec_norm",  dec.norm),
       ("head",      dec.head)]
):
    _hooks.append(mod.register_forward_hook(_hook(key)))

# ── Run ────────────────────────────────────────────────────────────────────────

with torch.no_grad():
    loss, preds, midx = model(
        torch.randn(B, W, N, V),          # x
        torch.ones(B, W, N, V),           # x_mask
        torch.randn(N, 15),               # spatial
        torch.zeros(B, W),                # x_hours
        torch.randn(B, N, V),             # y
        torch.ones(B, N, V),              # y_mask
        torch.zeros(B),                   # y_hours
        torch.ones(B, dtype=torch.long),  # delta_steps
    )

for h in _hooks:
    h.remove()

r = _records  # shorthand

# ── Formatting helpers ──────────────────────────────────────────────────────────

L  = 78   # total line width
NL = 22   # name column
SL = 28   # shape column

def row(indent, name, in_shape, out_shape, note=""):
    pre  = "│  " * indent
    istr = str(in_shape)  if in_shape  else "—"
    ostr = str(out_shape) if out_shape else "—"
    note = f"  # {note}" if note else ""
    print(f"  {pre}{name:<{NL}}  {istr:<{SL}} →  {ostr}{note}")

def banner(title):
    print()
    print(f"  ┌{'─'*(L-2)}┐")
    print(f"  │  {title:<{L-4}}│")
    print(f"  └{'─'*(L-2)}┘")

# ── Header ─────────────────────────────────────────────────────────────────────

print()
print("═" * L)
print(f"  Station-MAE  |  encoder={enc_mode}  decoder={dec_mode}")
print(f"  B={B}  W={W}  N={N}  V={V}  D={D}  "
      f"mask={args.mask_ratio:.0%}  enc×{args.enc_layers}  dec×{args.dec_layers}")
print("═" * L)

# ── Inputs ─────────────────────────────────────────────────────────────────────

print()
print("  INPUTS")
for name, shape, note in [
    ("x",       (B, W, N, V), "observations (0.0 where sensor absent)"),
    ("x_mask",  (B, W, N, V), "1.0 = sensor present"),
    ("spatial", (N, 15),      "cols [:2]=easting/northing  [2:]=topo"),
    ("x_hours", (B, W),       "hours since 1970-01-01 UTC per step"),
]:
    print(f"     {name:<{NL}}  {str(shape):<{SL}}   {note}")

# ── Encoder ────────────────────────────────────────────────────────────────────

banner(f"ENCODER  [{enc_mode}]  →  encoded: (B, W·N_vis, D)")
print()
print(f"  │  Token assembly                               (B·W={B*W}, N={N})  →  (B, W, N, D)")
row(1, "var_proj",  r["var_proj"][0], r["var_proj"][2],  "what:    sensor values → D")
row(1, "pos_emb",   r["pos_emb"][0],  r["pos_emb"][2],   "where-1: easting/northing Fourier")
row(1, "stn_emb",   r["stn_emb"][0],  r["stn_emb"][2],   "where-2: topo characteristics MLP")
row(1, "temp_emb",  r["temp_emb"][0], r["temp_emb"][2],  "when:    hours → Aurora Fourier")
row(1, "tok_norm",  r["tok_norm"][0], r["tok_norm"][2],  "Σ four embeddings → LayerNorm")
print()
if _temporal_win and _is_flat:
    _chunks   = W // _temporal_win
    _chunk_L  = _temporal_win * N_vis
    print(f"  │  Temporal windowing  tw={_temporal_win} → {_chunks} chunks × {_chunk_L} tokens/chunk")
    print(f"  │    (vs full flat L={W*N_vis:,}  —  {(W*N_vis)**2/(_chunks*_chunk_L**2):.0f}× cheaper attention)")
print(f"  │  Station masking  (mask_ratio={args.mask_ratio:.0%})")
print(f"  │    visible  {(B, W, N_vis, D)}   N_vis={N_vis}")
print(f"  │    masked   {(B, N_masked)}        N_masked={N_masked}  (indices only, not processed)")
print()
print(f"  │  Transformer blocks  [{enc_mode}]")
for i in range(args.enc_layers):
    k = f"enc[{i}]"
    if k in r:
        row(1, k, r[k][0], r[k][2])
row(1, "enc_norm",  r["enc_norm"][0], r["enc_norm"][2],  "(B,W,N_vis,D) → flatten → (B,W·N_vis,D)")

enc_out = r["enc_norm"][2]

# ── Decoder ────────────────────────────────────────────────────────────────────

banner(f"DECODER  [{dec_mode}]  →  preds: (B, N, V_target=5)")
print()
if not args.concat_dec:
    print(f"  │  Queries  (B, N, D) = {(B, N, D)}")
    print(f"  │    mask_token  +  stn_emb  +  temp_emb(y_hours)  +  delta_emb(Δt)")
    print(f"  │  Context K/V from encoder  {enc_out}")
    print()
    print(f"  │  Transformer blocks  [self-attn on Q  +  cross-attn Q→encoder]")
    for i in range(args.dec_layers):
        k = f"dec[{i}]"
        if k in r:
            kv_note = f"kv={r[k][1]}" if r[k][1] else ""
            row(1, k, r[k][0], r[k][2], kv_note)
else:
    full_len = W * N_vis + N
    print(f"  │  Full sequence  [encoder tokens ‖ N station queries]")
    print(f"  │  (B, W·N_vis + N, D) = {(B, full_len, D)}")
    print()
    print(f"  │  Transformer blocks  [self-attn over full sequence]")
    for i in range(args.dec_layers):
        k = f"dec[{i}]"
        if k in r:
            row(1, k, r[k][0], r[k][2])
    print(f"  │  (last N tokens → head)")

row(1, "dec_norm",  r["dec_norm"][0], r["dec_norm"][2])
row(1, "head",      r["head"][0],     r["head"][2],      "Linear: D → V_target=5  (no precip)")

# ── Outputs ────────────────────────────────────────────────────────────────────

print()
print("  OUTPUTS")
for name, shape, note in [
    ("preds",      tuple(preds.shape),  "(B, N, V_target=5)  all stations, 5 target vars"),
    ("masked_idx", tuple(midx.shape),   "(B, N_masked)       which stations were hidden"),
    ("loss",       f"scalar={loss.item():.5f}", "MSE over sensor-present entries at target time"),
]:
    print(f"     {name:<{NL}}  {str(shape):<{SL}}   {note}")

# ── Summary ────────────────────────────────────────────────────────────────────

print()
print("─" * L)
n_tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_enc = sum(p.numel() for p in enc.parameters()   if p.requires_grad)
n_dec = sum(p.numel() for p in dec.parameters()   if p.requires_grad)
print(f"  Parameters   total={n_tot:>10,}   encoder={n_enc:>8,}   decoder={n_dec:>8,}")
print("─" * L)
print()
