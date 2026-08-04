"""
model/token_balance.py — representation sanity check for the encoder token.

The encoder token is the SUM of five branches:

    z = var_proj(x, mask)      what was measured   <- the only branch that sees data
      + pos_emb(coords)        where  (position)
      + station_emb(topo)      where  (terrain)
      + temporal_emb(hours)    when   (absolute)
      + step_emb(step_index)   when   (position in window)

Four of the five are a deterministic function of the INDEX — which station,
which timestep.  Only var_proj sees an observation.  Nothing in the
architecture guarantees the data branch arrives at a comparable scale, and if
it does not, the encoder is reading metadata with a trace of weather on top.

This module measures that directly.  It is NOT a training metric — it is an
initialisation invariant, and it is cheap enough to print at every startup.

Two numbers
-----------
content_fraction
    Var(obs) / Var(total), with the variance taken ACROSS TOKENS and summed
    over d_model.  Taking it across tokens is what makes it meaningful: a
    branch that is constant for every token contributes large norm but zero
    variance, and is correctly scored as carrying no information.

    With five summed branches, PARITY IS 0.20 (each contributing a fifth) —
    not 0.5.  A healthy value is therefore ~0.15-0.35, not "as high as
    possible".

permutation battery
    Permute one input factor across stations (or across time), leave the rest
    untouched, and measure how far the LayerNormed token moves — i.e. what the
    first transformer block actually sees.  The invariant:

        a weather token should move about as much when the weather changes
        as when its metadata changes.

Reference values at fresh initialisation, synthetic N(0,1) observations,
d_model=384, V=6, stable to +/-0.3% across seeds:

                        content fraction    weather permutation
    before the rebalance        0.07%              0.032
    after  the rebalance       22.1%               0.60
    (metadata branches)            -               0.57-0.64

Usage
-----
    from model.token_balance import token_balance, format_report
    print(format_report(token_balance(model.encoder, **synthetic_batch())))
"""

from __future__ import annotations

import torch

__all__ = ["token_components", "token_balance", "format_report",
           "synthetic_batch", "CONTENT_FRACTION_FLOOR", "WEATHER_PERM_FLOOR"]


# Acceptance thresholds.  Deliberately well clear of BOTH observed regimes
# rather than tight around the healthy one: this guards an invariant, it does
# not track a number.  Parity is 0.20 / ~0.60; the broken regime is 0.0007 /
# 0.03, so these sit roughly 3x below healthy and 100x above broken.
CONTENT_FRACTION_FLOOR = 0.05
WEATHER_PERM_FLOOR     = 0.25


def synthetic_batch(batch: int = 2, window: int = 72, stations: int = 155,
                    num_vars: int = 6, seed: int = 0, device=None) -> dict:
    """
    A deterministic stand-in for a real batch, so the check runs with no dataset.

    Per-station normalised observations are ~N(0,1) by construction, and real
    sensor availability is ~94%, which is what is reproduced here.
    """
    g = torch.Generator().manual_seed(seed)
    x    = torch.randn(batch, window, stations, num_vars, generator=g)
    mask = (torch.rand(batch, window, stations, num_vars, generator=g) > 0.064).float()
    return {
        "x":       (x * mask).to(device) if device else x * mask,
        "x_mask":  mask.to(device) if device else mask,
        "spatial": (torch.randn(stations, 15, generator=g).to(device) if device
                    else torch.randn(stations, 15, generator=g)),
        "x_hours": ((torch.rand(batch, window, generator=g) * 1e3 + 4e5).to(device) if device
                    else torch.rand(batch, window, generator=g) * 1e3 + 4e5),
    }


@torch.no_grad()
def token_components(encoder, x, x_mask, spatial, x_hours) -> dict:
    """
    The five branches separately, each broadcast to (B, W, N, d_model).

    Mirrors StationMAEEncoder._build_tokens.  The caller-visible guarantee is
    enforced, not assumed: the sum of what this returns is checked against
    encoder._build_tokens() and an AssertionError is raised on any drift.  That
    makes it safe to leave this decomposition outside the encoder (no encoder
    changes) without it silently rotting when _build_tokens is edited.
    """
    B, W, N, V = x.shape
    d = encoder.d_model

    obs = encoder.var_proj(x.reshape(B * W, N, V),
                           x_mask.reshape(B * W, N, V)).reshape(B, W, N, d)
    sp  = spatial.unsqueeze(0) if spatial.dim() == 2 else spatial
    pos = encoder.pos_emb(sp[..., :2]).unsqueeze(1)
    sta = encoder.station_emb(sp[..., 2:]).unsqueeze(1)
    tem = encoder.temporal_emb(x_hours).unsqueeze(2)
    stp = encoder.step_emb(torch.arange(W, device=x.device)).view(1, W, 1, d)

    comps = {
        "obs (CONTENT)": obs,
        "position p1":   pos.expand(B, W, N, d),
        "station p2":    sta.expand(B, W, N, d),
        "temporal t":    tem.expand(B, W, N, d),
        "step s":        stp.expand(B, W, N, d),
    }

    # Guard against drift from the encoder's own token construction.
    ref  = encoder._build_tokens(x, x_mask, spatial, x_hours)
    mine = encoder.token_norm(sum(comps.values()))
    if not torch.allclose(ref, mine, atol=1e-4, rtol=1e-3):
        raise AssertionError(
            "token_balance.token_components has drifted from "
            "StationMAEEncoder._build_tokens (max abs diff "
            f"{(ref - mine).abs().max().item():.3e}). Update this module.")
    return comps


@torch.no_grad()
def token_balance(encoder, x, x_mask, spatial, x_hours, seed: int = 0) -> dict:
    """Content fraction, per-branch scales, and the permutation battery."""
    comps = token_components(encoder, x, x_mask, spatial, x_hours)
    d     = encoder.d_model
    B, W, N, _ = x.shape

    var = {k: v.reshape(-1, d).var(0).sum().item() for k, v in comps.items()}
    rms = {k: v.pow(2).mean(-1).sqrt().mean().item() for k, v in comps.items()}
    content_fraction = var["obs (CONTENT)"] / max(sum(var.values()), 1e-12)

    # Split the observation branch into its token-independent and data-dependent
    # halves: var_proj evaluated at zero observations with every sensor present.
    zero  = torch.zeros(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
    const = encoder.var_proj(zero, torch.ones_like(zero))
    weather_rms  = (comps["obs (CONTENT)"] - const).pow(2).mean().sqrt().item()
    constant_rms = const.pow(2).mean().sqrt().item()

    g    = torch.Generator().manual_seed(seed)
    base = encoder.token_norm(sum(comps.values()))
    perm = {}
    for what in ("weather", "station", "position", "time"):
        xx, mm, ss, hh = x, x_mask, spatial.clone(), x_hours
        if what == "weather":
            pi = torch.randperm(N, generator=g)
            xx, mm = x[:, :, pi], x_mask[:, :, pi]
        elif what == "station":
            pi = torch.randperm(N, generator=g)
            ss[..., 2:] = spatial[pi, 2:] if spatial.dim() == 2 else spatial[:, pi, 2:]
        elif what == "position":
            pi = torch.randperm(N, generator=g)
            ss[..., :2] = spatial[pi, :2] if spatial.dim() == 2 else spatial[:, pi, :2]
        else:
            hh = x_hours[:, torch.randperm(W, generator=g)]
        moved = encoder.token_norm(
            sum(token_components(encoder, xx, mm, ss, hh).values()))
        perm[what] = ((moved - base).norm() / base.norm()).item()

    meta = sum(perm[k] for k in ("station", "position", "time")) / 3.0
    return {
        "variance": var, "rms": rms,
        "content_fraction": content_fraction,
        "constant_rms": constant_rms, "weather_rms": weather_rms,
        "permutation": perm,
        "weather_over_metadata": perm["weather"] / max(meta, 1e-12),
        "passes": (content_fraction >= CONTENT_FRACTION_FLOOR
                   and perm["weather"] >= WEATHER_PERM_FLOOR),
    }


def format_report(r: dict) -> str:
    """One compact block, suitable for a training log."""
    L = ["", "token balance (initialisation invariant — not a training metric)",
         f"  {'branch':18s} {'RMS/token':>10s} {'var across tokens':>18s}"]
    for k in r["variance"]:
        L.append(f"    {k:16s} {r['rms'][k]:10.4f} {r['variance'][k]:18.3f}")
    L += [f"  content fraction {r['content_fraction']*100:6.2f}%"
          f"   (parity for 5 branches = 20.00%, floor = "
          f"{CONTENT_FRACTION_FLOOR*100:.2f}%)",
          f"    observation branch: constant RMS {r['constant_rms']:.4f}"
          f" | weather RMS {r['weather_rms']:.4f}",
          "  permutation  " + "  ".join(f"{k} {v:.4f}" for k, v in r["permutation"].items()),
          f"    weather / metadata = {r['weather_over_metadata']:.3f}",
          f"  -> {'OK' if r['passes'] else 'DEGRADED — the token barely encodes the weather'}",
          ""]
    return "\n".join(L)
