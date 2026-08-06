#!/usr/bin/env bash
# run_full_cloud.sh — full training run, A100 80GB PCIe (MIG 3g.20gb)
#
# The configuration this script currently launches is set in the CONFIG block
# below; the wandb run name records which version it is.
#
# Change history (v12 -> v15) lives in report/run-script-history.md.
# Change tables:
#   report/v12-v15-v17-changes.md    v12 -> v15 -> v17
#   report/v17-v19-v20-changes.md    v17 -> v19 -> v20, with test results
#
# Hardware context
# ----------------
# The A100 runs in MIG mode. The allocated partition is 3g.20gb:
#   * 19,968 MiB (~20 GB) VRAM  — NOT the full 80 GB
#   * 28 / 108 streaming multiprocessors
#   * no NVLink across MIG instances -> single-GPU only
#
# Flags that exist for that constraint:
#   --amp --bf16        native BF16 tensor cores; no loss scaling
#   --compile           "default" mode — MIG restricts CUDA graph capture, so
#                       "reduce-overhead" would silently fail
#   --grad_checkpoint   ~33% extra compute for ~66% less activation memory
#   --local_cache_dir   split-normalised tensors as numpy memmaps on tmpfs;
#                       workers mmap from the OS page cache, no IPC for source data
#
# Resuming:
#   bash src/scripts/run_full_cloud.sh          auto-resumes from last.ckpt
#   RESUME=0 bash src/scripts/run_full_cloud.sh forces a fresh run
#
# Resume is explicit: main.py only resumes when --resume is passed. A plain
# re-run used to start from random init AND write into the same SAVE_DIR, which
# is what produced the best-v1.ckpt / last-v1.ckpt pairs in full_run_cloud_v12.
#
# Quick check before committing the full budget:
#   SANITY=1 bash src/scripts/run_full_cloud.sh   ~minutes
#   SANITY=2 bash src/scripts/run_full_cloud.sh   ~1 hour

set -euo pipefail

DATA_ROOT="/home/renku/work/PeakWeatherDataset"

# Checkpoints saved inside the project directory — guaranteed writable on Renku.
# Saving to /home/renku/work/ sub-directories outside this project can silently
# fail due to filesystem permissions on the Renku mount.
# Saving directly to Polybox during training is unreliable (network latency,
# lock contention) and has been removed — copy checkpoints to Polybox manually
# after training with:
#   cp checkpoints/full_run_cloud/best.ckpt /home/renku/work/polybox-capstone/checkpoints/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/scripts
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../src   — python entry points live here
PROJ_DIR="$(cd "${SRC_DIR}/.." && pwd)"                      # project root — checkpoints/, test_results/, report/
cd "${SRC_DIR}"                                              # so `python main.py` and `from data...` resolve
SAVE_DIR="${PROJ_DIR}/checkpoints/full_run_cloud_v25"   # own dir — never share a SAVE_DIR (that is what produced the -v1 checkpoints)
LOCAL_CACHE="/tmp/station_mae_cache"

# ── WandB ────────────────────────────────────────────────────────────────────
# Defaults to ONLINE and prompts for `wandb login` if no credential is present,
# so a run never silently produces an empty dashboard.
#
# Caveat kept from earlier: Renku egress can drop for minutes at a time, and the
# wandb server then marks the run "crashed" (frozen at the last epoch it
# received) even though training continues normally. If that happens, re-launch
# with WANDB_MODE=offline and sync afterwards:
#   WANDB_MODE=offline bash src/scripts/run_full_cloud.sh
#   wandb sync "$(ls -dt wandb/run-* | head -1)"
source "${SCRIPT_DIR}/_wandb_preflight.sh"

# ── Station exclusion ────────────────────────────────────────────────────────
# Drop stations with insufficient historical coverage before training.
# Matched case-insensitively against the stations_table index / name / abbr.
# Add more names separated by spaces: EXCLUDE="110 BAS KLO"
EXCLUDE="--exclude_stations PFA"   # station 110 (PFA) — insufficient historical coverage
# EXCLUDE="--exclude_stations 110"  # alternative: match by numeric ID
# EXCLUDE=""                         # ← uncomment to disable exclusion
# ─────────────────────────────────────────────────────────────────────────────

# ── Encoder architecture ──────────────────────────────────────────────────────
#
# Two modes:
#
#   FACTORISED  (--factorised_encoder)   <- what every run since v15 uses
#     Temporal attention, then spatial, then a shared FFN. ~100x cheaper than
#     flat. Spatial attention is FULL over the station axis; the windowing
#     option below applies to the temporal axis only.
#
#   FLAT  (no flag)
#     Standard self-attention over all W*N_vis tokens. Kept so pre-v15
#     checkpoints stay loadable.
#
#     --no_spatial_attn drops the spatial sub-layer entirely — the
#     station-independent controlled study. See the SPATIAL block below.
#
# REMOVED in the v22 cleanup: --joint_encoder (JointSpatioTemporalBlock, full
# W*N attention with temporal RoPE) — ~317 lines no configuration had selected
# since v14. git history has it.
# ── Temporal tokenization ────────────────────────────────────────────────────
# Uniform patching: P consecutive 10-min steps become ONE encoder token.
#   P=1  raw       72 positions -> 5,616 tokens  (~273 GFLOP/pass)  no loss, slow
#   P=3  30-min    24 positions -> 1,872 tokens  (~48 GFLOP)        <- current
#   P=6  hourly    12 positions ->   936 tokens  (~19 GFLOP)        = v14
# P=3 keeps a 30-min token, so the +30 min forecast is one token ahead rather
# than half a token as it was with v14's hourly patches — the "hourly patches
# starve the short lead" diagnosis — while costing ~6x less than raw.
# P must divide --window (72): 1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72.
#
# WHY NOT P=1 (raw): the sanity2 run at P=1 collapsed. val/horizon_sensitivity
# FELL 0.17 -> 0.06 (the metric's own "Delta-embedding unused" floor is 0.05)
# and sanity/dispersion_min sat at ~0.08 — the decoder emitted one nearly
# constant value regardless of station AND of lead time. The telescopic run at
# the SAME learning rate did not collapse, so this is not an lr problem.
#   Mechanism: softmax over 5,616 keys starts almost uniform (~1/5616 each),
#   so every output is a near-average of all values — "predict the mean of
#   everything" is the natural starting point, and a constant has no lead-time
#   dependence. At 1,872 tokens that dilution is 3x milder.
# If P=3 also flatlines, the next levers are lr (3e-4) and the paradigm itself.
# ── Observation encoder (v18) ────────────────────────────────────────────────
# v17 embeds each variable as  e_v = x_v * w_v + b_v : rank 1 and linear, so
# different VALUES differ only in magnitude along a fixed direction.
#
#   --value_embedding mlp       1 -> 32 -> GELU -> d, per variable. Each hidden
#       unit is a GELU threshold, so the 32 units form a piecewise basis over
#       the value axis. Measured (held-out R^2 of a linear readout at init):
#
#           target             v17 linear   mlp   fourier
#           |x|                    0.000   1.000    0.000
#           1[x > 1] threshold     0.499   0.950    0.000
#           x^2 / sin(3x)          0.000   1.000    1.000
#
#       It wins on kinks and thresholds, which is what meteorological structure
#       looks like: saturation, freezing, calm/gust.
#
#   --value_embedding fourier   PLR (Gorishniy et al. 2022). Beat 'mlp' only on
#       ANGULAR quantities, which are not represented here — u and v are
#       z-scored per component per station, so neither sqrt(u^2+v^2) nor
#       atan2(v,u) is physical in normalised space. Not recommended for now.
#
# Wind pairing (--wind_encoder) is available but OFF: a shared (u,v) map is
# station-blind, and the per-component per-station normalisation distorts
# speed and direction by a per-station amount it cannot undo.
#
# This is INDUCTIVE BIAS, not new capability: the values are linearly
# recoverable from the v17 token (probe R^2 >= 0.94), so the block-1 FFN can
# already form cross-variable functions. This makes them cheap, not possible.
OBS_ENCODER="--value_embedding mlp"

# ── Static features: separate branch, or inside the variable block? (v21) ────
# Default (empty) keeps v18: position and topography are their own additive
# branches, each with its own MLP whose output scale is set independently of
# the observations — the structural origin of the token imbalance.
#
# --static_in_token follows Aurora, which "incorporate[s] static variables
# (orography, land-sea mask, and soil-type mask) by treating them as extra
# surface-level variables". Terrain then goes through the SAME per-slot
# mechanism and the SAME /sqrt(S) divisor as the weather, so a mismatch cannot
# arise by construction.
#
# The grouping matters more than the idea. Measured weather share of the token
# at initialisation, real data:
#
#     one slot per static feature   21 slots    9.6%   <- WORSE than v18
#     position + topography          8 slots   25.1%   <- what this flag does
#     all 15 in one slot             7 slots   28.7%
#     v18 separate branches                    18.3%
#
# Aurora has ~3 statics against many variables; here 15 statics against 6
# weather variables, so slot-per-feature would drown the weather in its own block.
# STATIC="--static_in_token"                       # <- v21 statics as slots
STATIC=""                                          # <- v20 / v18 separate branches

# ── Persistence residual head (v20) ──────────────────────────────────────────
# y_hat = base + f(.)   where base = the station's own last observation if it
# was VISIBLE to the encoder, and 0 if it was MASKED.
#
# WHY. Measured on v15's own validation metrics, against what a perfect copy of
# the station's last value would score at +30 min:
#
#     variable      persist r   copy bound   v15 actual   ratio
#     pressure         0.9995      0.0254       0.0528    2.08x
#     temperature      0.9956      0.0744       0.0873    1.17x
#     humidity         0.9722      0.1868       0.2086    1.12x
#     wind_u           0.8472      0.4240       0.3797    0.90x
#     wind_v           0.8355      0.4385       0.3987    0.91x
#
# The model BEATS a copy on wind and LOSES to it on pressure, temperature and
# humidity — worst exactly where copying is easiest. That is a retrieval
# failure, not a modelling one: the decoder query carries no observations, so
# reproducing a station's own last value means attending from the query to the
# right subset of 1,872 patched encoder tokens. The LSTM baseline gets it free,
# because its input IS that station's history.
#
# This was tried in v15 and DEGRADED, for a reason recorded at the time: the
# base differs per station (y(t0) if visible, 0 if masked) but the queries
# carried no maskedness signal, so one head had to emit a small deviation in
# one regime and a full absolute value in the other, blind to which. The
# decoder now has a learned two-state station_state embedding, so the regime is
# observable. Not leakage: which stations are offline is known at deployment
# time; only their VALUES stay hidden.
# ── v22: simple forecasting transformer, no masking, no decoder ─────────────
# (station, timestep) tokens and the factorised encoder are unchanged; what
# goes is the masking and the query decoder.
#
#   --mask_ratio 0    every station visible, in training AND validation. This
#                     removes the train/eval mismatch (v15-v21 trained at 0.5
#                     and validated at 0, i.e. 1,872 tokens trained against
#                     3,720 validated) and makes the LSTM comparison exact.
#   --direct_head     read one vector per station off the encoder, project
#                     straight to all 13 horizons. No queries, no cross-
#                     attention, so a station's own last observation reaches
#                     its own prediction through the residual stream instead
#                     of by retrieval over 1,872 tokens.
#   --readout last    most recent temporal slot, the analogue of an LSTM's
#                     final hidden state. --readout mean gives every temporal
#                     token direct gradient if `last` proves unstable.
#
# CAUTION: model/masked_transformer.py used this same shape and diverged. That
# run predates the observation-token rebalance and trained at a content
# fraction of 0.04%, so the readout was never cleanly implicated — but run
# SANITY=2 (~1 h) before committing the full budget.

# ── Prediction path — pick ONE ───────────────────────────────────────────────
#
#   v23  ANCHOR   (current) --query_anchor, mask 0.5, no residual
#        A visible station's query starts from that station's OWN final encoder
#        token, fetched by gather. The station index is known when the query is
#        built, so attention is no longer asked to perform a soft learned lookup
#        for something an index does exactly. Masked stations keep mask_token
#        and must still be solved from neighbours — the leak guard is structural.
#
#   v20  RESIDUAL --residual_head, mask 0.5
#        y(t0) added outside the attention path: one scalar per variable, a
#        single base broadcast over all 13 horizons so the shortcut fades as
#        delta grows. Retired — it teaches the model nothing and hides whether
#        retrieval was ever learned. Kept runnable for the A/B.
#
# WHY v23 RUNS NEITHER
# The Delta=0 supervision fix (now unconditional, see model/mae.py) and the
# query anchor target the SAME failure: the decoder query carries no
# observations, so a station's own last value had to be located by attention
# over 1,872 encoder tokens. Enabling both would confound them.
#
# v23 is the Delta=0 fix ALONE — the cheaper change (zero parameters, no
# architecture change) and the more informative result. If it works, the
# retrieval failure was a SUPERVISION gap rather than a capacity limit: the
# model could always do the lookup, it was simply never asked. Delta=0 is the
# cleanest possible training signal for "attend to your own station's last
# token", and the pattern it teaches transfers to every other horizon, since
# the station-identity part of the query is identical across leads.
#
# Expect pressure to move first: persistence correlation at +30 min is 0.9995
# for pressure, so "copy your own last value" and "forecast +30 min" are nearly
# the same function for the variable v15 failed 11.2x on.
#
# If it does NOT recover v20's pressure result (0.0249 against a 0.0237 copy
# bound), try --delta0_weight 2 first — Delta=0 carries 1/13 of the loss and may
# simply be under-weighted. Only then promote the anchor to v24.
ANCHOR=""                                          # <- v23: Delta=0 fix alone
#ANCHOR="--query_anchor"                          # <- v24 arm


# ── Spatial attention — the controlled study ─────────────────────────────────
# --no_spatial_attn removes the spatial sub-layer from every encoder block, so
# each station is encoded from its OWN temporal window with no cross-station
# mixing. The transformer then has the same information as the LSTM baseline
# and differs only in mechanism.
#
# Why run it: the LSTM reaches 0.1878 on test while being spatially BLIND,
# against v20's 0.1778 and simple-MAE's 0.1783. If the station-independent arm
# lands close to full v20, neighbouring stations are contributing almost
# nothing and the spatial machinery is not earning its cost — which is the
# take-home message either way. Compare the per-lead curves and the error
# TAILS, not just the means.
SPATIAL=""                                         # <- full axial (default)
# SPATIAL="--no_spatial_attn"                      # <- station-independent arm
DIRECT=""                                          # <- v23 keeps the decoder
# ANCHOR="" ; DIRECT="--direct_head --readout last"   # <- v22 arm
# DIRECT=""                                        # <- v20: keep the query decoder
#
# NOTE: _mask_stations used to PERMUTE the station axis (argsort of random
# noise, sliced) even at --mask_ratio 0, and nothing un-permuted it. Harmless
# for the query decoder — the encoder output is only a key/value set — but
# fatal for the direct head, which reads encoded.view(B, T, n_stations, d)
# positionally and would have matched station j's target to another station's
# tokens, redrawn every step. Both index groups are now sorted, so the ORDER is
# deterministic while WHICH stations are masked stays random. At mask_ratio 0
# the visible set is exactly arange(N). See tests/test_station_order.py.

RESIDUAL="--residual_head"                       # <- v20
# RESIDUAL=""                                        # <- v23: REMOVED. The anchor
#   supersedes it — it hands over the station's full learned representation
#   rather than one scalar per variable, and applies equally at every lead time
#   instead of fading as delta grows. Re-enable only to reproduce v20.
# RESIDUAL="--residual_head"                       # <- v20. Redundant here. A
#   'last' readout already puts the station's own final token one linear layer
#   from the prediction, so the base has nothing left to shortcut. Note the two
#   are NOT mutually exclusive in code — _persistence_base is added after the
#   direct head too — so enabling both just double-counts the shortcut.
#
# Measured on the v20 test dump (best.ckpt, epoch 31), +30 min, all stations
# visible, normalised — the residual head is what made pressure work:
#
#     run              overall   temp   pressure  humidity  wind_u  wind_v
#     v20               0.1778  0.0536   0.0249    0.1397   0.3275  0.3433
#     simple-mae-v2     0.1783  0.0557   0.0408    0.1397   0.3187  0.3367
#     lstm-baseline-v1  0.1878  0.0658   0.0422    0.1452   0.3358  0.3499
#     v15               0.2493  0.0819   0.2645    0.1827   0.3495  0.3679
#
# Pressure went from 11.8x the persistence bound (0.0224) to 1.11x. This is the
# first transformer configuration in the project to beat the LSTM.
#
# KNOWN DEFECT, Δ=0 only: the base equals the target exactly for a visible
# station, so the correct residual is zero and the model does not produce a
# clean one. At Δ=0 its VISIBLE stations score 0.389 against 0.263 for its
# MASKED ones (temperature 3.6x, pressure 4.8x) — handed the exact answer it
# does worse than when handed nothing. Δ>0 is unaffected. Consider
# --delta0_weight below 1: for a visible station the Δ=0 target is a copy of an
# input and teaches nothing.
# OBS_ENCODER=""                                   # <- exact v17 observation encoder

PATCH=3

#ENCODER=""
ENCODER="--factorised_encoder"            # flat self-attention over W·N tokens

# ── Temporal window (flat or factorised encoder) ─────────────────────────────
#
# Splits W timesteps into non-overlapping chunks of size tw.
# Odd layers use a Swin-style half-window shift for cross-chunk communication.
#
# Flat encoder (current default):
#   Full cross-station attention within each tw×N_vis chunk.
#   W=72, tw=6, N_vis≈65 → 390 tokens/chunk  (144× cheaper than full flat)
#
TEMPORAL_WINDOW="--temporal_window 0"   # windowing retired — patching replaces it
# TEMPORAL_WINDOW="--temporal_window 6"   # v9/v11/v12 setting (unmasked shift!)
# TEMPORAL_WINDOW=""   # ← disable windowing (full attention)

# ── Window sampling strategy ──────────────────────────────────────────────────
#
# Training uses index_mode=random — each __getitem__ draws independently
# and uniformly from the full pool of ~260k valid windows (with replacement).
# The station mask is re-drawn per sample, so revisiting the same time window
# in a later step is NOT redundant — a different 50% of stations are masked,
# creating a genuinely different training task.
#
# Delta grid: --delta_mode fixed_grid --delta_grid_stride 3 --max_delta 36
#   K=13 horizons: [0, 3, 6, …, 36] steps = [now, 30min, 1h, …, 6h]
#   k=0 (delta=0): inpainting — loss on MASKED stations only
#   k=1..12 (delta=3..36): forecasting — loss on ALL stations
#   Decoder step indices: W-1+delta = [71, 74, 77, …, 107] (unified timeline)
#
# Epoch size rationale
# --------------------
# The non-overlapping block count for 5 training years is:
#   pool (~260,000) / W (72) ≈ 3,611 windows
# This is the minimum epoch size that gives one "effective pass" over the data.
# We set random_epoch_size=10000 (≈ 2.8× non-overlapping) as the baseline.
# Each revisit has a different random station mask, so it is NOT redundant.
#
# This is a 4× reduction from the previous 40,000-step epochs, which was
# 11× redundant within each epoch and led to:
#   • 3h/epoch → very coarse LR scheduling and monitoring granularity
#   • Only ~33 epochs in a 100h budget
# With 10,000 steps/epoch (~2500 batches at batch_size=4):
#   • ~50 min/epoch → fine-grained LR decay and early stopping
#   • ~120 epochs in a 100h budget
#
# Validation uses non-overlapping blocks (hardcoded in main.py):
#   Val year 2022 → ~728 windows → ~23 batches at batch_size=4 → ~6 s/epoch
#   No --limit_val_batches needed — the full val set is evaluated every epoch.
#
INDEX_MODE="--index_mode random --random_epoch_size 10000" # ~2.8× non-overlapping (fast ablation)

# ── Sanity / smoke run ───────────────────────────────────────────────────────
# v15 changes the encoder tokenization AND the prediction head, so the FIRST
# launch should exercise those code paths cheaply before the 100-epoch budget.
# Enable with the env var (no need to edit this file):
#
#     SANITY=1 bash src/scripts/run_full_cloud.sh      # ~5 min, 2 epochs
#     SANITY=2 bash src/scripts/run_full_cloud.sh      # ~1 h, 15 epochs, 2 yr
#     bash src/scripts/run_full_cloud.sh               # full run (default)
#
# SANITY=1 — does it RUN?  2 epochs x 200 windows, subset years, no compile.
#            Checks: raw-token shapes (5,616-token sequence), finite val/loss,
#            sanity/* line printed, [cuda] preflight, checkpoint writes.
# SANITY=2 — does it LEARN? 15 epochs x 3,600 windows on 2020-21. Expect
#            val/overall_mae falling and sanity/ctx_ratio dropping below 1
#            (visible stations beating masked ones at Delta=0 — with the
#            residual head OFF this is once again a genuine test that the
#            model uses the observations it can see, as it was in v14).
# Both write to <SAVE_DIR>-sanity so they can never collide with the real run.
SANITY="${SANITY:-0}"
SANITY_ARGS=()
if [[ "$SANITY" == "1" ]]; then
    SANITY_ARGS=(--epochs 2 --random_epoch_size 200 --subset
                 --warmup_epochs 1 --patience 100 --val_check_interval 0)
    INDEX_MODE="--index_mode random"
    SAVE_DIR="${SAVE_DIR}-sanity"
    RUN_SUFFIX="-sanity1"
elif [[ "$SANITY" == "2" ]]; then
    SANITY_ARGS=(--epochs 15 --random_epoch_size 2500 --subset --warmup_epochs 2)
    INDEX_MODE="--index_mode random"
    SAVE_DIR="${SAVE_DIR}-sanity"
    RUN_SUFFIX="-sanity2"
else
    RUN_SUFFIX=""
fi
if [[ "$SANITY" != "0" ]]; then
    echo "[sanity] SANITY=${SANITY} — short run into ${SAVE_DIR}"
    echo "[sanity]   ${SANITY_ARGS[*]}"
    echo "[sanity]   this is NOT the real run; unset SANITY for the full budget."
fi
# INDEX_MODE="--index_mode blocks"                           # non-overlapping train (fast ablation)
# INDEX_MODE="--index_mode sliding --train_stride 9"         # sliding, hourly stride

# ── Loss function ─────────────────────────────────────────────────────────────
#
# v12 loss: Huber(δ=1.0) for ALL variables — SAME as v11, only the per-variable
# WEIGHTS change.  Clean A/B on the weights alone (δ, loss family unchanged).
#
#   Loss = (1/K) Σ_k  Σ_v  w_v · Huber(ŷ_kvn − y_kvn, δ=1.0)  /  Σ_v w_v
#
# Weights [T=1.0, P=1.0, RH=0.7, u=0.5, v=0.5] (set in model/mae.py):
#   • v11 (uniform [1,1,1,1,1]) wandb curves: temperature & pressure keep
#     improving and do NOT overfit; humidity + wind_u/v bottom ~epoch 30-40 then
#     degrade.  So DOWN-weight the noisy channels (opposite of v10, which
#     UP-weighted wind 1.5× and overfit worse) → encoder spends less capacity
#     fitting unpredictable gust / saturation noise.
#   • Temperature/pressure held at 1.0 → their gradient is unchanged vs v11,
#     keeping the comparison interpretable.
#   • δ=1.0 in normalised space = 1 std-dev: L2 for typical errors,
#     L1 (capped gradient) for extreme events.
#     If wind/humidity still overfit, next loss-only lever: δ=0.5.
#   • No σ² head — simpler decoder, stable gradients, no exploitation of
#     inflated uncertainty to artificially reduce NLL.
#   • Precipitation excluded from targets (num_target_vars=5).
#
# v9 (= saved best.ckpt, all current test results) used --nll_loss
# (heteroscedastic Gaussian NLL with σ² head, uniform variable weights).
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Batch size (reviewed 2026-07-31) ─────────────────────────────────────────
# LEFT AT 4 x 4 = effective batch 16, deliberately.
#
# The slice (A100-80GB MIG 3g.20gb, 19,968 MiB, 28/108 SMs) is the same one
# these settings were tuned on, and this run already uses --grad_checkpoint
# *because* 20 GB is tight. Training memory is dominated by activations and
# optimizer state, so doubling batch_size here is a real OOM risk — and an OOM
# several hours into a multi-day run is expensive.
#
# If you want the speedup, 8 x 2 keeps the EFFECTIVE batch at 16, so the
# optimisation problem is unchanged and the run stays comparable to v9/v11/v12:
#     --batch_size 8 --accumulate_grad_batches 2
# Test it first with --epochs 1 --random_epoch_size 200 and watch nvidia-smi.
# Never raise the effective batch itself without revisiting the learning rate.
#
# Inference is different: test.py holds no activations or optimizer state, so
# run_test_cloud.sh runs at batch_size 16.
#
# ── v15 FIRST-LAUNCH CHECKLIST ───────────────────────────────────────────────
# This branch changes the encoder's tokenization and the prediction head, so
# the very first launch should be a 2-minute smoke run, NOT the full budget:
#
#     SANITY=1 bash src/scripts/run_full_cloud.sh                  # ~5 min
#     SANITY=2 bash src/scripts/run_full_cloud.sh                  # ~1 h
#
# SANITY=1 answers "does it run?", SANITY=2 answers "does it learn?".
# Confirm in the output:
#   1. no shape/OOM error on the 1,872-token sequence (24 x 78);
#   2. "[cuda] ✓ torch ... A100" appeared at the top (preflight);
#   3. val/loss is finite and sanity/* prints one line per epoch;
#   4. val/horizon_sensitivity RISING off the floor (>0.05, ideally toward
#      0.2+) and sanity/dispersion_min climbing past ~0.3. Together these say
#      the model has left the constant-output basin and is using the
#      Delta-embedding. FALLING horizon_sensitivity is the collapse signature
#      that killed the P=1 run — kill the job and re-tune rather than waiting.
#   5. sanity ctx_ratio < 0.9 — VISIBLE stations beat masked ones at Delta=0,
#      i.e. the 30-min tokens carry recent observations to the decoder (what
#      the removed input_context pathway used to provide). NOTE this only
#      becomes meaningful once dispersion is up: a constant predictor scores
#      exactly 1.00 here by arithmetic, not by failure of the context path.
# --compile gives ~1.3x; it is a speedup, not a correctness requirement.
# ── Resume ───────────────────────────────────────────────────────────────────
# If last.ckpt exists in SAVE_DIR, continue from it: full training state
# (weights, optimiser, LR schedule, epoch counter) is restored.
#
# Without this, re-running would start from scratch and ModelCheckpoint would
# refuse to overwrite the existing files, silently creating best-v1.ckpt --
# leaving two runs in one folder with no indication which is which.
#
# NOTE: OverfitEarlyStop keeps no state_dict, so its patience counter resets to
# zero on resume. That is deliberate here (it gives a stopped run room to
# continue) but it means patience alone will not protect a resumed run --
# set --overfit_patience appropriately.
RESUME="${RESUME:-1}"
RESUME_ARG=()   # array, NOT a string: the project path contains a space
if [[ "$RESUME" == "1" && -f "${SAVE_DIR}/last.ckpt" ]]; then
    RESUME_ARG=(--resume "${SAVE_DIR}/last.ckpt")
    echo "[resume] continuing from ${SAVE_DIR}/last.ckpt"
    python - "$SAVE_DIR" <<'PYEOF'
import sys, os, torch
p = os.path.join(sys.argv[1], "last.ckpt")
try:
    c = torch.load(p, map_location="cpu", weights_only=False)
    print(f"[resume]   epoch {c.get('epoch','?')}, global_step {c.get('global_step','?')}")
except Exception as e:
    print(f"[resume]   (could not read epoch: {type(e).__name__})")
PYEOF
elif [[ "$RESUME" == "1" ]]; then
    echo "[resume] no last.ckpt in ${SAVE_DIR} — starting fresh"
else
    echo "[resume] RESUME=0 — starting fresh"
    if [[ -f "${SAVE_DIR}/best.ckpt" ]]; then
        echo "[resume] WARNING: ${SAVE_DIR} already contains best.ckpt."
        echo "[resume]          A fresh run will write best-v1.ckpt alongside it."
        echo "[resume]          Use a new SAVE_DIR instead."
        exit 1
    fi
fi

python main.py \
    --data_root        "$DATA_ROOT" \
    --cache_dir        "$DATA_ROOT" \
    --local_cache_dir  "$LOCAL_CACHE" \
    --window           72 \
    --max_delta        36 \
    --delta_mode       fixed_grid \
    --delta_grid_stride 3 \
    --mlp_ratio        4.0 \
    --d_model          384 \
    --enc_heads        8 \
    --dec_heads        8 \
    --enc_layers       8 \
    --dec_layers       2 \
    --mask_ratio       0.5 \
    --temporal_patch   $PATCH \
    --var_weights      1.0 1.0 1.0 1.0 1.0 \
    --dropout          0.1 \
    --drop_path_rate   0.1 \
    --batch_size       4 \
    --num_workers      3 \
    --epochs           100 \
    --lr               1e-4 \
    --warmup_epochs    10 \
    --weight_decay     0.05 \
    --grad_clip        1.0 \
    --accumulate_grad_batches 4 \
    --patience         40 \
    --monitor          val/overall_mae \
    --overfit_stop \
    --overfit_patience 20 \
    --min_lr           5e-7 \
    --amp \
    --bf16 \
    --compile \
    --grad_checkpoint \
    --cross_attn_decoder \
    $OBS_ENCODER \
    $STATIC \
    $RESIDUAL \
    $ANCHOR \
    $SPATIAL \
    $DIRECT \
    $ENCODER \
    $TEMPORAL_WINDOW \
    $INDEX_MODE \
    $EXCLUDE \
    --wandb_project    station-mae \
    --wandb_run_name   "patch${PATCH}-d384-L8-v25-res${RUN_SUFFIX}" \
    ${SANITY_ARGS[@]+"${SANITY_ARGS[@]}"} \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
    --save_dir         "$SAVE_DIR"
# WandB runs ONLINE (default). If Renku blocks outbound connections, add
# --wandb_offline above and sync later with: wandb sync <run-dir>
# NOTE: --polybox_dir removed — Polybox writes during training are unreliable.
# After training finishes, manually copy checkpoints:
#   cp "$SAVE_DIR/best.ckpt" /home/renku/work/polybox-capstone/checkpoints/tw6-d1024-v12-best.ckpt
