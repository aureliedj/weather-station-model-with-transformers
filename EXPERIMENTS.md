# Experiments — checkpoints, configurations and results

Every value below was **read from the checkpoint files themselves**
(`hyper_parameters.cfg` and the tensor shapes in `state_dict`), not from
launcher defaults or notes. Regenerate this table at any time with:

```bash
python src/inspect_checkpoints.py            # prints cfg for every checkpoint
```

---

## The two models

| | Transformer MAE | LSTM baseline |
|---|---|---|
| code | `src/model/mae.py` (+ `encoder.py`, `decoder.py`, `embeddings.py`) | `src/model/lstm_baseline.py` |
| train | `src/main.py` via `src/scripts/run_full_cloud.sh` | `src/train_lstm.py` via `src/scripts/run_lstm_cloud.sh` |
| evaluate | `src/test.py` via `src/scripts/run_test_cloud.sh` | `src/test_lstm.py` via `src/scripts/run_lstm_test_cloud.sh` |
| masking | station-level, ViT-MAE style | none — spatially blind by construction |

Both are trained on the same splits (train 2017–2021, val 2022, test 2023–2024,
defined in `src/data/dataset.py`), the same per-station normalisation, and the
same 13-lead grid (0 … 36 steps of 10 min = 0 … 6 h in 30-min increments).

---

## Checkpoint → configuration → results

All transformer runs share `d_model=384`, `enc_layers=8`, `dec_layers=2`,
`enc/dec_heads=8`, `mlp_ratio=4.0`, `window=72` (12 h), `temporal_patch=3`,
`max_delta=36`, `delta_grid_stride=3`, `residual_head=True`,
`cross_attn_decoder=True`, `factorised_encoder=True`, `dropout=0.1`,
`drop_path=0.1`, `var_weights=[1,1,1,1,1]`, and 25,629,031 parameters.
They differ **only** in the columns below.

| checkpoint dir | train mask ratio | loss | encoder spatial attn | decoder | epoch reached | test_results |
|---|---|---|---|---|---|---|
| `full_run_cloud_v27` | 0.5 | Huber δ=1 | yes | global | 40 | `v27/best_mr0.00`, `v27/best_mr0.50` |
| `full_run_cloud_v30-nll` | 0.5 | Gaussian NLL | yes | global | 34 | `v30-nll/best_mr0.00`, `v30-nll/best_mr0.50` |
| `full_run_cloud_v31` | 0.0 | Huber δ=1 | yes | global | 47 | `v31/best_mr0.00` |
| `full_run_cloud_v32-blind` | 0.0 | Huber δ=1 | **no** | **station-local** | 26 | `v32-blind/best_mr0.00` |
| `lstm-baseline-v1` | n/a | — | n/a | n/a | 40 | `lstm-baseline-v1/best_mr0.00` |

`v32-blind` has fewer parameters (no encoder spatial attention layers), which is
why its checkpoint is 243 MB against 300 MB for the others.

### What each run isolates

* **v27** — the reference transformer: trained with 50% of stations hidden, so
  it learns forecasting *and* spatial gap-filling.
* **v31 vs v27** — the effect of training-time masking. Same architecture, mask
  ratio 0.5 → 0.0.
* **v32-blind vs v31** — the value of cross-station information. Same training
  regime; v32 removes the encoder's spatial sub-layer *and* makes the decoder
  station-local, closing every cross-station pathway.
* **v30-nll vs v27** — Huber vs heteroscedastic Gaussian NLL. v30 additionally
  emits `log_var`, so it is the only run with predictive uncertainty.
* **lstm-baseline-v1** — per-station recurrence, spatially blind; the reference
  for "what does spatial context buy?".

---

## Test scenarios

`test.py` writes exactly one file per (checkpoint, mask ratio):
`test_results/<run>/best_mr<R>/predictions.pt`. No metric is computed there —
everything is derived downstream in the notebooks, so the numbers have a single
source of truth.

| mask ratio | meaning | valid for |
|---|---|---|
| `mr0.00` | all stations visible — pure forecasting | every run; the only regime where the LSTM is comparable |
| `mr0.50` | half the stations hidden — forecasting + gap-filling | v27, v30-nll (the runs trained with masking) |

`v31` and `v32-blind` were trained at mask ratio 0; `mr0.50` would be out of
distribution for v31 and is structurally refused for v32 (`station_local_decoder`
needs every station to contribute encoder tokens).

### `predictions.pt` contents

```
preds        (M, K, N, 5)   normalised predictions
targets      (M, K, N, 6)   normalised targets
masks        (M, K, N, 6)   sensor availability
masked_idx   (M, n_masked)  stations hidden from the encoder (width 0 at mr0.00, 77 at mr0.50)
delta_steps  (M, K)         lead times in 10-min steps
window_hours (M,)           window start, hours since epoch
target_hours (M, K)         target time per lead
spatial      (N, 15)        static station descriptors
log_var      (M, K, N, 5)   log sigma^2 — v30-nll only
```

M = 11,684 windows (sliding, stride 9 = 90 min, over 2023–2024), K = 13 leads,
N = 155 stations (160 meteo stations, minus 4 without any of the six
parameters, minus PFA which is excluded explicitly).

### Verified properties of the current dumps

* **Targets, masks and time axes are bit-identical across all seven dumps.**
  Checked cell-by-cell over all 141,259,560 target cells for the LSTM/v27 pair;
  spot-checked for the others. Model comparisons are therefore strictly paired.
* **The mr0.50 masked sets of v27 and v30-nll are NOT identical**, despite both
  using seed 42. Masked-vs-visible splits *within* a model are exact;
  masked-station comparisons *between* v27 and v30 are unpaired. Re-running one
  of the two at the other's batch size would fix it — reproducibility of the
  mask requires the same `--seed`, `--batch_size`, `--index_mode` and `--stride`.

---

## Station and variable exclusions

Applied in the analysis, not in the dumps:

| exclusion | reason |
|---|---|
| station `PFA` | excluded at dataset level (`--exclude_stations PFA`); N = 155 not 156 |
| `GES` pressure | 0.00% coverage in the training years |
| `LAE` wind_u | 0.00% coverage in the training years |
| `LAE` wind_v | **99.95% coverage** — currently excluded together with `wind_u`; see open questions |

25 further station×variable pairs have under 1% training coverage (e.g. `PRE`,
`AEG` have no temperature/pressure/humidity). None of them carry a test-set
score, so they do not affect any reported number.

---

## Open questions, deliberately not changed

1. **`LAE/wind_v` is excluded but has 99.95% training coverage.** If the
   exclusion was motivated by `wind_u` alone, usable data is being discarded.
2. **v27/v30 mr0.50 masks differ** (above). Affects only cross-model
   masked-station comparisons.
3. **`compute_persistence_metrics` used an averaged std** in an older version of
   the evaluation path; the current path computes baselines from raw
   observations, which avoids the issue. Historical results predating that
   change are not directly comparable.
4. **Two unused locals** (`c_enc_heads`, `c_dec_heads` in `test.py`) are left in
   place: they document the legacy-checkpoint branch and removing them changes
   nothing.

---

## What the checkpoints prove about the configuration surface

Read from `hyper_parameters.cfg` of all four transformer checkpoints. Of the 43
recorded parameters, **only four ever differ between runs** — these are the
experiments:

| parameter | v27 | v30-nll | v31 | v32-blind |
|---|---|---|---|---|
| `mask_ratio` | 0.5 | 0.5 | 0.0 | 0.0 |
| `use_nll_loss` | False | **True** | False | False |
| `encoder_spatial_attn` | True | True | True | **False** |
| `station_local_decoder` | absent | absent | absent | **True** |

The remaining 39 are identical everywhere, so any difference in results is
attributable to those four settings alone.

### Options that were never exercised — six removed, two retained

All four checkpoints record these eight flags at their default value, so no
result in this repository depends on any of them.

**Removed** from `main.py` and the model, together with the code they gated:

| flag | value in every run | what it gated |
|---|---|---|
| `--query_anchor` | False | v23/v24 arm: a visible station's query started from its own final encoder token instead of the shared `mask_token`. Removed with `StationMAE._query_anchor` and `decoder.anchor_norm`. |
| `--direct_head` | False | v22 arm: skipped the decoder and regressed from the encoder output. |
| `--masked_only_loss` | False | restricted the loss to masked stations. |
| `--use_persist_norm` | False | scaled the loss by a per-variable persistence MSE. |
| `--delta0_weight` | 1.0 | reweighted the Δ=0 horizon. |
| `--wind_encoder` | False | a separate encoder branch for the wind components. |

The `persist_mse` **buffer is deliberately retained** in `mae.py` even though
nothing writes it: it is present in every saved `state_dict`, and removing it
would turn a clean load into an `unexpected key`.

`decoder.station_state` looks like part of the same v23 arm and is **not** —
it is added on every forward pass and distinguishes visible from masked
queries. It is live code.

**Retained**, pending a `SANITY=1` run to confirm nothing depends on them:

| flag | value in every run | note |
|---|---|---|
| `--temporal_window` | 0 | local temporal attention; 33 references |
| `--static_in_token` | False | statics as token slots (v21 arm); 17 references |

**Loading old checkpoints after the removal.** Checkpoints trained before this
cleanup contain `decoder.anchor_norm.weight` / `.bias`. The module no longer
exists, so `load_state_dict(..., strict=False)` places them in `unexpected` and
ignores them; `test.py` filters them from the load report explicitly so they
are not mistaken for a mismatch. The removed branch was unreachable in every
run — all three decoder call sites passed `anchor=None` — so predictions are
unchanged. `run_full_cloud.sh` no longer carries the permanently-empty
`$STATIC` and `$ANCHOR` toggles.

### Load-bearing defaults — do not "simplify"

`--value_embedding` defaults to **`linear`**, but every run passed **`mlp`**
via `$OBS_ENCODER`. Dropping that argument would silently change the
architecture. Likewise `--residual_head` is passed explicitly and is True in
every run.
