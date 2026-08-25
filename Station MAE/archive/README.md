# archive/

Code kept for provenance, NOT part of the two documented workflows.

## train_masked_transformer.py / test_masked_transformer.py

An early third architecture ("masked-tf"): an encoder-only masked transformer
with in-place corruption and a pooled readout, superseded by the Delta-query
MAE in `src/model/`.

**These scripts cannot run.** Both do

    from model.masked_transformer import MaskedStationTransformer

and `src/model/masked_transformer.py` no longer exists in the repository. They
were moved here during the handover cleanup rather than deleted, because
`src/model/mae.py:395` and `src/scripts/run_full_cloud.sh:222` still refer to
this experiment in comments explaining why the current design differs from it.

No checkpoint or `test_results/` entry corresponds to this model.

## removed_deadcode_2026-08-25/

Code removed in the dead-code pass. Reachability was traced by AST from the
four launch scripts in `src/scripts/` (`main.py`, `test.py`, `train_lstm.py`,
`test_lstm.py`), recursively through imports, then cross-checked against the
notebooks.

| item | was | why removed |
|---|---|---|
| `lstm_metrics.py` | `src/engine/lstm_metrics.py` | `aggregate_metrics` referenced nowhere in `src/` or `notebooks/` |
| `metrics.py` | `src/model/metrics.py` | its 4 functions referenced nowhere; the apparent hits in `lightning_module.py` are **log-label strings** (`"sanity/persist_skill"`), not calls |
| `src_engine_evaluate.py.removed.txt` | `src/engine/evaluate.py` | `evaluate`, `print_metrics`, `print_full_metrics`, `print_gap_filling_metrics`, `build_delta_variable_matrix`, `compute_seasonal_metrics` — 377 lines. The printers served the CLI metrics path deleted earlier; `evaluate` is the superseded single-delta loop |
| `src_model_embeddings.py.removed.txt` | `src/model/embeddings.py` | `encode_spatial_static`, `compute_spatial_normalization` — 75 lines, no reference anywhere |

Two package `__init__` files were updated because they re-exported removed
names and would otherwise have broken `import model` / `import engine`:

* `src/model/__init__.py` — dropped `encode_spatial_static`,
  `compute_spatial_normalization` from the import list and `__all__`.
* `src/engine/__init__.py` — no longer re-exports anything. It is executed by
  every `from engine.evaluate import ...`, so an eager re-export made each
  training run import the whole evaluation stack.

To restore any of it, copy the block back into the original file and re-add the
`__init__` entry.

## Six never-exercised flags removed (2026-08-25, second pass)

Evidence: `hyper_parameters.cfg` of all four transformer checkpoints. Each flag
held the same value in every run, so the code branch it selected was the only
one ever taken; the removal hardcodes that branch.

| flag | value in all runs | what was removed |
|---|---|---|
| `--direct_head` | False | ctor guard, `direct_proj` head, `_direct_predict()`, the `if self.direct_head:` branch in `forward_multi_delta` (else-body kept), the mask-ratio guard in `test.py` |
| `--query_anchor` | False | flag, `self.query_anchor`, three `anchor=(... if self.query_anchor else None)` call sites → `anchor=None` |
| `--masked_only_loss` | False | flag, attribute, two `_midx` selections → `None` |
| `--persist_norm` | False | flag, `_estimate_persist_mse()`, `set_persist_mse()`, the loss-division branch, the cfg entry |
| `--delta0_weight` | 1.0 | flag, attribute, the `torch.where` horizon weighting → `new_ones(K) / K` |
| `--wind_encoder` | False | flag, the `wind_pair` derivation in `from_cfg`, `wind_pair=` at the `StationMAE` call site |

**Deliberately KEPT** because their tensors are present in every saved
`state_dict`, so removing them would break or noisily degrade checkpoint
loading: `decoder.anchor_norm`, `decoder.station_state`, and the `persist_mse`
buffer (now pinned at 1.0 and unused).

`wind_pair` parameters remain in `encoder.py` and `embeddings.py`: they default
to `None` and are no longer passed, so the behaviour is unchanged. Removing
them means editing `VariableProjection`, which was out of scope for this pass.

**Still present** (deferred pending a live SANITY run): `--temporal_window`
(33 refs, gates the Swin-style windowed temporal attention in `encoder.py`) and
`--static_in_token` (17 refs, alters token construction).

### 2026-08-25 — `query_anchor` removal completed

The `--query_anchor` flag was removed earlier in this cleanup, which left two
pieces of unreachable code behind. Both are now gone:

* `src/model/decoder.py` — `self.anchor_norm` (a `LayerNorm`), the `anchor`
  parameter of `forward`, and the `if anchor is None: … else: …` branch, which
  collapses to the `mask_token` line that every run took.
* `src/model/mae.py` — `StationMAE._query_anchor` (42 lines) and the three
  `anchor=None` keyword arguments at the decoder call sites.

Unreachability was established by parsing the pre-edit `mae.py` and confirming
all three decoder call sites passed a literal `None`, so the `else` branch was
never entered in any run. The remaining semantic diff is exactly the deletion
of that branch.

`decoder.station_state` and `decoder.mask_token` were **kept** — `station_state`
is added unconditionally on every forward pass and carries the visible/masked
distinction into the query.

`persist_mse` was likewise kept: it is an unused buffer, but it appears in
every saved `state_dict`.

Old checkpoints still contain `decoder.anchor_norm.weight` / `.bias`. These land
in `unexpected` under `strict=False` and are ignored; `src/test.py` now filters
them from the load report so a supervisor does not read them as a structural
mismatch. No checkpoint file was modified.

### 2026-08-25 — legacy in-script metrics moved out of `engine/evaluate.py`

`archive/removed_deadcode_2026-08-25/engine_evaluate_legacy_metrics.py`

`src/engine/evaluate.py` was 1,018 lines, of which 645 were unreachable. Moved
here verbatim:

| moved | lines | what it did |
|---|---|---|
| `evaluate_full` | 271 | full test-set evaluation in physical units — RMSE / MAE / bias / R², wind speed, circular wind-direction error, per-lead breakdown |
| `evaluate_gap_filling` | 269 | the same, split by masked vs visible stations |
| `_row_stat`, `_r2`, `_wind_dir_deg`, `_circular_mae_deg` | 61 | helpers called only by those two |
| `_VAR_UNITS`, `_IDX` | 10 | module constants read only by those two |

These are the functions that computed metrics **inside** the evaluation script.
They became unreachable when `test.py` switched to dumping raw tensors to
`predictions.pt` and letting `notebooks/` own every metric, so that the script
and the analysis cannot disagree. **Results predating that switch were produced
by this code**; results after it come from the notebooks. The two are not
guaranteed to agree — see `EXPERIMENTS.md`, "Open questions".

What remains in `src/engine/evaluate.py` (362 lines):

* `collect_predictions` — the live path, called by `src/test.py:848`
* `evaluate_per_station` — called by `notebooks/Station_MAE_Map.ipynb`

Both were verified **byte-for-byte identical** before and after the extraction
(SHA-256 of the exact source span). Removal was gated on an AST check that
neither retained function references any moved name.

The archived file is a record, not a module: it was cut out of a package and
still expects `engine/evaluate.py`'s imports. To run any of it, paste the
function back.

Also removed in the same pass: `import shutil` in `src/main.py`, left behind by
the `CopyCheckpointToPolybox` deletion.

**Deliberately kept** (they look unused and are not):

* `rioxarray` in `data/visualize.py` — imported for the `.rio` accessor side
  effect; already marked `# noqa: F401`
* `from __future__ import annotations` in `model/token_balance.py`
* `StationMAE.mask_ratio` — read by `test.py:788`
* `StationMAE.readout`, `StationMAE.num_horizons` — one-line assignments never
  read back, left in place as a record of the configuration

### 2026-08-25 — package `__init__.py` re-exports removed

No file was moved and no import statement outside the three `__init__.py` files
was touched. `data/__init__.py` and `model/__init__.py` were emptied of their
eager re-exports, bringing them in line with the policy already documented in
`engine/__init__.py`.

Why it mattered: a package `__init__.py` executes on every import of any of its
submodules.

* `data/__init__.py` re-exported from `.visualize`, so every
  `from data.dataset import ...` — i.e. **all four entry points** — imported
  geopandas, rioxarray and matplotlib. Training runs paid for a plotting stack
  used by one notebook.
* `model/__init__.py` re-exported `StationMAELightning`, so every
  `from model.mae import ...` imported pytorch_lightning. `src/test.py` does
  this at line 657 and never references Lightning.

Verified safe before the change: all 44 package imports in `src/`, `notebooks/`
and `tests/` already use the submodule form (`from data.dataset import`,
`from model.mae import`). Nothing used the flat re-export, so nothing broke.
After the change, 112 imported names were re-checked against the modules that
define them — all resolve.

Transitive imports of each entry point, after:

| entry point | pytorch_lightning | geopandas / rioxarray |
|---|---|---|
| `main.py` | yes — it trains | no |
| `train_lstm.py` | yes — it trains | no |
| `test.py` | **no** (was yes) | **no** (was yes) |
| `test_lstm.py` | no | **no** (was yes) |

The old re-export list is preserved in git history; it is not reproduced here
because every name in it is still importable from its own module.

Considered and NOT done, to keep import paths stable for the supervisor:
moving `data/visualize.py` out of the data package, and `model/lightning_module.py`
into `engine/`. Both are defensible on layering grounds — `visualize` is a
notebook helper and `lightning_module` is a training wrapper rather than a
model — but neither is required now that the `__init__` files no longer force
them onto unrelated processes.
