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
