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
