# Archive

Superseded files, kept because some contain results that have not been
re-derived elsewhere. Nothing here is imported by `src/` or `notebooks/`.

| file | last touched | superseded by |
|---|---|---|
| `Data_Exploration.ipynb` | 2026-07-28 | `notebooks/Train_Test_Exploration.ipynb` |
| `plot_test_results.ipynb` | moved 2026-08-21 | `notebooks/Station_MAE_Map.ipynb` (DEM/map sections, updated for the v27 predictions.pt schema — no more `per_station_metrics.csv`) and `notebooks/Test_Results_Exploration.ipynb` (aggregate model-comparison stats) |
| `plot_model.ipynb` | 2026-07-24 | `notebooks/Model_Architecture.ipynb` |
| `plot_stations.ipynb` | 2026-06-19 | `notebooks/Train_Test_Exploration.ipynb` (station/terrain sections) |
| `analysis.ipynb` | 2026-06-19 | `notebooks/Test_Results_Exploration.ipynb` |
| `arch_tree.py` | 2026-06-04 | `notebooks/Model_Architecture.ipynb` (ModelSummary + torchinfo) |
| `profile_encoder.py`, `run_profile_cloud.sh` | 2026-04/05 | one-off encoder profiling, not part of the pipeline |
| `run_subset_cloud.sh` | 2026-06-04 | subset runs; `run_full_cloud.sh` covers this via `--subset` |
| `run_sweep.sh`, `sweep.yaml` | 2026-07-01 | wandb sweep from the d128 era — hyperparameters no longer current |

`_junk/` holds emptied logs, `.DS_Store` and `__pycache__` directories that
could not be deleted directly from the sandbox. It is safe to remove.

These paths were **not** updated for the `src/` reorganisation — if you revive
one, fix its imports first.
