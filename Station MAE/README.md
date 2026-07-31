# Station-MAE — weather station modelling with transformers

DAS capstone, SDSC / ETH. A Masked AutoEncoder over MeteoSwiss SwissMetNet
station data ([PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather)),
with a spatially-blind LSTM and persistence as baselines.

## Layout

```
Station MAE/
├── src/                     ← everything that trains or evaluates a model
│   ├── data/                  dataset, windowing, normalisation
│   ├── engine/                evaluation loop, metric aggregation
│   ├── model/                 MAE (encoder/decoder/embeddings) + LSTM baseline
│   ├── main.py                train the transformer
│   ├── test.py                evaluate the transformer  → test_results/
│   ├── train_lstm.py          train the LSTM baseline
│   ├── test_lstm.py           evaluate the LSTM         → test_results/
│   ├── download.py            fetch PeakWeather
│   └── scripts/               launchers — run these, not the .py directly
│       ├── run_full_cloud.sh        transformer training  (Renku)
│       ├── run_test_cloud.sh        transformer evaluation
│       ├── run_lstm_cloud.sh        LSTM training (chains into the test script)
│       └── run_lstm_test_cloud.sh   LSTM evaluation
│
├── notebooks/               ← analysis only; never imported by src/
│   ├── Train_Test_Exploration.ipynb      dataset QC: ranges, gaps, drift, climatology
│   ├── Test_Results_Exploration.ipynb    masking, vs persistence, terrain, per-station, σ
│   ├── Missing_In_Training_TestPerf.ipynb performance by training coverage (the GES finding)
│   ├── Priority2_Error_Diagnostics.ipynb  distribution shift, pressure deep-dive
│   ├── Model_Architecture.ipynb          layer/shape/parameter summaries
│   ├── plot_training_logs.ipynb          wandb curves per epoch
│   └── explore_pipeline.ipynb            end-to-end walkthrough of the data pipeline
│
├── tests/                   ← run with plain `python tests/<file>.py`
├── docs/                    ← reports, literature comparison, supervisor notes
├── archive/                 ← superseded notebooks and scripts, kept for reference
│
├── checkpoints/             ┐
├── test_results/            ├ generated — gitignored
├── report/                  │  (report/ holds notebook exports: summaries, CSVs)
└── PeakWeatherDataset/      ┘
```

## Running things

**Training and evaluation** — always launch through `src/scripts/`. Each script
resolves its own location, `cd`s into `src/`, and writes artefacts to the
project root, so it works from any working directory:

```bash
bash "Station MAE/src/scripts/run_full_cloud.sh"       # transformer
bash "Station MAE/src/scripts/run_lstm_cloud.sh"       # LSTM (trains, then evaluates)
bash "Station MAE/src/scripts/run_test_cloud.sh"       # evaluate a transformer checkpoint
```

Edit the config block at the top of each script (`DATA_ROOT`, `RUN_NAME`,
`CHECKPOINT`, hyperparameters) rather than passing flags by hand.

**Notebooks** — open from `notebooks/`. The first cell is a bootstrap that
`chdir`s to the project root and puts `src/` on `sys.path`, so relative paths
like `test_results/` resolve and `from data.dataset import …` works. It is
idempotent; re-running it does nothing.

**Tests** — no pytest needed:

```bash
python "Station MAE/tests/test_lstm_eval.py"
python "Station MAE/tests/test_uncertainty_cells.py"
```

## Model runs

| run | model | loss | notes |
|---|---|---|---|
| `v9`  | transformer tw6 d1024 | heteroscedastic Gaussian NLL | predicts σ — see §5 of the results notebook |
| `v11` | transformer tw6 d1024 | Huber, unweighted | overfits early |
| `v12` | transformer tw6 d1024 | Huber, down-weighted noisy vars | dropout 0.1, early stop on overfit |
| `lstm-baseline-v1` | per-station LSTM | Huber | spatially blind; isolates the value of spatial modelling |

Evaluation is rolling-origin over the 2023–2024 test years, stride 9 (1 h 30),
on a fixed 13-horizon grid (Δ = 0 … 6 h at 30-min spacing). All models and
persistence share the identical target tensor, so the comparison is paired.
