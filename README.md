# Station-MAE — weather station modelling with transformers

DAS capstone, SDSC / ETH Zürich. A Masked AutoEncoder over MeteoSwiss
SwissMetNet station data
([PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather)),
with a spatially-blind LSTM and two temporal baselines as references.

**The task.** Given a 12-hour window of observations from 155 Swiss weather
stations, forecast five variables (temperature, pressure, humidity, wind_u,
wind_v) at 13 lead times from 0 to 6 hours. A configurable fraction of stations
is hidden from the encoder, so the model must also reconstruct them from their
neighbours — forecasting and spatial gap-filling in one objective.

**Start here:** [`EXPERIMENTS.md`](EXPERIMENTS.md) maps every checkpoint to its
configuration and its results.

---

## Setup

```bash
pip install -r requirements.txt
python src/scripts/verify_env.py      # ALWAYS run this: a CUDA/driver mismatch
                                      # does not raise, it silently uses CPU
```

Fetch the dataset (~4 GB):

```bash
python src/download.py                       # -> ./PeakWeatherDataset
DATA_ROOT=/somewhere python src/download.py  # or elsewhere
```

Every launcher reads `DATA_ROOT` from the environment and falls back to the
Renku path, so nothing needs editing to run on a different machine:

```bash
export DATA_ROOT=/path/to/PeakWeatherDataset
```

Weights & Biases is optional. Set `WANDB_API_KEY` or run `wandb login`; without
it the run logs offline and no metric is lost from the checkpoints.

---

## The two models

| | Transformer MAE | LSTM baseline |
|---|---|---|
| code | `src/model/mae.py`, `encoder.py`, `decoder.py`, `embeddings.py` | `src/model/lstm_baseline.py` |
| idea | axial-attention encoder over (time × station) tokens; Δ-query cross-attention decoder; station-level masking | per-station recurrence, no cross-station information |
| purpose | the model under study | isolates what spatial context is worth |

Both share the dataset, splits, per-station normalisation, lead grid and loss,
so every comparison is like-for-like. Architecture details are in
`analysis_outputs/figures/v27_architecture.svg`.

---

## Workflows

Always launch through `src/scripts/`. Each script locates itself, `cd`s into
`src/`, and writes to the project root, so it works from any directory.

### Transformer MAE

```bash
# TRAIN — edit the config block at the top, or override by environment
SANITY=1 bash src/scripts/run_full_cloud.sh     # ~minutes, smoke test
bash src/scripts/run_full_cloud.sh              # full run -> checkpoints/<SAVE_DIR>/

# EVALUATE / PREDICT — writes test_results/<RUN_NAME>/best_mr<R>/predictions.pt
RUN_NAME=v27 MASK_RATIOS="0.0 0.5" bash src/scripts/run_test_cloud.sh
```

`run_test_cloud.sh` accepts `RUN_NAME`, `MASK_RATIOS`, `SEED`, `BATCH_SIZE` and
`DATA_ROOT` from the environment. Architecture is read from the checkpoint's
saved config, so the model is always rebuilt exactly as trained.

### LSTM baseline

```bash
bash src/scripts/run_lstm_cloud.sh          # trains, then chains into evaluation
bash src/scripts/run_lstm_test_cloud.sh     # evaluate an existing LSTM checkpoint
```

### Reproducing an existing result

```bash
RUN_NAME=v27 MASK_RATIOS="0.5" SEED=42 BATCH_SIZE=4 \
  bash src/scripts/run_test_cloud.sh
```

Reproducing the *masked set* additionally requires the same `SEED`,
`BATCH_SIZE`, `INDEX_MODE` and `STRIDE` — the mask is drawn from the global RNG,
and those four determine how much randomness each pass consumes.

---

## Repository structure

```
weather-station-model-with-transformers/
├── EXPERIMENTS.md          ← checkpoint → config → results mapping. Read first.
├── README.md
├── CITATION.cff            how to cite this work
├── LICENSING.md            licence not yet chosen — see the file
├── requirements.txt
│
├── src/
│   ├── data/dataset.py       windowing, splits, per-station normalisation
│   ├── engine/evaluate.py    prediction collection (metrics live in notebooks/)
│   ├── model/                MAE (encoder/decoder/embeddings) + LSTM baseline
│   ├── main.py               train the transformer
│   ├── test.py               evaluate a transformer → predictions.pt
│   ├── train_lstm.py         train the LSTM
│   ├── test_lstm.py          evaluate the LSTM
│   ├── download.py           fetch PeakWeather
│   ├── inspect_checkpoints.py  print the saved config of every checkpoint
│   └── scripts/              launchers — run these, not the .py directly
│
├── notebooks/
│   ├── analysis/           ← the structured suite; start at 01
│   │   ├── common.py         shared loaders, station table, caching, metrics
│   │   └── 01…13_*.ipynb     audit → baselines → stations → ablations → terrain
│   ├── Test_Results_Exploration.ipynb   main results notebook
│   ├── Station_MAE_Map.ipynb            per-station maps on Swiss topography
│   ├── Train_Test_Exploration.ipynb     dataset QC
│   └── …                                supporting exploration
│
├── analysis_outputs/       generated figures, tables and caches
├── archive/                superseded code, kept for provenance (see its README)
├── docs/, report/          written material
├── meetings/               supervisor decks, by date
│
├── checkpoints/            ┐ NOT in git, and not synchronised.
├── test_results/           ├ Kept locally; ~1.6–2 GB per predictions.pt.
└── PeakWeatherDataset/     ┘
```

---

## Tests

**The pytest suite is not in this copy of the repository.** It lives on the
Renku session box and has to be copied across; `requirements.txt` still installs
pytest so it runs as soon as `tests/` is restored. Source comments reference the
individual files it contains — `test_station_order.py`,
`test_shared_embeddings.py`, `test_station_local_decoder.py`,
`test_embedding_numerics.py`, `test_token_balance.py` — which is a reasonable
inventory of what to look for. **[VERIFY]** before handover.

Once restored:

```bash
python -m pytest tests/ -v
```

---

## Notebooks

`notebooks/analysis/` is the structured suite and shares one module,
`common.py`, which owns dataset discovery, the station table, caching and the
metric helpers. Run **01 first** — it asserts every structural assumption
(station ordering against the dumps, bit-identical targets across runs, mask
recovery, the normalisation round-trip) and stops if any fails.

Then 02 (builds the aggregation cache, a few minutes), and 03–13 in any order.
Results are cached under `analysis_outputs/cache/`, so only the first run is
slow.

`Test_Results_Exploration.ipynb` is the standalone main results notebook and
does not depend on the suite.

Notebooks never import from each other, and `src/` never imports a notebook.

### Import convention

**No package re-exports anything.** `data/__init__.py`, `model/__init__.py` and
`engine/__init__.py` hold a docstring and nothing else, so always import from
the module:

```python
from data.dataset import load_peakweather      # yes
from data import load_peakweather              # no — not exported
```

A package `__init__.py` runs on *every* import of any of its submodules, so a
re-export there is paid by every process. Before this rule was enforced,
`data/__init__.py` pulled geopandas and rioxarray into every training run, and
`model/__init__.py` pulled pytorch_lightning into `test.py`, which never uses
it. Module layering is documented in `model/__init__.py`; there are no import
cycles.

---

## Checkpoints and results

Both directories are gitignored and were **not** synchronised to GitHub. They
are present locally and must be copied by hand when moving machines.

```
checkpoints/<run>/best.ckpt              trained weights + the config used
test_results/<run>/best_mr<R>/predictions.pt   raw tensors, no metrics
```

`predictions.pt` holds normalised predictions and targets only. Every metric is
derived downstream in the notebooks, so there is one source of truth and the
script cannot disagree with the analysis. Contents and verified properties are
documented in [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Scientific conventions worth knowing

* **Δ=0 is not a forecast.** The target is the last observation, so for a
  visible station the answer is an input. It is excluded from forecast averages
  and reported separately as reconstruction.
* **Normalisation is per station and per variable**, fitted on the training
  years only. Errors are converted to physical units with the same statistics.
* **Baselines are computed from raw observations**, not by denormalising —
  persistence carries `x(t0)` forward, the 24h-lag uses `x(t−24h)`.
* **Time-of-day analyses bin in UTC.** Civil local time aliases against the
  90-minute origin stride via DST and produces spurious structure.
