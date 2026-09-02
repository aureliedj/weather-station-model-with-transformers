# Station-MAE: masked-autoencoder Transformers for the Swiss weather-station network

Code for the capstone project *Weather station modelling with Transformers*
(Certificate of Advanced Studies in Data Science, ETH Zürich / Swiss Data Science
Center, in collaboration with MeteoSwiss), built on the
[PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather) dataset
of [Zambon et al. (2025)](https://arxiv.org/abs/2506.13652).

**Task.** Given a 12-hour window of 10-minute observations from 155 MeteoSwiss
stations, predict temperature, pressure, humidity and the two wind components
at every station for lead times from 0 to 6 hours in 30-minute steps. During
training, a fraction of the stations is hidden from the encoder, so the same
model learns to forecast and to reconstruct a missing station from its
neighbours.

**Model.** A Transformer with alternating temporal and cross-station attention
over a (time step × station) token grid; each station is embedded by its LV95
coordinates and 13 topographic descriptors rather than by an identity table;
a decoder conditioned on the lead time produces all lead times in one pass;
the output is added to the station's last observation (persistence residual).

## Repository layout

```
src/
  main.py                 train the Transformer (all variants)
  test.py                 write test-set predictions of a Transformer checkpoint
  train_lstm.py           train the per-station LSTM baseline
  test_lstm.py            write test-set predictions of the LSTM
  download.py             fetch PeakWeather
  inspect_checkpoints.py  print the configuration stored in a checkpoint
  data/dataset.py         PeakWeather loading, normalisation, windowing, station exclusion
  model/embeddings.py     value, position, topography, time, step and lead-time embeddings
  model/encoder.py        station masking, temporal patching, factorised attention blocks
  model/decoder.py        lead-time-conditioned cross-attention decoder
  model/mae.py            full model, residual head, Huber / Gaussian-NLL loss
  model/lightning_module.py
  model/lstm_baseline.py
  engine/evaluate.py      prediction collector used by test.py
scripts/                  the exact configurations of the reported runs
notebooks/analysis/       figures and tables of the report (41-50) + common.py
notebooks/                data exploration, model summary, result maps
```

## Getting started

Tested with Python 3.10, PyTorch 2.x and PyTorch Lightning 2.x on a single GPU.

```bash
pip install -r requirements.txt
python src/download.py            # -> ./PeakWeatherDataset (~4 GB)
export DATA_ROOT=$PWD/PeakWeatherDataset
```

Weights & Biases is optional: set `WANDB_PROJECT` to log there, otherwise
Lightning writes CSV logs next to the checkpoints.

## Training and evaluation

The four Transformer variants and the LSTM baseline of the report:

| variant | script argument | training mask ratio | objective | cross-station attention | run directory |
|---|---|---|---|---|---|
| MAE Transformer | `mae` | 0.5 | Huber | yes | `checkpoints/full_run_cloud_v27` |
| Probabilistic MAE | `prob` | 0.5 | Gaussian NLL | yes | `checkpoints/full_run_cloud_v30-nll` |
| Dense | `dense` | 0.0 | Huber | yes | `checkpoints/full_run_cloud_v31` |
| Spatially Blind | `blind` | 0.0 | Huber | no | `checkpoints/full_run_cloud_v32-blind` |
| LSTM | – | – | Huber | no | `checkpoints/lstm-baseline-v1` |

All Transformer variants share `d_model=384`, 8 encoder blocks, 2 decoder
blocks, 8 heads, temporal patch 3, dropout and drop-path 0.1, 12-hour window,
lead grid 0–6 h every 30 min (K = 13) and 24.99 M trainable parameters
(20.25 M for the Spatially Blind variant, 21.11 M for the LSTM). Station PFA
is excluded from every split.

```bash
# Transformers
bash scripts/train_transformer.sh mae        # or prob | dense | blind
bash scripts/test_transformer.sh  mae 0.0 0.5  # evaluation mask ratios; dense/blind: 0.0 only

# LSTM baseline
bash scripts/train_lstm.sh
bash scripts/test_lstm.sh
```

`test.py` writes one `predictions.pt` per (checkpoint, evaluation mask ratio)
under `test_results/<run>/best_mr<R>/` with the normalised predictions,
targets, sensor masks, hidden-station indices, lead times, timestamps and
static station features (plus `log_var` for the probabilistic model). The
evaluation protocol of every reported number is: sliding test windows with a
90-minute stride over 2023–2024 (11,684 windows), seed 42, batch size 4; at
mask ratio 0.5 a new random set of 77 hidden stations is drawn for every
window.

All entry points accept `--help`; `python src/inspect_checkpoints.py` prints
the configuration saved in each checkpoint.

## Data

`src/data/dataset.py` loads the meteo stations of PeakWeather at 10-minute
frequency with wind as (u, v) components, splits the years into train
(2017–2021), validation (2022) and test (2023–2024), standardises every
(station, variable) pair with training-year statistics (sparse pairs borrow
the statistics of the most similar station), and builds gap-free windows.
The first run builds a tensor cache next to the data (several minutes);
later runs start in seconds.

## Analysis notebooks

`notebooks/analysis/41_…50_*.ipynb` produce the figures and tables of the
report from the `predictions.pt` dumps; `common.py` holds the shared loading,
aggregation and caching code (results are cached under `analysis_outputs/`).
`notebooks/Train_Test_Exploration.ipynb` covers the data, `Model_Architecture.ipynb`
the layer-by-layer summary of both models, `Station_MAE_Map.ipynb` and
`Test_Results_Exploration.ipynb` the per-station maps. The map notebooks need
the swissBOUNDARIES3D shapefile (`SWISSSHAPE` environment variable).

## Citation

If you use this code, please cite the repository (see `CITATION.cff`) and the
PeakWeather dataset:

```
@misc{zambon2025peakweather,
  title={PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning},
  author={Zambon, Daniele and Cattaneo, Michele and Marisca, Ivan and Bhend, Jonas and Nerini, Daniele and Alippi, Cesare},
  year={2025},
  eprint={2506.13652},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.13652},
}
```
