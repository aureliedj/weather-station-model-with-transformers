# Supervisor Discussion — Station-MAE, Current Status

---

## 1. Baselines and Evaluation Standard

**What is the appropriate baseline to compare against?**
The natural competitors for station gap-filling are classical spatial interpolation methods — inverse-distance weighting, ordinary kriging, and thin-plate splines. Do we have benchmark RMSE numbers from any of these on the same 2022 validation year? Without a baseline, it is hard to know whether the current normalised RMSE figures (temperature ~0.79, pressure ~0.21, humidity ~1.48) are good, mediocre, or worse than interpolation.

**Should we evaluate in physical units for any comparisons?**
All metrics are currently in normalised space (zero-mean, unit-variance per variable). For the supervisor and any paper, physical units (°C, hPa, %, m/s) are more interpretable. Is there a preference for reporting both? Denormalising just requires multiplying by `obs_stats["std"]`.

**Is the 2022 validation year representative?**
2022 was an exceptionally dry and hot year in Switzerland (record drought, heat waves). Results on 2022 may be harder than an average year for temperature and humidity. Is it worth also evaluating on a held-out subset of training years to check whether 2022 is an outlier?

---

## 2. Masking Strategy

**Is station-level masking the right granularity?**
We currently mask entire stations across all timesteps in the window. This simulates a station going offline — a realistic gap-filling scenario. However, in practice sensors within a station fail individually (e.g., anemometer breaks while temperature still reports). Should we consider *variable-level* masking as an alternative or supplement?

**What masking ratio is appropriate for the Swiss network?**
The default mask ratio is 50 %. The original image MAE used 75 % because neighbouring pixels are highly correlated. Weather stations are much more sparse and unevenly distributed across Switzerland (many more in the plateau, few in the high Alps). Should the ratio be tuned empirically, or is there a principled argument for a specific value given the network density and decorrelation length scales?

---

## 3. Architecture

**Is the asymmetric encoder/decoder depth well-calibrated?**
The encoder has 4 transformer layers; the decoder has 2. This follows the MAE paper's recommendation for pre-training representations. However, if the primary use case is forecasting (delta > 0) rather than reconstruction (delta = 0), a heavier decoder might help. What is the intended downstream use — gap-filling, short-range forecasting, or both?

**Should we consider cross-attention in the decoder instead of full self-attention?**
The decoder currently concatenates encoder context tokens with station query tokens and runs self-attention over the full sequence. Standard MAE decoders use cross-attention (queries attend over encoder context). The concatenation approach is simpler and allows encoder tokens to attend to each other in the decoder, but scales as O((W·N_vis + N)²). For longer windows or larger station networks, is memory a concern?

**Model size vs. data quantity — risk of overfitting?**
With 156 stations, ~370 k timesteps at 10-minute resolution, and a model with ~1–2 M parameters (to be confirmed), overfitting is plausible, especially if training for many epochs. Has regularisation been considered beyond dropout and weight decay (e.g., stochastic depth, increased masking noise)?

---

## 4. Loss and Training

**Should we weight the loss per variable?**
Humidity currently has ~2× the RMSE of most other variables (normalised RMSE ~1.48 after 5 epochs vs. ~0.7 for wind components). This may reflect genuine difficulty or a poor normalisation (humidity is not Gaussian — it is bounded 0–100 % and often near-saturated). Should the MSE loss be weighted per variable, or should humidity be transformed (e.g., logit) before normalisation?

**How should precipitation be handled long-term?**
Precipitation is currently used as an input signal but excluded from the loss because MSE is ill-suited to a zero-inflated distribution. Is the long-term plan to keep it as context only, or to add a separate head with a more appropriate loss (e.g., Bernoulli for occurrence + gamma for amount)?

**Compute budget and training length?**
The 5-epoch mini-model is clearly still learning (temperature RMSE ~0.79 is well above what a converged model should achieve given spatial correlation in the Swiss network). What is the target training duration and compute platform for the full run?

---

## 5. Data and Feature Engineering

**Are there known data quality issues in the historical record?**
Station relocations, sensor replacements, and prolonged outages can introduce artificial discontinuities in the time series. The normalisation is computed globally, so a station that moved 200 m uphill mid-record would have a biased mean/std. Is there a data quality flag we should respect, or a known list of problematic stations to exclude?

**Are there other topographic or meteorological features worth adding to the spatial embedding?**
The current 15-dim spatial vector includes coordinates, aspects, slope, S-N and W-E derivatives, and TPI. Features that might be informative include: sky-view factor (radiation exposure), distance to nearest water body, or the fraction of surrounding area covered by snow. Are any of these available in the PeakWeather metadata?

**Should we use the raw 10-minute resolution, or aggregate to hourly?**
Training on 10-minute data gives the finest temporal resolution but also the most noise (gusty wind, convective showers). Aggregating to hourly would reduce noise and expand the effective number of independent samples for a given compute budget. What is the scientific requirement?

---

## 6. Regionalisation and Scientific Validity

**Should evaluation be stratified by region or weather regime?**
Switzerland has strongly contrasting climates — the northern plateau, the Alps, and the southern Ticino all behave differently. A global RMSE masks whether the model performs well on the plateau but poorly in complex Alpine terrain (where gap-filling is most needed). Is regime- or region-stratified evaluation expected for publication?

**Is temporal autocorrelation a concern for the train/val split?**
The split is strictly temporal (train 2017–21, val 2022, test 2023–24), which avoids data leakage. However, slow climate trends and multi-year oscillations (NAO, ENSO teleconnections) could mean the validation year is systematically harder or easier than training. Should we report variance across multiple held-out years rather than a single validation year?

---

## 7. Downstream Applications and Publication Target

**What is the primary use case driving design decisions?**
Gap-filling (delta = 0, reconstruct missing stations) and short-range forecasting (delta > 0, predict future states) have different requirements. Gap-filling favours a high masking ratio and rich spatial context; forecasting favours a heavier decoder and potentially autoregressive rollout. Clarifying the primary use case would help prioritise architecture choices.

**Is this intended as a standalone model or a foundation for fine-tuning?**
If the goal is a pre-trained representation that can be fine-tuned for downstream tasks (e.g., extreme-event detection, NWP bias correction), the pre-training objective and model size should reflect that. If it is a production gap-filler, operational constraints (inference latency, model size) matter more.

**What is the target venue and timeline?**
Understanding whether this is headed for a conference (NeurIPS, ICLR, AI4Science) or a journal (GMD, AMS journals) would inform how rigorous the baseline comparisons, ablation studies, and statistical testing need to be.
