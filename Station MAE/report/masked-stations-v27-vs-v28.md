# Masked stations only: v27 vs v28, per variable and lead time

**The condition:** MR = 0.5, restricted to stations hidden from the encoder.
Their own observations are unavailable, so cross-station information is the
*only* possible source of signal. This is the sharpest available test of what
encoder spatial attention contributes.

**The contrast:** v27 and v28 differ in `encoder_spatial_attn` only
(True vs False). Δ% below is `(v28 − v27) / v27`; **negative = spatial attention
OFF is better**.

📊 `report/figures/fig6_masked_lead_by_variable.png`

---

## Validity check performed first ⚠️

The two evaluation runs drew **independent random masks**. Measured:

- 0 / 11,684 windows share the same masked station set
- mean overlap 38.1 / 77 stations = **49.5 %** (chance level for two independent
  draws of 77 from 155 is 49.7 %)

So this is an **unpaired** comparison. To check whether that biases it, I
recomputed every number station-balanced (per-station MAE first, then an
unweighted mean over stations, which removes any effect of one run happening to
mask harder stations more often):

| lead (h) | Δ% raw (as drawn) | Δ% station-balanced |
|---|---|---|
| 0.0 | −0.76 | −0.74 |
| 1.0 | −0.38 | −0.35 |
| 3.0 | +0.03 | +0.05 |
| 6.0 | +0.27 | +0.26 |

Agreement is within 0.03 pp at every lead, and per-station masked-window counts
differ by only 1.0 % on average (max 3.6 %). **The unpaired draw does not bias
the comparison.** Raw numbers are used below.

---

## Normalised MAE — masked stations only

| lead (h) | temp v27 | temp v28 | **Δ%** | pres v27 | pres v28 | **Δ%** | hum v27 | hum v28 | **Δ%** | wind_u v27 | wind_u v28 | **Δ%** | wind_v v27 | wind_v v28 | **Δ%** | all v27 | all v28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.1029 | 0.0980 | **−4.79** | 0.0635 | 0.0618 | **−2.67** | 0.2600 | 0.2605 | +0.20 | 0.4276 | 0.4233 | −1.01 | 0.4424 | 0.4430 | +0.12 | 0.2629 | 0.2609 |
| 0.5 | 0.1024 | 0.0979 | **−4.32** | 0.0644 | 0.0620 | **−3.70** | 0.2596 | 0.2603 | +0.26 | 0.4249 | 0.4218 | −0.74 | 0.4412 | 0.4425 | +0.29 | 0.2621 | 0.2604 |
| 1.0 | 0.1028 | 0.0991 | −3.53 | 0.0666 | 0.0648 | −2.66 | 0.2621 | 0.2630 | +0.34 | 0.4264 | 0.4241 | −0.55 | 0.4430 | 0.4449 | +0.44 | 0.2637 | 0.2627 |
| 1.5 | 0.1046 | 0.1012 | −3.25 | 0.0698 | 0.0680 | −2.65 | 0.2672 | 0.2680 | +0.31 | 0.4305 | 0.4281 | −0.56 | 0.4482 | 0.4497 | +0.34 | 0.2676 | 0.2666 |
| 2.0 | 0.1063 | 0.1037 | −2.48 | 0.0731 | 0.0724 | −0.92 | 0.2732 | 0.2743 | +0.42 | 0.4341 | 0.4332 | −0.21 | 0.4534 | 0.4545 | +0.25 | 0.2716 | 0.2712 |
| 2.5 | 0.1093 | 0.1069 | −2.19 | 0.0770 | 0.0767 | −0.36 | 0.2800 | 0.2810 | +0.37 | 0.4394 | 0.4393 | −0.04 | 0.4589 | 0.4598 | +0.18 | 0.2765 | 0.2763 |
| 3.0 | 0.1121 | 0.1103 | −1.63 | 0.0815 | 0.0818 | +0.30 | 0.2879 | 0.2886 | +0.23 | 0.4456 | 0.4459 | +0.07 | 0.4653 | 0.4665 | +0.26 | 0.2821 | 0.2822 |
| 3.5 | 0.1152 | 0.1137 | −1.30 | 0.0868 | 0.0870 | +0.17 | 0.2956 | 0.2962 | +0.18 | 0.4510 | 0.4516 | +0.14 | 0.4707 | 0.4725 | +0.37 | 0.2875 | 0.2878 |
| 4.0 | 0.1188 | 0.1175 | −1.11 | 0.0924 | 0.0925 | +0.07 | 0.3034 | 0.3040 | +0.20 | 0.4569 | 0.4578 | +0.19 | 0.4766 | 0.4781 | +0.32 | 0.2932 | 0.2936 |
| 4.5 | 0.1223 | 0.1213 | −0.76 | 0.0981 | 0.0990 | +0.90 | 0.3122 | 0.3125 | +0.09 | 0.4642 | 0.4651 | +0.19 | 0.4834 | 0.4842 | +0.17 | 0.2997 | 0.3000 |
| 5.0 | 0.1259 | 0.1250 | −0.75 | 0.1043 | 0.1054 | +1.06 | 0.3204 | 0.3203 | −0.04 | 0.4700 | 0.4707 | +0.16 | 0.4897 | 0.4902 | +0.11 | 0.3057 | 0.3059 |
| 5.5 | 0.1296 | 0.1289 | −0.53 | 0.1105 | 0.1131 | **+2.36** | 0.3287 | 0.3286 | −0.05 | 0.4765 | 0.4775 | +0.20 | 0.4953 | 0.4958 | +0.09 | 0.3118 | 0.3123 |
| 6.0 | 0.1336 | 0.1329 | −0.58 | 0.1170 | 0.1201 | **+2.60** | 0.3371 | 0.3370 | −0.03 | 0.4825 | 0.4845 | +0.42 | 0.5014 | 0.5018 | +0.08 | 0.3179 | 0.3188 |

## Physical units — masked stations only

| lead (h) | temp v27 (°C) | temp v28 | pres v27 (hPa) | pres v28 | hum v27 (%) | hum v28 | wind_u v27 (m/s) | wind_u v28 | wind_v v27 (m/s) | wind_v v28 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.8101 | **0.7713** | 0.4761 | **0.4632** | 5.0712 | 5.0865 | 0.9225 | **0.9101** | 0.8799 | 0.8806 |
| 0.5 | 0.8062 | **0.7712** | 0.4827 | **0.4647** | 5.0646 | 5.0824 | 0.9168 | **0.9071** | 0.8781 | 0.8800 |
| 1.0 | 0.8093 | **0.7806** | 0.4988 | **0.4854** | 5.1099 | 5.1333 | 0.9198 | **0.9129** | 0.8831 | 0.8860 |
| 2.0 | 0.8375 | **0.8167** | 0.5470 | 0.5419 | 5.3220 | 5.3492 | 0.9387 | 0.9348 | 0.9069 | 0.9083 |
| 3.0 | 0.8835 | **0.8692** | 0.6101 | 0.6118 | 5.6060 | 5.6227 | 0.9662 | 0.9654 | 0.9360 | 0.9372 |
| 4.0 | 0.9358 | **0.9256** | 0.6907 | 0.6911 | 5.9051 | 5.9216 | 0.9950 | 0.9960 | 0.9640 | 0.9674 |
| 5.0 | 0.9918 | 0.9846 | 0.7790 | 0.7871 | 6.2336 | 6.2379 | 1.0280 | 1.0284 | 0.9965 | 0.9972 |
| 6.0 | 1.0527 | 1.0467 | 0.8740 | 0.8965 | 6.5570 | 6.5644 | 1.0594 | 1.0627 | 1.0261 | 1.0269 |

---

## What the numbers say *(measured)*

1. **Temperature — spatial attention is actively harmful at every lead.**
   v28 is better by **−4.8 % at t=0**, decaying monotonically to −0.6 % at 6 h.
   In physical terms, 0.810 → 0.771 °C at t=0. This is the largest single
   effect in the table, and it runs *opposite* to the expected direction.

2. **Pressure — sign flips at ~3 h.** v28 better by −3.7 % at 30 min;
   v27 better by +2.6 % at 6 h. Crossover between 2.5 and 3 h.

3. **Humidity — no effect.** |Δ| ≤ 0.5 % at every lead.

4. **wind_u — mirrors pressure**, smaller: −1.0 % at t=0 → +0.4 % at 6 h,
   crossing zero at ~2.5 h.

5. **wind_v — no meaningful effect.** +0.1 % to +0.4 %, always small.

6. **Overall** the two curves cross at 3 h: v28 better below (−0.76 % at t=0),
   v27 better above (+0.27 % at 6 h). Net effect across all leads ≈ 0.

## Interpretation *(my reading — flagged as interpretation)*

The two regimes look like different jobs:

- **Short leads (< 3 h)** are dominated by *spatial interpolation* — inferring a
  hidden station's current state from its neighbours. Here the encoder's
  spatial sub-layer makes things **worse**, most clearly for temperature. A
  plausible reading is that mixing station representations inside the encoder
  blurs the station-specific signal the decoder's cross-attention would
  otherwise retrieve cleanly, i.e. the two spatial pathways partly compete
  rather than compose.

- **Long leads (> 3 h)** are dominated by *advection* — weather arriving from
  elsewhere. Here the encoder's spatial mixing helps slightly, and it helps
  most for **pressure**, the variable with the largest coherent spatial
  structure. That is physically sensible.

The magnitudes are small (max 4.8 %, mostly under 1 %), and the sign is
variable-dependent, so the honest summary is: **encoder spatial attention does
not provide a consistent benefit even on masked stations, and for temperature
it is a consistent penalty.**

## Caveats

- Single seeds. Most effects here are ≤ 1 %, well inside plausible seed noise.
  Only the temperature result (−4.8 % → −0.6 %, monotone across 13 leads) and
  the pressure crossover show a systematic pattern hard to explain by noise.
- v27 has 4.74 M more parameters and trained 40 epochs vs v28's 51 — capacity
  and training length move together with the ablation.
- Both models retain full cross-station access in the decoder, so this measures
  the **encoder** spatial sub-layer's marginal contribution, not whether
  spatial information matters.
- Stations 59 and 83 carry donor-borrowed normalisation statistics; their
  physical-unit values are unreliable (they are 2 of 155, so the pooled effect
  is small).
