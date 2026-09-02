"""
model/mae.py

Station-MAE: masked autoencoder for a weather-station network.

    x (B, W, N, V), x_mask, spatial, x_hours
        -> StationMAEEncoder   (hides a fraction of the stations)
        -> StationMAEDecoder   (one query per station and lead time)
        -> preds (B, K, N, V_t)  [+ log_var with the Gaussian objective]

Residual head: for a station the encoder could see, the decoder output is
added to that station's last observation (persistence); for a masked station
the base is zero, so the decoder must produce the full value from the other
stations.

Loss: Huber (delta = 1 in normalised units) or heteroscedastic Gaussian NLL,
averaged over every station with a present sensor, masked and visible alike,
and averaged uniformly over the K lead times.
"""

import torch
import torch.nn as nn

from .encoder import StationMAEEncoder
from .decoder import StationMAEDecoder
from .embeddings import (
    NUM_VARIABLES,
    NUM_TARGET_VARIABLES,
    TEMPORAL_FOURIER_DIM,
    STATION_CHAR_DIM,
    POSITION_FOURIER_DIM,
    StepIndexEmbedding,
    PositionalEmbedding,
    StationEmbedding,
    TemporalEmbedding,
)


class StationMAE(nn.Module):
    """
    Args:
        d_model, enc_heads, enc_layers, dec_heads, dec_layers, mlp_ratio, dropout:
                              transformer size.
        mask_ratio:           fraction of stations hidden from the encoder.
        use_checkpoint:       gradient checkpointing in encoder and decoder.
        encoder_spatial_attn: False removes cross-station attention from the encoder.
        temporal_patch:       time steps merged into one encoder token.
        station_local_decoder: decoder attends within one station only
                              (requires mask_ratio = 0).
        drop_path_rate:       maximum stochastic-depth rate.
        residual_head:        add the last observation of visible stations to the output.
        var_weights:          per-variable loss weights (default uniform).
        use_nll_loss:         Gaussian NLL objective with a log-variance head.
        window_size:          input window W (decoder step indices are W-1+delta).
        num_horizons:         K, recorded for reference.
    """

    # Keys of the checkpoint ``hyper_parameters["cfg"]`` dict that define the
    # architecture, mapped to constructor arguments. Used by ``from_cfg``.
    _CFG_TO_ARG = {
        "d_model":               "d_model",
        "enc_heads":             "enc_heads",
        "enc_layers":            "enc_layers",
        "dec_heads":             "dec_heads",
        "dec_layers":            "dec_layers",
        "mlp_ratio":             "mlp_ratio",
        "mask_ratio":            "mask_ratio",
        "window":                "window_size",
        "encoder_spatial_attn":  "encoder_spatial_attn",
        "temporal_patch":        "temporal_patch",
        "residual_head":         "residual_head",
        "station_local_decoder": "station_local_decoder",
        "drop_path_rate":        "drop_path_rate",
        "var_weights":           "var_weights",
        "use_nll_loss":          "use_nll_loss",
    }

    @classmethod
    def from_cfg(cls, cfg: dict, **overrides) -> "StationMAE":
        """Build a model from a checkpoint's ``hyper_parameters["cfg"]``; overrides win."""
        kw = {arg: cfg[key] for key, arg in cls._CFG_TO_ARG.items() if key in cfg}
        if "max_delta" in cfg:
            stride = int(cfg.get("delta_grid_stride", 3) or 3)
            kw["num_horizons"] = int(cfg["max_delta"]) // stride + 1
        kw.update(overrides)
        return cls(**kw)

    def __init__(
        self,
        d_model:               int   = 128,
        enc_heads:             int   = 4,
        enc_layers:            int   = 4,
        dec_heads:             int   = 4,
        dec_layers:            int   = 2,
        mlp_ratio:             float = 4.0,
        dropout:               float = 0.1,
        mask_ratio:            float = 0.5,
        num_vars:              int   = NUM_VARIABLES,
        num_target_vars:       int   = NUM_TARGET_VARIABLES,
        fourier_dim:           int   = TEMPORAL_FOURIER_DIM,
        use_checkpoint:        bool  = False,
        encoder_spatial_attn:  bool  = True,
        temporal_patch:        int   = 1,
        num_horizons:          int   = 13,
        station_local_decoder: bool  = False,
        drop_path_rate:        float = 0.0,
        residual_head:         bool  = False,
        var_weights:           "list | None" = None,
        use_nll_loss:          bool  = False,
        window_size:           int   = 72,
    ):
        super().__init__()
        self.mask_ratio            = mask_ratio
        self.num_vars              = num_vars
        self.num_target_vars       = num_target_vars
        self.use_nll_loss          = use_nll_loss
        self.residual_head         = bool(residual_head)
        self.num_horizons          = int(num_horizons)
        self.station_local_decoder = bool(station_local_decoder)
        self.huber_delta           = 1.0

        if self.station_local_decoder:
            assert mask_ratio == 0.0, (
                "station_local_decoder needs every station present in the encoder; "
                f"got mask_ratio={mask_ratio}.")

        # Unused buffer kept so that the saved checkpoints load with strict=True.
        self.register_buffer("persist_mse", torch.ones(num_target_vars, dtype=torch.float32))

        w = torch.tensor(var_weights if var_weights else [1.0] * num_target_vars,
                         dtype=torch.float32)
        self.register_buffer("var_weights", w)

        # Embeddings shared between encoder and decoder.
        shared_step_emb     = StepIndexEmbedding(d_model=d_model, dropout=dropout)
        shared_pos_emb      = PositionalEmbedding(d_model=d_model, fourier_dim=POSITION_FOURIER_DIM,
                                                  dropout=dropout)
        shared_station_emb  = StationEmbedding(d_model=d_model, input_dim=STATION_CHAR_DIM,
                                               dropout=dropout)
        shared_temporal_emb = TemporalEmbedding(d_model=d_model, fourier_dim=fourier_dim,
                                                dropout=dropout)

        self.encoder = StationMAEEncoder(
            d_model=d_model, num_heads=enc_heads, num_layers=enc_layers,
            mlp_ratio=mlp_ratio, dropout=dropout, mask_ratio=mask_ratio,
            num_vars=num_vars, fourier_dim=fourier_dim, use_checkpoint=use_checkpoint,
            spatial_attn=encoder_spatial_attn, temporal_patch=temporal_patch,
            drop_path_rate=drop_path_rate,
            step_emb=shared_step_emb, pos_emb=shared_pos_emb,
            station_emb=shared_station_emb, temporal_emb=shared_temporal_emb,
        )
        self.decoder = StationMAEDecoder(
            d_model=d_model, num_heads=dec_heads, num_layers=dec_layers,
            mlp_ratio=mlp_ratio, dropout=dropout, num_vars=num_vars,
            num_target_vars=num_target_vars, fourier_dim=fourier_dim,
            use_checkpoint=use_checkpoint, station_local=station_local_decoder,
            drop_path_rate=drop_path_rate, predict_uncertainty=use_nll_loss,
            window_size=window_size,
            step_emb=shared_step_emb, pos_emb=shared_pos_emb,
            station_emb=shared_station_emb, temporal_emb=shared_temporal_emb,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _station_masked(masked_idx, B, N, device) -> torch.Tensor:
        """(B, N) bool, True where the encoder could not see the station."""
        flag = torch.zeros(B, N, dtype=torch.bool, device=device)
        if masked_idx is not None and masked_idx.numel() > 0:
            flag.scatter_(1, masked_idx, True)
        return flag

    def _persistence_base(self, x, x_mask, masked_idx):
        """
        (B, N, V_t) last observation of visible stations (0 where the sensor is
        absent or the station is masked); None when the residual head is off.
        """
        if not self.residual_head:
            return None
        Vt   = self.num_target_vars
        base = x[:, -1, :, :Vt] * x_mask[:, -1, :, :Vt]
        if masked_idx is not None and masked_idx.numel() > 0:
            B, N = base.shape[:2]
            vis = torch.ones(B, N, 1, dtype=base.dtype, device=base.device)
            vis.scatter_(1, masked_idx.unsqueeze(-1), 0.0)
            base = base * vis
        return base

    def _decode(self, x, x_mask, spatial, x_hours, y_hours, delta_steps):
        """Encoder + decoder + residual base. Returns (preds, log_var|None, masked_idx)."""
        encoded, masked_idx, _ = self.encoder(x, x_mask, spatial, x_hours)
        if self.station_local_decoder and masked_idx is not None and masked_idx.numel() > 0:
            raise RuntimeError(
                f"station_local_decoder needs every station present, but "
                f"{masked_idx.shape[-1]} of {x.shape[2]} stations were masked "
                f"(encoder.mask_ratio={self.encoder.mask_ratio}).")

        out = self.decoder(
            encoded, spatial, y_hours, delta_steps,
            station_masked=self._station_masked(masked_idx, x.shape[0], x.shape[2], x.device),
        )
        preds, log_var = out if self.use_nll_loss else (out, None)

        base = self._persistence_base(x, x_mask, masked_idx)
        if base is not None:
            preds = preds + (base.unsqueeze(1) if preds.dim() == 4 else base)
        return preds, log_var, masked_idx

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x, x_mask, spatial, x_hours, y, y_mask, y_hours, delta_steps):
        """
        Single lead time per sample: y, y_mask (B, N, V); y_hours, delta_steps (B,).
        Returns (loss, preds (B, N, V_t), masked_idx).
        """
        preds, log_var, masked_idx = self._decode(x, x_mask, spatial, x_hours, y_hours, delta_steps)
        Vt   = self.num_target_vars
        loss = self._supervised_loss(preds, y[..., :Vt], y_mask[..., :Vt], log_var=log_var)
        return loss, preds, masked_idx

    def forward_multi_delta(self, x, x_mask, spatial, x_hours, y, y_mask, y_hours,
                            delta_steps, return_log_var: bool = False):
        """
        K lead times per sample: y, y_mask (B, K, N, V); y_hours, delta_steps (B, K).
        The encoder runs once and the decoder processes all N*K queries in one pass.
        Returns (loss, preds (B, K, N, V_t), masked_idx[, log_var]).
        """
        K = delta_steps.shape[1]
        preds, log_var, masked_idx = self._decode(x, x_mask, spatial, x_hours, y_hours, delta_steps)

        Vt   = self.num_target_vars
        loss = preds.new_zeros(())
        for k in range(K):
            lv_k = log_var[:, k] if log_var is not None else None
            loss = loss + self._supervised_loss(
                preds[:, k], y[:, k, :, :Vt], y_mask[:, k, :, :Vt], log_var=lv_k) / K

        if return_log_var:
            return loss, preds, masked_idx, log_var
        return loss, preds, masked_idx

    @torch.no_grad()
    def predict(self, x, x_mask, spatial, x_hours, y_hours, delta_steps):
        """Inference without loss. Returns the mean prediction."""
        self.eval()
        preds, _, _ = self._decode(x, x_mask, spatial, x_hours, y_hours, delta_steps)
        return preds

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _supervised_loss(self, preds, y, y_mask, log_var=None) -> torch.Tensor:
        """
        Per-variable loss over present sensors, weighted by ``var_weights``
        (uniform weights in NLL mode, where sigma^2 already balances the terms).

            Huber:  0.5 err^2 for |err| <= 1, |err| - 0.5 otherwise
            NLL:    0.5 (err^2 exp(-log_var) + log_var), log_var clamped to [-10, 10]
        """
        ok = y_mask.bool()
        V  = preds.shape[-1]
        weights = preds.new_ones(V) if log_var is not None else self.var_weights[:V]

        per_var = []
        for v in range(V):
            err = preds[..., v] - y[..., v]
            if log_var is not None:
                lv   = log_var[..., v].clamp(-10.0, 10.0)
                elem = 0.5 * (err.pow(2) * (-lv).exp() + lv)
            else:
                abs_err = err.abs()
                elem = torch.where(abs_err <= self.huber_delta,
                                   0.5 * err.pow(2),
                                   self.huber_delta * (abs_err - 0.5 * self.huber_delta))
            okv = ok[..., v].float()
            per_var.append((elem * okv).sum() / okv.sum().clamp(min=1.0))

        return (weights * torch.stack(per_var)).sum() / weights.sum()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
