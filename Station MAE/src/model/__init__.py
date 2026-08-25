# model/__init__.py
from .embeddings import (
    # Embedding modules
    PositionalEmbedding,     # p1 — Fourier 2D positional (easting/northing)
    StationEmbedding,        # p2 — MLP over topographic characteristics
    TemporalEmbedding,       # t  — multi-scale Fourier encoding (Aurora-inspired)
    DeltaTimeEmbedding,      # Δt — Fourier lead-time encoding [decoder only]
    VariableProjection,      # v  — per-variable measurement projection
    # Helper functions
    encode_temporal,
    # Constants
    VARIABLE_NAMES,
    NUM_VARIABLES,
    TARGET_VARIABLE_NAMES,
    NUM_TARGET_VARIABLES,
    SPATIAL_FEATURE_NAMES,
    SPATIAL_INPUT_DIM,
    POSITION_DIM,
    STATION_CHAR_DIM,
    POSITION_FOURIER_DIM,
    TEMPORAL_FOURIER_DIM,
    DELTA_FOURIER_DIM,
)
from .encoder          import StationMAEEncoder, TransformerBlock, FactorisedTransformerBlock
from .decoder          import StationMAEDecoder, CrossAttentionBlock
from .mae              import StationMAE
from .lightning_module import StationMAELightning

__all__ = [
    "StationMAE",
    "StationMAELightning",
    "StationMAEEncoder",
    "StationMAEDecoder",
    "TransformerBlock",
    "FactorisedTransformerBlock",
    "CrossAttentionBlock",
    "PositionalEmbedding",
    "StationEmbedding",
    "TemporalEmbedding",
    "DeltaTimeEmbedding",
    "VariableProjection",
    "encode_temporal",
    "VARIABLE_NAMES",
    "NUM_VARIABLES",
    "TARGET_VARIABLE_NAMES",
    "NUM_TARGET_VARIABLES",
    "SPATIAL_FEATURE_NAMES",
    "SPATIAL_INPUT_DIM",
    "POSITION_DIM",
    "STATION_CHAR_DIM",
    "POSITION_FOURIER_DIM",
    "TEMPORAL_FOURIER_DIM",
    "DELTA_FOURIER_DIM",
]
