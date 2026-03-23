# model/__init__.py
from .embeddings import (
    # Embedding modules
    SpatialEmbedding,
    TemporalEmbedding,        # multi-scale Fourier encoding (Aurora-inspired)
    DeltaTimeEmbedding,
    VariableProjection,
    # Helper functions
    encode_spatial_static,
    encode_temporal,          # returns hours-since-epoch float for TemporalEmbedding
    compute_spatial_normalization,
    # Constants
    VARIABLE_NAMES,
    NUM_VARIABLES,
    SPATIAL_FEATURE_NAMES,
    SPATIAL_INPUT_DIM,
    TEMPORAL_FOURIER_DIM,
)
from .encoder import StationMAEEncoder, TransformerBlock
from .decoder import StationMAEDecoder
from .mae    import StationMAE

__all__ = [
    "StationMAE",
    "StationMAEEncoder",
    "StationMAEDecoder",
    "TransformerBlock",
    "SpatialEmbedding",
    "TemporalEmbedding",
    "DeltaTimeEmbedding",
    "VariableProjection",
    "encode_spatial_static",
    "encode_temporal",
    "compute_spatial_normalization",
    "VARIABLE_NAMES",
    "NUM_VARIABLES",
    "SPATIAL_FEATURE_NAMES",
    "SPATIAL_INPUT_DIM",
    "TEMPORAL_FOURIER_DIM",
]
