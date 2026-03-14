# model/__init__.py
from .embeddings import (
    SpatialEmbedding,
    TemporalEmbedding,
    DeltaTimeEmbedding,
    VariableProjection,
    encode_spatial_static,
    encode_temporal,
    compute_spatial_normalization,
    VARIABLE_NAMES,
    NUM_VARIABLES,
    SPATIAL_FEATURE_NAMES,
    SPATIAL_INPUT_DIM,
    TEMPORAL_INPUT_DIM,
)
from .encoder import StationMAEEncoder