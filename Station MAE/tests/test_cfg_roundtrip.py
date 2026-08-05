"""
tests/test_cfg_roundtrip.py — the checkpoint cfg must rebuild the same model.

The defect this guards
----------------------
main.py and test.py each built StationMAE from their own hand-maintained kwargs
list. They drifted. test.py never read ``value_embedding``, ``wind_encoder``,
``static_in_token``, ``direct_head`` or ``readout``, so every v18+ checkpoint
silently rebuilt itself with the pre-v18 defaults — a "linear" per-variable map
where the checkpoint held an MLP. ``load_state_dict(strict=False)`` then dropped
the trained tensors into ``unexpected`` and reinitialised the replacements.

The structural guard in test.py caught it, but only after a four-minute dataset
build, and only because ``var_proj`` happens to be in its watch list. Nothing
would have caught a drifted *loss* setting, which changes no tensor shapes.

``StationMAE._CFG_TO_ARG`` is now the single table both callers use. These tests
pin it from both ends: every entry must name a real constructor argument, and
every architecture-affecting argument must be reachable from a cfg.

No torch execution beyond construction; no checkpoint or dataset needed.
"""

import ast
import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.mae import StationMAE                                     # noqa: E402


# ---------------------------------------------------------------------------
# The table matches the constructor
# ---------------------------------------------------------------------------

def _ctor_params():
    sig = inspect.signature(StationMAE.__init__)
    return {n for n in sig.parameters if n != "self"}


def test_every_table_entry_is_a_real_argument():
    unknown = set(StationMAE._CFG_TO_ARG.values()) - _ctor_params()
    assert not unknown, f"_CFG_TO_ARG names arguments StationMAE does not take: {unknown}"


def test_no_duplicate_targets():
    """Two cfg keys writing the same argument would make the winner arbitrary."""
    vals = list(StationMAE._CFG_TO_ARG.values())
    dupes = {v for v in vals if vals.count(v) > 1}
    assert not dupes, f"two cfg keys map to the same argument: {dupes}"


# Arguments deliberately NOT driven by a cfg key, with the reason. Anything that
# affects the architecture and is missing from both this set and the table is a
# drift bug waiting to happen, so the test below fails on it.
_EXEMPT = {
    "self",
    "wind_pair",       # derived from the boolean cfg key `wind_encoder`
    "num_horizons",    # derived from max_delta // delta_grid_stride + 1
    "use_nll_loss",    # inferred from the WEIGHTS; the cfg key is unreliable pre-v14
    "dropout",         # always overridden to 0.0 at inference
    "use_checkpoint",  # training-time memory tactic, no effect on the weights
    "num_vars", "num_target_vars", "fourier_dim",   # dataset constants
}


def test_every_architecture_argument_is_reachable():
    covered = set(StationMAE._CFG_TO_ARG.values()) | _EXEMPT
    missing = _ctor_params() - covered
    assert not missing, (
        f"these StationMAE arguments cannot be restored from a checkpoint cfg: "
        f"{sorted(missing)}. Add them to _CFG_TO_ARG, or to _EXEMPT with a reason.")


def test_main_records_every_key_the_table_reads():
    """
    The table is only useful if main.py actually writes those keys into cfg.
    Parsed statically — importing main.py would require the dataset.
    """
    src = (ROOT / "src" / "main.py").read_text()
    tree = ast.parse(src)
    written = {
        k.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    missing = set(StationMAE._CFG_TO_ARG) - written
    assert not missing, (
        f"_CFG_TO_ARG reads cfg keys main.py never writes: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Round trip: cfg -> model -> the settings actually took effect
# ---------------------------------------------------------------------------

_CFG = {
    "d_model": 64, "enc_heads": 4, "enc_layers": 1,
    "dec_heads": 4, "dec_layers": 1, "mlp_ratio": 2.0,
    "mask_ratio": 0.5, "window": 12,
    "factorised_encoder": True, "temporal_window": 0, "temporal_patch": 3,
    "value_embedding": "mlp", "wind_encoder": True,
    "static_in_token": False, "residual_head": True,
    "cross_attn_decoder": True, "direct_head": False, "readout": "last",
    "max_delta": 36, "delta_grid_stride": 3,
}


def test_roundtrip_sets_the_structural_flags():
    m = StationMAE.from_cfg(_CFG)
    assert m.residual_head is True
    assert m.direct_head is False
    assert m.mask_ratio == 0.5
    assert m.encoder.temporal_patch == 3
    assert m.encoder.var_proj.value_embedding == "mlp"
    assert m.encoder.var_proj.wind_pair == (3, 4), "wind_encoder did not become wind_pair"
    assert m.num_horizons == 13, "36 // 3 + 1"


def test_overrides_beat_the_cfg():
    m = StationMAE.from_cfg(_CFG, dropout=0.0, mask_ratio=0.0)
    assert m.mask_ratio == 0.0


def test_unknown_cfg_keys_are_ignored():
    """A cfg carries optimiser and data settings too; they must not reach __init__."""
    noisy = dict(_CFG, lr=1e-4, batch_size=4, data_root="/nowhere", epochs=100)
    m = StationMAE.from_cfg(noisy)
    assert m.mask_ratio == 0.5


def test_empty_cfg_gives_constructor_defaults():
    """Legacy .pt checkpoints carry no cfg; the model must still build."""
    m = StationMAE.from_cfg({})
    assert m.residual_head is False and m.direct_head is False


def test_a_v22_cfg_builds_the_direct_head():
    cfg = dict(_CFG, direct_head=True, mask_ratio=0.0, residual_head=False)
    m = StationMAE.from_cfg(cfg)
    assert m.direct_head and hasattr(m, "direct_proj")
    assert m.direct_proj.out_features == m.num_horizons * m.num_target_vars_


def test_direct_head_still_refuses_masking_through_from_cfg():
    cfg = dict(_CFG, direct_head=True, mask_ratio=0.5)
    with pytest.raises(AssertionError, match="mask_ratio"):
        StationMAE.from_cfg(cfg)


# ---------------------------------------------------------------------------
# Removed options stay removed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gone", [
    "joint_encoder",          # JointSpatioTemporalBlock, deleted in the v22 cleanup
    "patch_schedule",         # telescopic tokenization, never selected by any run
])
def test_deleted_arguments_are_gone(gone):
    assert gone not in _ctor_params(), (
        f"{gone} is back in the StationMAE signature — it was deleted along with "
        f"the code path behind it")
