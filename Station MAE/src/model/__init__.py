"""Station-MAE model components.

Nothing is re-exported eagerly. This file is executed by any
`from model.<mod> import ...`, so a re-export here would pull the entire model
stack — including lightning_module, and therefore pytorch_lightning — into
processes that need only one piece. src/test.py imports `model.mae` and never
touches Lightning; it should not pay for it.

Import from the module:

    from model.embeddings      import TARGET_VARIABLE_NAMES, VariableProjection
    from model.encoder         import StationMAEEncoder
    from model.decoder         import StationMAEDecoder
    from model.mae             import StationMAE
    from model.lightning_module import StationMAELightning
    from model.lstm_baseline   import StationLSTM, StationLSTMLightning
    from model.token_balance   import token_balance, format_report

Every import in this repository already uses that form; the re-export block
that used to live here was unused. Same policy as engine/__init__.py and
data/__init__.py.

Layering (no cycles):

    embeddings
      ├─> encoder ─┐
      ├─> decoder ─┴─> mae ──> lightning_module
      ├─> lstm_baseline
      └─> (engine.evaluate)
    token_balance is standalone.
"""
