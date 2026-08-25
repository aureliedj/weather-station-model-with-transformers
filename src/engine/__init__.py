"""Evaluation utilities.

Nothing is re-exported eagerly: `engine/__init__.py` is executed by any
`from engine.evaluate import ...`, so a re-export here would make every
training run import the whole evaluation stack. Import from the module:

    from engine.evaluate import collect_predictions
"""
