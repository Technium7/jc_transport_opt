"""Field-conditioned probabilistic surrogates for final motional heating.

Import ``ion_npe.surrogate.model`` for the networks. The package init stays
light so the dataset tools do not require PyTorch until a model is built.
"""

__all__ = ["JOINT_DIM", "legal_bounds", "joint_tokens", "field_features"]


def __getattr__(name):
    if name in __all__:
        from . import layout
        return getattr(layout, name)
    if name in ("SurrogateConfig", "SurrogatePredictor", "EnsemblePredictor"):
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
