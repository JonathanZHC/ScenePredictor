"""Multi-view tracked-instance scene-flow prediction pipeline."""

from .config import PipelineConfig, load_config

__all__ = [
    "PipelineConfig",
    "ScenePredictionPipeline",
    "load_config",
]


def __getattr__(name: str):
    # Keep lightweight data/config imports usable without eagerly loading the
    # CUDA model dependencies. The production entrypoint still gets the exact
    # same public ScenePredictionPipeline symbol.
    if name == "ScenePredictionPipeline":
        from .pipeline import ScenePredictionPipeline

        return ScenePredictionPipeline
    raise AttributeError(name)
