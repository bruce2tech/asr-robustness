"""A small registry mapping model keys to factory callables.

Adapters register themselves with :func:`register`; the evaluation runner builds
models with :func:`create`. Keeping this module free of heavy imports (torch,
espnet) means the registry can be inspected cheaply -- the cost is paid only
when a specific model is actually instantiated.
"""

from __future__ import annotations

from typing import Callable

from asr_robustness.models.base import ASRModel

_FACTORIES: dict[str, Callable[..., ASRModel]] = {}


def register(key: str) -> Callable[[Callable[..., ASRModel]], Callable[..., ASRModel]]:
    """Decorator: register a factory (class or function) under ``key``."""

    def decorator(factory: Callable[..., ASRModel]) -> Callable[..., ASRModel]:
        if key in _FACTORIES:
            raise ValueError(f"model key already registered: {key!r}")
        _FACTORIES[key] = factory
        return factory

    return decorator


def create(key: str, **kwargs) -> ASRModel:
    """Instantiate the model registered under ``key``."""
    if key not in _FACTORIES:
        raise KeyError(f"unknown model {key!r}; registered: {sorted(_FACTORIES)}")
    return _FACTORIES[key](**kwargs)


def available() -> list[str]:
    """List the currently-registered model keys."""
    return sorted(_FACTORIES)
