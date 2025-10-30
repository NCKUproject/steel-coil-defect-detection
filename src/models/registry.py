"""Simple model registry to build models by name."""

from __future__ import annotations

from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}


def register(name: str) -> Callable:
    """Decorator to register a model builder under a name."""

    def _wrap(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return _wrap


def build(name: str, *args, **kwargs):
    """Build a model by name using the registered builder."""
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' is not registered. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](*args, **kwargs)
