from typing import Any, Callable, Dict, Optional, Type, Union

# Avoid importing maps/dynamics here to prevent circular imports.
# Use broad callable/class annotations.
MapFactory = Union[Type[Any], Callable[..., Any]]
DynamicFactory = Union[Type[Any], Callable[..., Any]]

# global registries
MAP_REGISTRY: Dict[str, MapFactory] = {}
DYNAMIC_REGISTRY: Dict[str, DynamicFactory] = {}


def register_map(name: Optional[str] = None):
    def decorator(cls: MapFactory) -> MapFactory:
        key = name or cls.__name__
        MAP_REGISTRY[key] = cls
        return cls

    return decorator


def register_dynamic(name: Optional[str] = None):
    def decorator(cls: DynamicFactory) -> DynamicFactory:
        key = name or cls.__name__
        DYNAMIC_REGISTRY[key] = cls
        return cls

    return decorator


def register_map_class(name: str, factory: MapFactory) -> None:
    MAP_REGISTRY[name] = factory


def register_dynamic_class(name: str, factory: DynamicFactory) -> None:
    DYNAMIC_REGISTRY[name] = factory
