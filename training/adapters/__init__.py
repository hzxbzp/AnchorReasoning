"""Backbone adapters: ``get_adapter(name)`` / ``REGISTRY``."""
from training.adapters.base import (ABS_PIXEL_DESC, NORM1000_DESC, AbsPixelCodec, BackboneAdapter,
                                   Norm1000Codec, PointCodec, make_point_codec)
from training.adapters.registry import REGISTRY, get_adapter

__all__ = ["ABS_PIXEL_DESC", "NORM1000_DESC", "AbsPixelCodec", "BackboneAdapter", "Norm1000Codec",
           "PointCodec", "make_point_codec", "REGISTRY", "get_adapter"]
