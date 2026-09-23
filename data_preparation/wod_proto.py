"""Waymo Open Dataset End-to-End Driving protos, without the official wheel.

The published ``waymo-open-dataset-tf-*`` wheel pulls in TensorFlow and pins an
old numpy, which conflicts with the training environment. This module therefore
resolves the one message the repository needs, ``E2EDFrame``, in two steps:

1. If the real ``waymo_open_dataset`` package is importable, use it as-is, so a
   user who already installed it needs nothing further.
2. Otherwise build the message classes at runtime from a protobuf
   *FileDescriptorSet*. A descriptor set is plain data, so only the ``protobuf``
   runtime is required and there is no protoc-gencode / runtime version
   coupling.

The descriptor set is generated, not shipped. Produce it once with::

    bash data_preparation/make_wod_desc.sh

which writes ``third_party/wod_e2e.desc`` under the repository root. Set the
``WOD_DESC`` environment variable to read it from somewhere else.

Usage::

    from data_preparation.wod_proto import E2EDFrame, camera_name, intent_name
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PKG_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.core.paths import ROOT  # noqa: E402

#: Environment variable overriding the descriptor-set location.
DESC_ENV = "WOD_DESC"
#: Default descriptor-set path (generated; ``third_party/`` is gitignored).
DEFAULT_DESC = os.path.join(ROOT, "third_party", "wod_e2e.desc")
#: Script that regenerates the descriptor set.
GENERATOR = "data_preparation/make_wod_desc.sh"

_FRAME_MESSAGE = "waymo.open_dataset.E2EDFrame"


def descriptor_path() -> str:
    """Return the descriptor-set path, honouring ``$WOD_DESC``."""
    return os.environ.get(DESC_ENV) or DEFAULT_DESC


def _from_package():
    """Build ``E2EDFrame`` from the installed ``waymo_open_dataset`` package."""
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e

    return e2e.E2EDFrame


def _from_descriptor_set(path: str):
    """Build ``E2EDFrame`` from a serialized ``FileDescriptorSet``."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    with open(path, "rb") as fh:
        file_set = descriptor_pb2.FileDescriptorSet()
        file_set.ParseFromString(fh.read())

    pool = descriptor_pool.DescriptorPool()
    for file_proto in file_set.file:  # protoc --include_imports emits topological order
        pool.Add(file_proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(_FRAME_MESSAGE))


def _resolve():
    """Return ``(E2EDFrame, source)`` or raise an actionable error."""
    try:
        return _from_package(), "waymo_open_dataset"
    except Exception as pkg_err:  # absent, half-installed, or gencode/runtime mismatch
        package_error = pkg_err

    path = descriptor_path()
    try:
        return _from_descriptor_set(path), path
    except Exception as desc_err:  # missing file, stale/corrupt set, no protobuf
        raise RuntimeError(
            "cannot load the Waymo E2E protos.\n"
            f"  - the 'waymo_open_dataset' package is unusable ({package_error})\n"
            f"  - the descriptor set at {path} could not be used ({desc_err})\n"
            f"Generate the descriptor set with 'bash {GENERATOR}' (it needs curl,\n"
            f"network access and either pip or uv), or point ${DESC_ENV} at an\n"
            f"existing copy."
        ) from desc_err


E2EDFrame, SOURCE = _resolve()

_FRAME_FIELD = E2EDFrame.DESCRIPTOR.fields_by_name["frame"].message_type
_IMAGE_FIELD = _FRAME_FIELD.fields_by_name["images"].message_type
_CAMERA_ENUM = _IMAGE_FIELD.fields_by_name["name"].enum_type
_INTENT_ENUM = E2EDFrame.DESCRIPTOR.fields_by_name["intent"].enum_type

#: Camera enum number -> name, e.g. ``1 -> 'FRONT'``.
CAMERA_NAME = {v.number: v.name for v in _CAMERA_ENUM.values}
#: Ego-intent enum number -> name, e.g. ``1 -> 'GO_STRAIGHT'``.
INTENT_NAME = {v.number: v.name for v in _INTENT_ENUM.values}


def camera_name(number: int) -> str:
    """Return the camera name for an enum ``number`` (never raises)."""
    return CAMERA_NAME.get(number, f"UNKNOWN_{number}")


def intent_name(number: int) -> str:
    """Return the ego-intent name for an enum ``number`` (never raises)."""
    return INTENT_NAME.get(number, f"UNKNOWN_{number}")
