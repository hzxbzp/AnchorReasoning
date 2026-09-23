#!/usr/bin/env bash
# Regenerate the Waymo Open Dataset E2E descriptor set used by wod_proto.py.
#
# Downloads the six .proto files the E2E frame message depends on, compiles them
# into a single self-contained FileDescriptorSet and writes it to
# third_party/wod_e2e.desc (override with the first argument or $WOD_DESC).
#
# protoc is taken from grpcio-tools, installed into a throwaway prefix so the
# working environment is never modified. Downloads and that prefix live in a
# wod_proto_build/ directory next to the output file, so pointing the output
# somewhere else keeps the whole build out of the tree. With the default output
# that is third_party/wod_proto_build, which is gitignored; re-running is cheap
# and deleting the directory is always safe.
#
# The interpreter is python3, or python, or whatever $PYTHON names; it is used
# only to run pip and protoc, never imported from.
#
# Usage: bash data_preparation/make_wod_desc.sh [output.desc]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="${1:-${WOD_DESC:-$ROOT/third_party/wod_e2e.desc}}"

mkdir -p "$(dirname "$OUT")"
OUT_DIR="$(cd "$(dirname "$OUT")" && pwd)"
OUT="$OUT_DIR/$(basename "$OUT")"
BUILD="$OUT_DIR/wod_proto_build"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "error: no python interpreter on PATH (set \$PYTHON to choose one)" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required to fetch the .proto sources" >&2
  exit 1
fi

BASE="https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset"
PROTOS=(
  dataset.proto
  label.proto
  protos/map.proto
  protos/vector.proto
  protos/keypoint.proto
  protos/end_to_end_driving_data.proto
)

SRC="$BUILD/protosrc/waymo_open_dataset"
mkdir -p "$SRC/protos"
for p in "${PROTOS[@]}"; do
  echo "fetching $p"
  curl -sSfL -o "$SRC/$p" "$BASE/$p"
done

# protoc from grpcio-tools; installed once, reused on later runs.
TOOLS="$BUILD/tools"
if [ ! -d "$TOOLS/grpc_tools" ]; then
  echo "installing grpcio-tools into $TOOLS"
  if "$PY" -m pip --version >/dev/null 2>&1; then
    "$PY" -m pip install --quiet --upgrade --target "$TOOLS" grpcio-tools
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$PY" --target "$TOOLS" grpcio-tools
  else
    echo "error: need pip or uv to install grpcio-tools (it provides protoc)" >&2
    exit 1
  fi
fi

PYTHONPATH="$TOOLS" "$PY" -m grpc_tools.protoc \
  -I "$BUILD/protosrc" \
  --include_imports \
  --descriptor_set_out="$OUT" \
  "$SRC/protos/end_to_end_driving_data.proto"

echo "wrote $OUT"
echo "build files kept in $BUILD (safe to delete)"
