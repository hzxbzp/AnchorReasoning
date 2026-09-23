"""Backbone adapter base class and point codecs.

An adapter isolates everything that differs between the 8 backbones so that
dataset / target builder / trainer / eval can stay backbone-agnostic:

* image processing (patch size, merge size, pixel bounds -> grid -> #image tokens),
* the point coordinate convention (``PointCodec``: absolute processed-image pixels
  for the Qwen2.5-VL family; coordinates normalized to 0-1000 -- ``NORM1000_DESC`` -- for the
  Qwen3-VL family, the codec clipping the encoded value to [0, 999]),
* model construction (``load_model``: base weights or a fine-tuned checkpoint directory),
* tokens inherited from a released backbone that must stay in the vocabulary so its
  weights load unchanged but must never be generated (``extra_vocab_hook`` /
  ``bad_token_ids``),
* the parameter prefix of the vision tower (lr grouping / freezing),
* a golden check that the hand-rendered prompt (``core.prompts.render_prompt``)
  is token-identical to ``processor.apply_chat_template``.

Only ``PointCodec`` and ``BackboneAdapter`` are interface classes; the helper
functions in this file are shared by the concrete adapters.
"""
from __future__ import annotations

import glob
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

from training.core.paths import PANO_H, PANO_W, WEIGHTS

# Coordinate description sentences substituted for {POINT_COORD_DESC} in SYSTEM_S1/S2.
ABS_PIXEL_DESC = "pixel coords of the panorama (processed image)"
NORM1000_DESC = "(x,y) normalized to 0-1000"
NORM_MAX = 1000          # Qwen3-VL grounding range: coordinates in [0, 1000)

Scale = Tuple[float, float]
ImageLike = Union[str, "os.PathLike[str]", Image.Image]


# --------------------------------------------------------------------------------------
# point codecs
# --------------------------------------------------------------------------------------
class PointCodec:
    """Maps panorama pixel coordinates <-> the coordinates written in the target text.

    ``scale = (rw/ow, rh/oh)`` is the processed/original size ratio returned by
    ``BackboneAdapter.image_inputs``; ``orig_size`` (optional) is the original image
    size and defaults to the panorama size ``(PANO_W, PANO_H)``.
    """

    mode: str = ""
    desc: str = ""

    def encode(self, px: float, py: float, scale: Scale,
               orig_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Panorama pixel (px, py) -> integer target coordinates."""
        raise NotImplementedError

    def decode(self, x: float, y: float, scale: Scale,
               orig_size: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        """Target coordinates (x, y) -> panorama pixel coordinates (float)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mode={self.mode!r})"


class AbsPixelCodec(PointCodec):
    """Qwen2.5-VL convention: absolute pixel coordinates of the *processed* image
    (what the vision tower actually sees), i.e. panorama pixel × scale."""

    mode = "abs_pixel"
    desc = ABS_PIXEL_DESC

    def encode(self, px: float, py: float, scale: Scale,
               orig_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        sx, sy = scale
        return int(round(float(px) * sx)), int(round(float(py) * sy))

    def decode(self, x: float, y: float, scale: Scale,
               orig_size: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        sx, sy = scale
        return float(x) / sx, float(y) / sy


class Norm1000Codec(PointCodec):
    """Qwen3-VL convention: coordinates normalized to [0, 1000) of the image width/height.

    Because ``scale = (rw/ow, rh/oh)`` by definition, normalizing w.r.t. the processed
    image (``px*sx/rw``) equals normalizing w.r.t. the original (``px/ow``); only the
    original size is needed, which defaults to the panorama size.
    """

    mode = "norm1000"
    desc = NORM1000_DESC

    def __init__(self, orig_size: Tuple[int, int] = (PANO_W, PANO_H)):
        self.orig_size = (int(orig_size[0]), int(orig_size[1]))

    def encode(self, px: float, py: float, scale: Scale,
               orig_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        ow, oh = orig_size or self.orig_size
        x = int(round(float(px) / ow * NORM_MAX))
        y = int(round(float(py) / oh * NORM_MAX))
        return min(max(x, 0), NORM_MAX - 1), min(max(y, 0), NORM_MAX - 1)

    def decode(self, x: float, y: float, scale: Scale,
               orig_size: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        ow, oh = orig_size or self.orig_size
        return float(x) / NORM_MAX * ow, float(y) / NORM_MAX * oh


def make_point_codec(mode: str) -> PointCodec:
    """Factory: ``'abs_pixel'`` -> AbsPixelCodec, ``'norm1000'`` -> Norm1000Codec."""
    if mode == "abs_pixel":
        return AbsPixelCodec()
    if mode == "norm1000":
        return Norm1000Codec()
    raise ValueError(f"unknown point codec mode {mode!r}")


# --------------------------------------------------------------------------------------
# path helpers
# --------------------------------------------------------------------------------------
def unique_snapshot(repo_dir: str) -> str:
    """Return the unique ``<repo_dir>/snapshots/<hash>`` directory of an HF-hub style cache
    entry; raise if there is none or more than one."""
    snaps = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*")))
    snaps = [s for s in snaps if os.path.isdir(s)]
    if len(snaps) != 1:
        raise FileNotFoundError(
            f"expected exactly one snapshot under {repo_dir}/snapshots, found {len(snaps)}: {snaps}")
    return snaps[0]


def resolve_weights_dir(rel: str, snapshot: bool = False) -> str:
    """``WEIGHTS/<rel>`` (plain model dir) or its unique snapshot (``snapshot=True``).
    Raises FileNotFoundError with a clear message if the directory does not exist."""
    path = os.path.join(WEIGHTS, rel)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"backbone weights not found: {path}")
    return unique_snapshot(path) if snapshot else path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_image(img: ImageLike) -> Image.Image:
    """Open a path as RGB PIL image; pass PIL images through (converted to RGB)."""
    if isinstance(img, Image.Image):
        return img.convert("RGB") if img.mode != "RGB" else img
    return Image.open(os.fspath(img)).convert("RGB")


# --------------------------------------------------------------------------------------
# adapter base
# --------------------------------------------------------------------------------------
class BackboneAdapter:
    """Base class for the 8 registry entries; concrete subclasses set the class attributes
    and override the model-loading / vocabulary hooks where the backbone deviates.

    Attributes (required):
        name, family ('qwen25'|'qwen3vl'), patch, merge, point_codec, base_path.
    Extra attributes:
        processor_path  -- where the processor/tokenizer is loaded from (== base_path unless
                           the base is not an HF directory, e.g. AutoVLA's Lightning ckpt);
        model_cls       -- transformers ``*ForConditionalGeneration`` class of the family;
        long_edge       -- default long edge used by ``load_processor`` (2916 = no scaling).
    """

    name: str = ""
    family: str = ""
    patch: int = 0
    merge: int = 2
    point_mode: str = ""
    model_cls: Any = None
    long_edge: int = PANO_W

    def __init__(self) -> None:
        self.point_codec: PointCodec = make_point_codec(self.point_mode)
        self.base_path: str = self._staged_or(self._resolve_base_path())
        self.processor_path: str = self._resolve_processor_path()
        self.last_template_diff: Optional[Dict[str, Any]] = None

    # ---- paths -------------------------------------------------------------------------
    @staticmethod
    def _staged_or(path: str) -> str:
        """Redirect the base weights to a node-local copy when the launcher staged one.

        safetensors are memory-mapped, and random-access page faults over a shared network
        filesystem can be orders of magnitude slower than a sequential read. The launcher
        scripts therefore copy the weight directory to node-local storage and export
        ``ANCHOR_BASE_PATH``, so the mmap hits a local disk instead.
        """
        staged = os.environ.get("ANCHOR_BASE_PATH", "").strip()
        if staged and os.path.isdir(staged):
            return staged
        return path

    def _resolve_base_path(self) -> str:
        raise NotImplementedError

    def _resolve_processor_path(self) -> str:
        return self.base_path

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name!r}, family={self.family!r}, patch={self.patch}, "
                f"merge={self.merge}, point={self.point_codec.mode!r}, base={self.base_path!r})")

    # ---- image processing ----------------------------------------------------------------
    def pixel_bounds(self, long_edge: Optional[int] = None,
                     orig_size: Tuple[int, int] = (PANO_W, PANO_H)) -> Tuple[int, int]:
        """(min_pixels, max_pixels): ``max = long_edge² · short/long`` so the long edge lands
        at ``long_edge`` (aspect ratio kept), ``min = 256 · patch²``."""
        le = int(long_edge or self.long_edge)
        ow, oh = orig_size
        ratio = min(ow, oh) / max(ow, oh)
        return 256 * self.patch * self.patch, int(le * le * ratio)

    def configure_image_processor(self, imgproc: Any, long_edge: Optional[int] = None) -> None:
        """Set the processor's resolution bounds (slow processors read ``min_pixels`` /
        ``max_pixels``; fast processors read ``size['shortest_edge'|'longest_edge']``) and
        sanity-check patch / merge against this adapter."""
        mn, mx = self.pixel_bounds(long_edge)
        for attr, val in (("min_pixels", mn), ("max_pixels", mx)):
            if hasattr(imgproc, attr):
                setattr(imgproc, attr, val)
        size = getattr(imgproc, "size", None)
        if isinstance(size, dict):
            size["shortest_edge"] = mn
            size["longest_edge"] = mx
            size.pop("min_pixels", None)
            size.pop("max_pixels", None)
        ps = getattr(imgproc, "patch_size", None)
        ms = getattr(imgproc, "merge_size", None)
        if ps is not None and int(ps) != self.patch:
            raise ValueError(f"{self.name}: processor patch_size={ps} != adapter patch={self.patch}")
        if ms is not None and int(ms) != self.merge:
            raise ValueError(f"{self.name}: processor merge_size={ms} != adapter merge={self.merge}")

    def load_processor(self, path: Optional[str] = None) -> Any:
        """AutoProcessor (image processor + tokenizer) of the backbone, configured for the
        panorama resolution, with the backbone's extra vocabulary registered (idempotent).
        ``path`` (optional) loads from a checkpoint directory instead of the base."""
        from transformers import AutoProcessor
        src = path or self.processor_path
        proc = AutoProcessor.from_pretrained(src)
        self.configure_image_processor(proc.image_processor, self.long_edge)
        self._extend_tokenizer(proc.tokenizer)
        return proc

    def image_inputs(self, img_path: ImageLike, long_edge: Optional[int] = None,
                     processor: Any = None) -> Dict[str, Any]:
        """Process one panorama.

        Returns ``dict(pixel_values, image_grid_thw, n_img_tokens, scale, orig_size, proc_size)``
        where ``scale = (rw/ow, rh/oh)`` maps original pixels to processed pixels and
        ``n_img_tokens = prod(grid) // merge²`` is the number of ``<|image_pad|>`` tokens.
        ``processor`` (optional) reuses an already-loaded processor.
        """
        imgproc = (processor or self.processor).image_processor
        img = open_image(img_path)
        ow, oh = img.size
        mn, mx = self.pixel_bounds(long_edge, (ow, oh))
        out = imgproc(images=[img], min_pixels=mn, max_pixels=mx, return_tensors="pt")
        grid = out["image_grid_thw"][0]
        rh, rw = int(grid[1]) * self.patch, int(grid[2]) * self.patch
        n_img = int(grid.prod()) // (self.merge ** 2)
        return {
            "pixel_values": out["pixel_values"],
            "image_grid_thw": out["image_grid_thw"],
            "n_img_tokens": n_img,
            "scale": (rw / ow, rh / oh),
            "orig_size": (ow, oh),
            "proc_size": (rw, rh),
        }

    @property
    def processor(self) -> Any:
        """Lazily-loaded default processor (cached on the adapter)."""
        proc = getattr(self, "_processor", None)
        if proc is None:
            proc = self.load_processor()
            self._processor = proc
        return proc

    # ---- model ---------------------------------------------------------------------------
    @staticmethod
    def resolve_attn(attn: Optional[str]) -> Optional[str]:
        """flash_attention_2 needs CUDA; fall back to sdpa on CPU-only hosts (with a warning)
        so smoke tests can run. Any other value is passed through."""
        if attn == "flash_attention_2" and not torch.cuda.is_available():
            warnings.warn("flash_attention_2 requested without CUDA -> using 'sdpa'")
            return "sdpa"
        return attn

    def _from_pretrained_kwargs(self, dtype: Any, attn: Optional[str],
                                device_map: Any) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"dtype": dtype}
        attn = self.resolve_attn(attn)
        if attn:
            kw["attn_implementation"] = attn
        if device_map is not None:
            kw["device_map"] = device_map
        return kw

    def build_config(self, path: Optional[str] = None) -> Any:
        """The HF config ``load_model(path)`` would instantiate (config-only; no weights).
        Used for meta-device / CPU checks."""
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(path or self.base_path)

    def load_model(self, path: Optional[str] = None, dtype: Any = torch.bfloat16,
                   attn: Optional[str] = "flash_attention_2", device_map: Any = None) -> torch.nn.Module:
        """``path=None`` -> base weights; ``path=<dir>`` -> a fine-tuned checkpoint directory
        written by ``save_pretrained``."""
        src = path or self.base_path
        return self.model_cls.from_pretrained(src, **self._from_pretrained_kwargs(dtype, attn, device_map))

    # ---- vocabulary ------------------------------------------------------------------------
    def _extend_tokenizer(self, tokenizer: Any) -> int:
        """Register the backbone's extra tokens (idempotent); returns #tokens newly added.
        Base: nothing to add."""
        return 0

    def legacy_tokens(self) -> List[str]:
        """Tokens inherited from the released backbone that must never be generated
        (empty for plain Qwen)."""
        return []

    def extra_vocab_hook(self, tokenizer: Any, model: Optional[torch.nn.Module]) -> None:
        """Make tokenizer and model vocabularies consistent (register extra tokens, resize
        embeddings when the model has fewer rows than the tokenizer). Base: only the
        consistency check."""
        self._extend_tokenizer(tokenizer)
        self._sync_embeddings(tokenizer, model)

    def _sync_embeddings(self, tokenizer: Any, model: Optional[torch.nn.Module]) -> None:
        if model is None:
            return
        rows = int(model.get_input_embeddings().num_embeddings)
        n = len(tokenizer)
        if rows < n:
            warnings.warn(f"{self.name}: embedding rows {rows} < tokenizer size {n}; resizing")
            model.resize_token_embeddings(n, mean_resizing=False)

    def bad_token_ids(self, tokenizer: Any) -> List[int]:
        """Ids of ``legacy_tokens()`` that exist in this tokenizer (sorted, unique); they are
        suppressed at generation time."""
        toks = self.legacy_tokens()
        if not toks:
            return []
        vocab = tokenizer.get_vocab()
        ids = {int(vocab[t]) for t in toks if t in vocab}
        return sorted(ids)

    # ---- vision tower ------------------------------------------------------------------------
    def visual_param_prefix(self) -> str:
        """Parameter-name prefix of the vision tower (transformers 4.57 layout for both
        Qwen2.5-VL and Qwen3-VL: ``model.visual``)."""
        return "model.visual"

    def freeze_vision(self, model: torch.nn.Module) -> None:
        """Set ``requires_grad=False`` on every vision-tower parameter (incl. the merger)."""
        prefix = self.visual_param_prefix() + "."
        n = 0
        for name, p in model.named_parameters():
            if name.startswith(prefix):
                p.requires_grad = False
                n += 1
        if n == 0:                          # older layout fallback (visual.* at top level)
            vis = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
            if vis is None:
                raise ValueError(f"{self.name}: no vision parameters found with prefix {prefix!r}")
            for p in vis.parameters():
                p.requires_grad = False
                n += 1

    # ---- prompt golden check -----------------------------------------------------------------
    @staticmethod
    def _frame_and_image(frame_like: Any) -> Tuple[Dict[str, Any], Image.Image]:
        """Accept a frame dir path, a dict with ``fdir``, or a frame.json dict (optionally with
        ``_img_path``); returns (frame, image)."""
        fdir = None
        frame: Optional[Dict[str, Any]] = None
        img_path = None
        if isinstance(frame_like, (str, os.PathLike)):
            fdir = os.fspath(frame_like)
        elif isinstance(frame_like, dict):
            fdir = frame_like.get("fdir") or frame_like.get("_fdir")
            img_path = frame_like.get("_img_path") or frame_like.get("img_path")
            if "past_states" in frame_like:
                frame = frame_like
        else:
            raise TypeError(f"frame_like must be a path or dict, got {type(frame_like)}")
        if fdir:
            if frame is None:
                frame = read_json(os.path.join(fdir, "frame.json"))
            cand = os.path.join(fdir, "panorama_geo.png")
            if img_path is None and os.path.isfile(cand):
                img_path = cand
        if frame is None:
            raise ValueError("frame_like carries no frame.json content")
        if not img_path:
            raise ValueError("frame_like carries no panorama image path")
        return frame, open_image(img_path)

    def chat_template_check(self, processor: Any, frame_like: Any,
                            tasks: Sequence[str] = ("s1", "s2", "attrqa"),
                            long_edge: Optional[int] = None) -> bool:
        """True iff ``core.prompts.render_prompt`` is token-identical to
        ``processor.apply_chat_template`` + ``processor(text, images)`` for every task.
        On mismatch ``self.last_template_diff`` holds the first differing task/position."""
        from training.core import prompts    # other module; imported lazily on purpose
        frame, img = self._frame_and_image(frame_like)
        self.configure_image_processor(processor.image_processor, long_edge or self.long_edge)
        tok = processor.tokenizer
        desc = self.point_codec.desc
        attrqa = {"type": "Car", "x": 100, "y": 200}
        self.last_template_diff = None
        for task in tasks:
            extra = attrqa if task == "attrqa" else None
            messages = [
                {"role": "system", "content": prompts.system_for(task, desc)},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompts.user_text(frame, task, attrqa=extra)},
                ]},
            ]
            ref_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ref = processor(text=[ref_text], images=[img], return_tensors="pt")
            ref_ids = ref["input_ids"][0].tolist()
            n_img = int(ref["image_grid_thw"][0].prod()) // (self.merge ** 2)
            mine = tok(prompts.render_prompt(task, frame, n_img, desc, attrqa=extra),
                       add_special_tokens=False).input_ids
            if mine != ref_ids:
                pos = next((i for i, (a, b) in enumerate(zip(mine, ref_ids)) if a != b),
                           min(len(mine), len(ref_ids)))
                self.last_template_diff = {
                    "task": task, "pos": pos, "len_mine": len(mine), "len_ref": len(ref_ids),
                    "mine": tok.decode(mine[max(0, pos - 8):pos + 8]),
                    "ref": tok.decode(ref_ids[max(0, pos - 8):pos + 8]),
                }
                return False
        return True
