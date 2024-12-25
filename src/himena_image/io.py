from __future__ import annotations

from pathlib import Path
from himena import WidgetDataModel
from himena.standards.model_meta import ImageMeta
from himena.plugins import register_reader_provider, register_writer_provider
from himena_image.utils import image_to_model
import impy as ip


_SUPPORTED_EXT = frozenset(
    [".tif", ".tiff",
     ".mrc", ".rec", ".st", ".map", ".mrc.gz", ".map.gz",
     ".nd2",
     ]
)  # fmt: skip


def read_image(path: Path):
    img = ip.imread(path)
    is_rgb = "c" in img.axes and path.suffix in [".png", ".jpg", ".jpeg"]
    model = image_to_model(img, is_rgb=is_rgb)
    model.extension_default = path.suffix
    return model


def write_image(model: WidgetDataModel, path: Path):
    img = model.value
    _axes = None
    _scales = {}
    _units = {}
    if isinstance(meta := model.metadata, ImageMeta):
        if axes := meta.axes:
            _axes = [a.name for a in axes]
            _scales = {a.name: a.scale for a in axes}
            _units = {a.name: a.unit for a in axes}
    img = ip.asarray(img, axes=_axes)

    for a in img.axes:
        a.scale = _scales.get(str(a))
        a.unit = _units.get(str(a))
    img.imsave(path)


@register_reader_provider
def read_image_provider(path: Path):
    ext = "".join(path.suffixes)
    if ext in _SUPPORTED_EXT:
        return read_image
    return None


@register_writer_provider
def write_image_provider(model: WidgetDataModel, path: Path):
    ext = "".join(path.suffixes)
    if ext in _SUPPORTED_EXT:
        return write_image
    return None
