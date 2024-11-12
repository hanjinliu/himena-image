from __future__ import annotations

from pathlib import Path
from himena import WidgetDataModel
from himena.plugins import register_reader_provider, register_writer_provider
from himena_image.utils import image_to_model
import impy as ip


_SUPPORTED_EXT = frozenset(
    [".tif", ".tiff",
     ".mrc", ".rec", ".st", ".map", ".mrc.gz", ".map.gz",
     ".nd2",
     ".npy", ".npz",
     ".png", ".jpg", ".jpeg",
     ]
)  # fmt: skip


def read_image(path: Path):
    img = ip.imread(path)
    is_rgb = "c" in img.axes and path.suffix in [".png", ".jpg", ".jpeg"]
    return image_to_model(img, is_rgb=is_rgb)


def write_image(model: WidgetDataModel, path: Path):
    img = model.value
    if isinstance(img, ip.ImgArray):
        img.imsave(path)
    else:
        img = ip.asarray(img)
        img.imsave(path)


@register_reader_provider
def read_image_provider(path: Path):
    ext = "".join(path.suffixes)
    if ext in _SUPPORTED_EXT:
        return read_image
    return None


@register_writer_provider
def write_image_provider(path: Path):
    ext = "".join(path.suffixes)
    if ext in _SUPPORTED_EXT:
        return write_image
    return None
