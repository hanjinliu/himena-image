from __future__ import annotations
from himena import WidgetDataModel
import impy as ip
from himena.consts import StandardType
from himena.standards.model_meta import ImageMeta


def image_to_model(
    img: ip.ImgArray,
    title: str | None = None,
    is_rgb: bool = False,
    orig: WidgetDataModel | None = None,
    is_previewing: bool = False,
) -> WidgetDataModel:
    if "c" in img.axes and not is_rgb:
        channel_axis = img.axes.index("c")
    else:
        channel_axis = None
    meta = ImageMeta(
        axes=[str(a) for a in img.axes],
        scale=list(img.scale.values()),
        channel_axis=channel_axis,
        is_rgb=is_rgb,
    )
    if is_previewing:
        meta.current_indices = None
    if orig:
        out = orig.with_value(img.value, title=title, metadata=meta)
    else:
        out = WidgetDataModel(
            value=img.value,
            type=StandardType.IMAGE,
            title=title,
            metadata=meta,
        )
    return out


def make_dims_annotation(model: WidgetDataModel[ip.ImgArray]) -> list[tuple[str, int]]:
    img = model.value
    if not isinstance(img, ip.ImgArray):
        choices = [("2 (xy)", 2)]
    elif len(img.spatial_shape) == 2:
        choices = [("2 (xy)", 2)]
    else:
        choices = [("2 (xy)", 2), ("3 (xyz)", 3)]
    return choices


def model_to_image(
    model: WidgetDataModel,
    is_previewing: bool = False,
) -> ip.ImgArray | ip.LazyImgArray:
    import dask.array as da

    img = model.value
    if not isinstance(meta := model.metadata, ImageMeta):
        raise ValueError("Model must have ImageMeta.")
    axes = meta.axes
    if isinstance(img, da.Array) or is_previewing:
        out = ip.lazy.asarray(img, axes=axes)
    else:
        out = ip.asarray(model.value, axes=axes)
    if meta.scale is not None:
        scales = {a: s for a, s in zip(axes, meta.scale)}
        if isinstance(meta.unit, str):
            unit = meta.unit
        else:
            unit = None
        out.set_scale(scales, unit=unit)
    return out
