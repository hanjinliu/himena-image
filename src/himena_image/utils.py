from __future__ import annotations
from himena import WidgetDataModel
import impy as ip
from himena.consts import StandardType
from himena.model_meta import ImageMeta


def image_to_model(
    img: ip.ImgArray,
    title: str | None = None,
    is_rgb: bool = False,
) -> WidgetDataModel[ip.ImgArray]:
    if "c" in img.axes and not is_rgb:
        channel_axis = img.axes.index("c")
    else:
        channel_axis = None
    return WidgetDataModel(
        value=img,
        type=StandardType.IMAGE,
        title=title,
        metadata=ImageMeta(
            axes=[str(a) for a in img.axes],
            scale=list(img.scale.values()),
            channel_axis=channel_axis,
            is_rgb=is_rgb,
        ),
    )


def make_dims_annotation(model: WidgetDataModel[ip.ImgArray]) -> list[tuple[str, int]]:
    img = model.value
    if not isinstance(img, ip.ImgArray):
        choices = [("2 (xy)", 2)]
    elif len(img.spatial_shape) == 2:
        choices = [("2 (xy)", 2)]
    else:
        choices = [("2 (xy)", 2), ("3 (xyz)", 3)]
    return choices
