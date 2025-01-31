from __future__ import annotations
from typing import Literal, overload
from himena import WidgetDataModel
import impy as ip
from himena.consts import StandardType
from himena.standards.model_meta import (
    ImageMeta,
    ImageChannel,
    ArrayAxis,
    LinearCoordinates,
)


def image_to_model(
    img: ip.ImgArray | ip.LazyImgArray,
    title: str | None = None,
    is_rgb: bool = False,
    orig: WidgetDataModel | None = None,
    is_previewing: bool = False,
) -> WidgetDataModel:
    """Convert impy image array to WidgetDataModel."""
    if "c" in img.axes and not is_rgb:
        channel_axis = img.axes.index("c")
        if channel_labels := img.axes[channel_axis].labels:
            channel_labels = [str(name) for name in channel_labels]
        else:
            channel_labels = [f"Ch-{i}" for i in range(img.shape[channel_axis])]
    else:
        channel_axis = None
        channel_labels = [""]
    nchannels = img.shape[channel_axis] if channel_axis is not None else 1
    meta = ImageMeta(
        axes=[
            ArrayAxis(name=str(a)).with_linear_coords(scale=a.scale, unit=a.unit)
            if str(a) != "c"
            else ArrayAxis(name="c").with_categorical_coords(channel_labels)
            for a in img.axes
        ],
        channel_axis=channel_axis,
        channels=[
            ImageChannel(name=channel_labels[i], contrast_limits=None)
            for i in range(nchannels)
        ],
        is_rgb=is_rgb,
    )
    if orig:
        out = orig.with_value(img.value, title=title, metadata=meta)
        if title is None:
            out = out.with_title_numbering()
    else:
        out = WidgetDataModel(
            value=img.value,
            type=StandardType.IMAGE,
            title=title,
            metadata=meta,
        )
    if is_previewing:
        meta.current_indices = None
        meta.contrast_limits = None
    return out


def label_to_model(
    img: ip.ImgArray | ip.LazyImgArray,
    title: str | None = None,
    is_rgb: bool = False,
    orig: WidgetDataModel | None = None,
    is_previewing: bool = False,
) -> WidgetDataModel:
    out = image_to_model(img, title, is_rgb, orig, is_previewing)
    out.type = StandardType.IMAGE_LABELS
    return out


def make_dims_annotation(model: WidgetDataModel) -> list[tuple[str, int]]:
    if not isinstance(meta := model.metadata, ImageMeta):
        return [("2 (xy)", 2)]
    elif (axes := meta.axes) is None:
        return [("2 (xy)", 2)]
    axis_names = [a.name for a in axes]
    if "z" in axis_names and "y" in axis_names and "x" in axis_names:
        choices = [("2 (xy)", 2), ("3 (xyz)", 3)]
    elif "y" in axis_names and "x" in axis_names:
        choices = [("2 (xy)", 2)]
    elif len(axis_names) > 1:
        if len(axis_names[-1]) == len(axis_names[-2]) == 1:
            label = "".join(axis_names[-2:])
        else:
            label = ", ".join(axis_names[-2:])
        choices = [(f"2 ({label})", 2)]
    else:
        choices = [("2 (xy)", 2)]
    return choices


@overload
def model_to_image(
    model: WidgetDataModel,
    is_previewing: Literal[False] = False,
) -> ip.ImgArray: ...


@overload
def model_to_image(
    model: WidgetDataModel,
    is_previewing: bool = False,
) -> ip.ImgArray | ip.LazyImgArray: ...


def model_to_image(
    model: WidgetDataModel,
    is_previewing: bool = False,
):
    # TODO: consider RGB images
    import dask.array as da

    img = model.value
    if not isinstance(meta := model.metadata, ImageMeta):
        raise ValueError("Model must have ImageMeta.")
    if meta.axes is not None:
        scale = {}
        unit = {}
        axes = {}
        for a in meta.axes:
            if isinstance(a.coords, LinearCoordinates):
                scale[a.name] = a.coords.scale or 1
                unit[a.name] = a.coords.unit
            else:
                scale[a.name] = 1
        axes = list(scale.keys())
    else:
        axes = None
        scale = None
        unit = None
    if isinstance(img, da.Array):
        out = ip.lazy.asarray(img, axes=axes, chunks=img.chunksize)
    elif is_previewing:
        n_multi = img.ndim - 3 if meta.is_rgb else img.ndim - 2
        if n_multi == 0:
            out = ip.asarray(img, axes=axes)
        else:
            chunks_rest = (-1, -1, -1) if meta.is_rgb else (-1, -1)
            out = ip.lazy.asarray(img, axes=axes, chunks=(1,) * n_multi + chunks_rest)
    else:
        out = ip.asarray(img, axes=axes)
    if scale is not None and unit is not None:
        for a in out.axes:
            a.scale = scale.get(str(a))
            a.unit = unit.get(str(a))
    return out
