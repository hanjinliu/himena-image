from typing import Literal
from cmap import Colormap

import numpy as np
import impy as ip

from himena import WidgetDataModel, Parametric
from himena.plugins import register_function, configure_gui
from himena.consts import StandardType
from himena.standards.model_meta import ImageMeta, DataFramePlotMeta
from himena.standards import roi
from himena_image.utils import image_to_model, model_to_image

MENU = "image/calculate"


@register_function(
    title="Projection ...",
    menus=MENU,
    types=[StandardType.IMAGE],
    run_async=True,
    command_id="himena-image:projection",
)
def projection(model: WidgetDataModel) -> Parametric:
    """Project the image along an axis."""
    img = model_to_image(model)
    axis_choices = [str(a) for a in img.axes]
    if "z" in axis_choices:
        value = "z"
    elif "t" in axis_choices:
        value = "t"
    else:
        value = axis_choices[0]

    @configure_gui(
        axis={"choices": axis_choices, "value": value, "widget_type": "Select"},
    )
    def run_projection(
        axis: str,
        method: Literal["mean", "max", "min", "sum", "std"],
    ) -> WidgetDataModel:
        img = model_to_image(model)
        out = img.proj(axis=axis, method=method)
        return image_to_model(out, title=model.title)

    return run_projection


@register_function(
    title="Invert",
    menus=MENU,
    types=[StandardType.IMAGE],
    run_async=True,
    command_id="himena-image:invert",
)
def invert(model: WidgetDataModel) -> WidgetDataModel:
    """Invert the image."""
    img = -model.value
    out = model.with_value(img)
    if isinstance(model.metadata, ImageMeta):
        assert isinstance(out.metadata, ImageMeta)
        out.metadata.contrast_limits = None
    return out


@register_function(
    title="Profile line",
    menus=MENU,
    types=[StandardType.IMAGE],
    command_id="himena-image:profile:profile_line",
    keybindings=["/"],
    run_async=True,
)
def profile_line(model: WidgetDataModel) -> Parametric:
    """Get the line profile of the current image slice."""
    if not isinstance(meta := model.metadata, ImageMeta):
        raise ValueError("Metadata is missing.")

    @configure_gui(
        coords={"bind": lambda *_: _get_profile_coords(meta)},
        indices={"bind": lambda *_: _get_indices_channel_composite(meta)},
    )
    def run_profile_line(
        coords: list[list[float]],
        indices: list[int | None],
    ) -> WidgetDataModel:
        img = model_to_image(model)
        _indices = tuple(slice(None) if i is None else i for i in indices)
        img_slice = img[_indices]
        if isinstance(img_slice, ip.LazyImgArray):
            img_slice = img_slice.compute()
        order: int = 0 if img.dtype.kind == "b" else 3
        sliced = img_slice.reslice(coords, order=order)

        if sliced.ndim == 2:  # multi-channel
            sliced_arrays = [sliced[i] for i in range(sliced.shape[0])]
            slice_headers = [
                _channed_name(ch.name, i) for i, ch in enumerate(meta.channels)
            ]
        elif sliced.ndim == 1:
            sliced_arrays = [sliced]
            slice_headers = ["intensity"]
        else:
            raise ValueError(f"Invalid shape: {sliced.shape}.")
        scale = sliced.axes[0].scale
        distance = np.arange(sliced_arrays[0].shape[0]) * scale
        df = {"distance": distance}
        for array, header in zip(sliced_arrays, slice_headers):
            df[header] = array
        color_cycle = [Colormap(ch.colormap or "gray")(0.5).hex for ch in meta.channels]
        return WidgetDataModel(
            value=df,
            type=StandardType.DATAFRAME_PLOT,
            title=f"Profile of {model.title}",
            metadata=DataFramePlotMeta(plot_color_cycle=color_cycle),
        )

    return run_profile_line


@register_function(
    title="Kymograph",
    menus=MENU,
    types=[StandardType.IMAGE],
    run_async=True,
    command_id="himena-image:profile:kymograph",
)
def kymograph(model: WidgetDataModel) -> Parametric:
    """Calculate the kymograph along the specified line."""
    if not isinstance(meta := model.metadata, ImageMeta):
        raise ValueError("Metadata is missing.")

    if meta.current_indices is None:
        raise ValueError("`current_indices` is missing in the image metadata")
    if meta.axes is None:
        raise ValueError("`axes` is missing in the image metadata")
    axes_names = [axis.name for axis in meta.axes]
    if meta.channel_axis is not None:
        axes_names.pop(meta.channel_axis)
    axes_names = axes_names[:-2]  # remove xy

    @configure_gui(
        coords={"bind": _get_profile_coords(meta)},
        along={"choices": axes_names},
    )
    def run_kymograph(
        coords,
        along: str,
    ) -> WidgetDataModel:
        img = model_to_image(model)
        # NOTE: ImgArray supports __getitem__ with dict
        sl: dict[str, int] = {}
        for i, axis in enumerate(img.axes):
            if str(axis) == along:
                continue
            if not hasattr(meta.current_indices[i], "__index__"):
                continue
            sl[str(axis)] = meta.current_indices[i]
        if sl:
            img_slice = img[sl]
        else:
            img_slice = img
        order = 0 if img.dtype.kind == "b" else 3
        sliced = ip.asarray(img_slice.reslice(coords, order=order))
        return image_to_model(sliced, title=f"Kymograph of {model.title}")

    return run_kymograph


def _channed_name(ch: str | None, i: int) -> str:
    if ch is None:
        return f"Ch-{i}"
    return ch


def _get_profile_coords(meta: ImageMeta):
    if isinstance(r := meta.current_roi, roi.LineRoi):
        points = [[r.y1, r.x1], [r.y2, r.x2]]
    elif isinstance(r := meta.current_roi, roi.SegmentedLineRoi):
        points = np.stack([r.ys, r.xs], axis=-1).tolist()
    else:
        raise TypeError(
            "`profile_line` requires a line or segmented line ROI, but the current ROI "
            f"item is {r!r}."
        )
    return points


def _get_indices_channel_composite(meta: ImageMeta):
    """Return the current indices with the channel axis set to None."""
    if meta.current_indices is None:
        raise ValueError("Tried to obtain current indices but it is not set.")
    indices = list(meta.current_indices)
    if meta.channel_axis is not None:
        indices[meta.channel_axis] = None
    indices = tuple(indices)
    return indices
