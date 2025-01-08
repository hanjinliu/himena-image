from typing import Literal
from cmap import Colormap

from himena import WidgetDataModel, Parametric
from himena.plugins import register_function, configure_gui
from himena.consts import StandardType
from himena.standards.model_meta import ImageMeta, DataFramePlotMeta
from himena.standards import roi
import numpy as np
from himena_image.utils import image_to_model, model_to_image

MENU = "image/calculate"


@register_function(
    title="Projection ...",
    menus=MENU,
    types=[StandardType.IMAGE],
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
        run_async=True,
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
    command_id="himena-image:profile_line",
    keybindings=["/"],
)
def profile_line(model: WidgetDataModel) -> WidgetDataModel:
    """Profile line."""
    img = model_to_image(model)
    if not isinstance(meta := model.metadata, ImageMeta):
        raise ValueError("Metadata is missing.")

    indices = meta.current_indices_channel_composite

    if isinstance(r := meta.current_roi, roi.LineRoi):
        points = [[r.y1, r.x1], [r.y2, r.x2]]
    elif isinstance(r := meta.current_roi, roi.SegmentedLineRoi):
        points = np.stack([r.ys, r.xs], axis=-1)
    else:
        raise TypeError(
            "`profile_line` requires a line or segmented line ROI, but the current ROI "
            f"item is {r!r}."
        )
    img_slice = img[indices]
    sliced = img_slice.reslice(points)

    def _channed_name(ch: str | None, i: int) -> str:
        if ch is None:
            return f"Ch {i}"
        return ch

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
    color_cycle = [Colormap(ch.colormap)(0.5) for ch in meta.channels]
    return WidgetDataModel(
        value=df,
        type=StandardType.DATAFRAME_PLOT,
        title=model.title,
        metadata=DataFramePlotMeta(plot_color_cycle=color_cycle),
    )
