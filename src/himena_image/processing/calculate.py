from typing import Literal
from cmap import Colormap

import numpy as np
import impy as ip
from superqt.utils import qthrottled, ensure_main_thread
from himena import WidgetDataModel, Parametric, StandardType, create_model
from himena.plugins import register_function, configure_gui
from himena.standards.model_meta import ImageMeta, DataFramePlotMeta
from himena.standards import roi
from himena.widgets import SubWindow
from himena_image.utils import image_to_model, model_to_image
from himena_builtins.qt.image import QImageView, QtRois
from himena_builtins.qt.dataframe import QDataFramePlotView

MENU = ["tools/image/calculate", "/model_menu/calculate"]


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
        method: Literal["mean", "median", "max", "min", "sum", "std"],
        # range: tuple[int, int],
    ) -> WidgetDataModel:
        img = model_to_image(model)
        out = img.proj(axis=axis, method=method)
        return image_to_model(
            out, title=model.title, extension_default=model.extension_default
        )

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
    title="Profile Line",
    menus=MENU,
    types=[StandardType.IMAGE],
    command_id="himena-image:profile:profile-line",
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
        out = _run_profile_line(model_to_image(model), meta, coords, indices)
        out.title = f"Profile of {model.title}"
        return out

    return run_profile_line


@register_function(
    title="Profile Line (Live)",
    menus=MENU,
    types=[StandardType.IMAGE],
    command_id="himena-image:profile:profile-line-live",
)
def profile_line_live(win: SubWindow[QImageView]):
    """Live-plot the line profile of the current image slice."""
    plot_view = QDataFramePlotView()
    update_model_throttled = ensure_main_thread(plot_view.update_model)

    @qthrottled(timeout=50)
    def _callback():
        qroi = win.widget._img_view._current_roi_item
        if isinstance(qroi, (QtRois.QLineRoi, QtRois.QSegmentedLineRoi)):
            model_input = win.widget.to_model()
            coords = _get_profile_coords(model_input.metadata)
            if len(coords) < 2:
                model = _empty_dataframe_model()
            else:
                model = _run_profile_line(
                    model_to_image(model_input),
                    model_input.metadata,
                    coords,
                    _get_indices_channel_composite(model_input.metadata),
                )
        else:
            model = _empty_dataframe_model()
        update_model_throttled(model)

    win.widget.current_roi_updated.connect(_callback)
    win.widget.dims_slider.valueChanged.connect(_callback)
    child = win.add_child(plot_view, title="Profile Line (Live)")
    child.closed.connect(lambda: win.widget.current_roi_updated.disconnect(_callback))
    _callback()


def _run_profile_line(
    img: ip.ImgArray,
    meta: ImageMeta,
    coords: list[list[float]],
    indices: list[int | None],
    order: int = 3,
) -> WidgetDataModel:
    _indices = tuple(slice(None) if i is None else i for i in indices)
    img_slice = img[_indices]
    if isinstance(img_slice, ip.LazyImgArray):
        img_slice = img_slice.compute()
    order: int = 0 if img.dtype.kind == "b" else order
    if meta.is_rgb:
        img_slice = np.moveaxis(img_slice, -1, 0)
    sliced = img_slice.reslice(coords, order=order)

    if sliced.ndim == 2:  # multi-channel
        sliced_arrays = [sliced[i] for i in range(sliced.shape[0])]
        slice_headers = [
            _channed_name(axis.name, i) for i, axis in enumerate(meta.axes)
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
    return create_model(
        value=df,
        type=StandardType.DATAFRAME_PLOT,
        metadata=DataFramePlotMeta(plot_color_cycle=color_cycle),
    )


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
    along_choices = [axis.name for axis in meta.axes]
    stack_over_choices = along_choices.copy()
    stack_over_default = []
    if meta.channel_axis is not None:
        along_choices.pop(meta.channel_axis)
        stack_over_default.append(stack_over_choices[meta.channel_axis])
    along_choices = along_choices[:-2]  # remove xy
    stack_over_choices = stack_over_choices[:-2]  # remove xy

    @configure_gui(
        coords={"bind": _get_profile_coords(meta)},
        along={"choices": along_choices},
        stack_over={
            "choices": stack_over_choices,
            "widget_type": "Select",
            "value": stack_over_default,
        },
        same_dtype={"label": "Keep same data type"},
    )
    def run_kymograph(
        coords,
        along: str,
        stack_over: list[str],
        same_dtype: bool = True,
    ) -> WidgetDataModel:
        if along in stack_over:
            raise ValueError("Duplicated axis name in `along` and `stack_over`.")
        img = model_to_image(model)
        # NOTE: ImgArray supports __getitem__ with dict
        sl: dict[str, int] = {}
        for i, axis in enumerate(img.axes):
            axis = str(axis)
            if axis == along or axis in stack_over:
                continue
            if not hasattr(meta.current_indices[i], "__index__"):
                continue
            sl[axis] = meta.current_indices[i]
        if sl:
            img_slice = img[sl]
        else:
            img_slice = img
        order = 0 if img.dtype.kind == "b" else 3
        if same_dtype:
            dtype = img.dtype
        else:
            dtype = None
        sliced = ip.asarray(img_slice.reslice(coords, order=order), dtype=dtype)
        sliced = np.swapaxes(sliced, along, -2)
        return image_to_model(sliced, title=f"Kymograph of {model.title}")

    return run_kymograph


def _channed_name(ch: str | None, i: int) -> str:
    if ch is None:
        return f"Ch-{i}"
    return ch


def _get_profile_coords(meta: ImageMeta) -> list[list[float]]:
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


def _empty_dataframe_model():
    return create_model(
        {"x": [], "y": []},
        type=StandardType.DATAFRAME_PLOT,
    )
