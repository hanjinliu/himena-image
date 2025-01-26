from __future__ import annotations

from functools import singledispatch
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from himena import WidgetDataModel, Parametric, StandardType
from himena.plugins import register_function, configure_gui
from himena.standards import roi, model_meta
from himena.data_wrappers import wrap_array


@singledispatch
def slice_array(r: roi.RoiModel, arr_nd: np.ndarray):
    mask = r.to_mask(arr_nd.shape)
    return arr_nd[..., mask]


@slice_array.register
def _(r: roi.RectangleRoi, arr_nd: np.ndarray):
    bb = r.bbox().adjust_to_int("inner")
    arr = arr_nd[..., bb.top : bb.bottom, bb.left : bb.right]
    return arr.reshape(*arr.shape[:-2], arr.shape[-2] * arr.shape[-1])


@slice_array.register
def _(r: roi.EllipseRoi, arr_nd: np.ndarray):
    bb = r.bbox().adjust_to_int("inner")
    arr = arr_nd[..., bb.top : bb.bottom, bb.left : bb.right]
    _yy, _xx = np.indices(arr.shape[-2:])
    mask = (_yy - r.y) ** 2 / r.height**2 + (_xx - r.x) ** 2 / r.width**2 <= 1
    return arr[..., mask]


@slice_array.register
def _(r: roi.PointRoi2D, arr_nd: np.ndarray):
    out = ndi.map_coordinates(arr_nd, [[r.y], [r.x]], order=1, mode="nearest")
    return out


@slice_array.register
def _(r: roi.PointsRoi2D, arr_nd: np.ndarray):
    coords = np.stack([r.ys, r.xs], axis=1)
    out = ndi.map_coordinates(arr_nd, coords, order=1, mode="nearest")
    return out


@slice_array.register
def _(r: roi.LineRoi, arr_nd: np.ndarray):
    xs, ys = r.arange()
    return _slice_array_along_line(arr_nd, xs, ys)


@slice_array.register
def _(r: roi.SegmentedLineRoi, arr_nd: np.ndarray):
    xs, ys = r.arange()
    return _slice_array_along_line(arr_nd, xs, ys)


def _slice_array_along_line(arr_nd: NDArray[np.number], xs, ys):
    coords = np.stack([ys, xs], axis=1)
    out = np.empty(arr_nd.shape[:-2] + (coords.shape[0],), dtype=np.float32)
    for sl in np.ndindex(arr_nd.shape[:-2]):
        arr_2d = arr_nd[sl]
        out[sl] = ndi.map_coordinates(arr_2d, coords, order=1, mode="nearest")
    return out


@register_function(
    title="Measure ROIs ...",
    types=StandardType.IMAGE,
    menus=["tools/image/roi", "/model_menu/roi"],
    command_id="himena-image:roi-measure",
)
def roi_measure(model: WidgetDataModel) -> Parametric:
    metrics_choices = ["mean", "std", "min", "max", "sum"]
    arr = wrap_array(model.value)
    if not isinstance(meta := model.metadata, model_meta.ImageMeta):
        raise ValueError("Image must have an ImageMeta.")
    if axes := meta.axes:
        axis_names = [axis.name for axis in axes[:-2]]
    else:
        axis_names = [f"axis_{i}" for i in range(arr.ndim - 2)]
    axes_choices = [(axis_name, i) for i, axis_name in enumerate(axis_names)]

    @configure_gui(
        metrics={"choices": metrics_choices, "widget_type": "Select"},
        along={"choices": axes_choices, "widget_type": "Select"},
        run_async=True,
    )
    def run_measure(
        metrics: list[str],
        along: list[int],
    ) -> WidgetDataModel:
        if isinstance(meta.rois, roi.RoiListModel):
            rois = meta.rois
        else:
            rois = meta.rois()
        if len(rois) == 0:
            raise ValueError("No ROIs to measure.")

        ndindex_shape = tuple(arr.shape[i] for i in along)
        funcs = [getattr(np, metric) for metric in metrics]
        out: dict[str, list] = {}
        for along_i in along:
            axis_name = axis_names[along_i]
            out[axis_name] = []
        out["name"] = []
        for metric in metrics:
            if metric in axis_names:
                raise ValueError(
                    f"Name collision between axis names and metrics: {metric!r}"
                )
            out[metric] = []
        for sl in np.ndindex(ndindex_shape):
            for indices, each_roi in rois.iter_with_indices():
                sl_placeholder = list(indices)
                for i, along_i in enumerate(along):
                    sl_placeholder[along_i] = sl[i]
                arr_slice = arr.get_slice(tuple(sl_placeholder))
                target = slice_array(each_roi, arr_slice)
                out["name"].append(each_roi.name)
                for sl_i, axis_name in zip(sl, axis_names):
                    out[axis_name].append(sl_i)
                for func, metric in zip(funcs, metrics):
                    out[metric].append(func(target))
        return WidgetDataModel(
            value=out,
            type=StandardType.DATAFRAME,
            title=f"Results of {model.title}",
        )

    return run_measure
