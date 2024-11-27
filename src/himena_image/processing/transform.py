from typing import Annotated, Literal
import impy as ip

from himena import WidgetDataModel, Parametric
from himena.consts import StandardType
from himena.plugins import register_function
from himena_image.consts import PaddingMode

MENUS = ["image/transform", "/model_menu/transform"]


@register_function(
    title="Shift ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def shift(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    def run_shift(
        shift: tuple[float, float],
        mode: PaddingMode = "constant",
        value: float = 0.0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        if len(shift) != dimension:
            raise ValueError("The length of shift must be equal to the dimension.")
        out = model.value.shift(shift, mode=mode, value=value, dims=dimension)
        return model.with_value(out)

    return run_shift


@register_function(
    title="Rotate ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    preview=True,
)
def rotate(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    def run_rotate(
        degree: Annotated[float, {"min": -90, "max": 90, "widget_type": "FloatSlider"}],
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        if abs(degree) < 1e-4:
            return model
        out = ip.asarray(model.value).rotate(
            degree, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_rotate


@register_function(
    title="Flip ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def flip(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    def run_flip(
        axis: str,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).isel({axis: slice(None, None, -1)})
        return model.with_value(out)

    return run_flip


@register_function(
    title="Zoom ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def zoom(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    def run_zoom(
        factor: float,
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        same_shape: bool = False,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).zoom(
            factor,
            mode=mode,
            cval=cval,
            same_shape=same_shape,
            dims=dimension,
        )
        return model.with_value(out)

    return run_zoom


@register_function(
    title="Bin ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def bin(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    def run_bin(
        bin_size: Literal[2, 3, 4, 5, 6, 7, 8],
        method: str = "mean",
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).binning(
            bin_size, method=method, check_edges=False, dims=dimension
        )
        return model.with_value(out)

    return run_bin
