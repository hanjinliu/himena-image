from typing import Annotated
import impy as ip

from himena import WidgetDataModel, Parametric
from himena.consts import StandardType
from himena.plugins import register_function, configure_gui
from himena_image.consts import PaddingMode
from himena_image.utils import make_dims_annotation

MENUS = ["image/morphology", "/model_menu/morphology"]


@register_function(
    title="Dilation ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def dilation(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)}, preview=True)
    def run_dilation(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel:
        out = ip.asarray(model.value).dilation(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_dilation


@register_function(
    title="Erosion ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def erosion(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)}, preview=True)
    def run_erosion(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel:
        out = ip.asarray(model.value).erosion(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_erosion


@register_function(
    title="Opening ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def opening(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)}, preview=True)
    def run_opening(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel:
        out = ip.asarray(model.value).opening(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_opening


@register_function(
    title="Closing ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def closing(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)}, preview=True)
    def run_closing(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel:
        out = ip.asarray(model.value).closing(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_closing


@register_function(
    title="Top-hat Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def tophat(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)}, preview=True)
    def run_tophat(
        radius: Annotated[float, {"min": 0.0}] = 30.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel:
        out = ip.asarray(model.value).tophat(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_tophat
