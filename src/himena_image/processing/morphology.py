from typing import Annotated

from himena import WidgetDataModel, Parametric
from himena.consts import StandardType
from himena.plugins import register_function, configure_gui
from himena_image.consts import PaddingMode
from himena_image.utils import make_dims_annotation, image_to_model, model_to_image

MENUS = ["image/morphology", "/model_menu/morphology"]


@register_function(
    title="Dilation ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:dilation",
)
def dilation(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_dilation(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.dilation(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_dilation


@register_function(
    title="Erosion ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:erosion",
)
def erosion(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_erosion(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.erosion(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_erosion


@register_function(
    title="Opening ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:opening",
)
def opening(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_opening(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.opening(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_opening


@register_function(
    title="Closing ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:closing",
)
def closing(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_closing(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.closing(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_closing


@register_function(
    title="Top-hat Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:tophat",
)
def tophat(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_tophat(
        radius: Annotated[float, {"min": 0.0}] = 30.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.tophat(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_tophat


@register_function(
    title="skeletonize ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:skeletonize",
)
def skeletonize(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_skeletonize(
        radius: Annotated[float, {"min": 0.0}] = 0.0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.skeletonize(radius=radius, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_skeletonize
