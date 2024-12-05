from typing import Annotated, Literal

from himena import WidgetDataModel, Parametric
from himena.consts import StandardType
from himena.plugins import register_function, configure_gui
from himena_image.consts import PaddingMode
from himena_image.utils import make_dims_annotation, image_to_model, model_to_image

MENUS = ["image/transform", "/model_menu/transform"]


@register_function(
    title="Shift ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def shift(model: WidgetDataModel) -> Parametric:
    @configure_gui(preview=True, dimension={"choices": make_dims_annotation(model)})
    def run_shift(
        shift: tuple[float, float],
        mode: PaddingMode = "constant",
        value: float = 0.0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        if len(shift) != dimension:
            raise ValueError("The length of shift must be equal to the dimension.")
        img = model_to_image(model, is_previewing)
        out = img.shift(shift, mode=mode, value=value, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_shift


@register_function(
    title="Rotate ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def rotate(model: WidgetDataModel) -> Parametric:
    @configure_gui(preview=True, dimension={"choices": make_dims_annotation(model)})
    def run_rotate(
        degree: Annotated[float, {"min": -90, "max": 90, "widget_type": "FloatSlider"}],
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        if abs(degree) < 1e-4:
            return model
        img = model_to_image(model, is_previewing)
        out = img.rotate(degree, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_rotate


@register_function(
    title="Flip ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def flip(model: WidgetDataModel) -> Parametric:
    @configure_gui(preview=True, dimension={"choices": make_dims_annotation(model)})
    def run_flip(
        axis: str,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        idx = img.axes.index(axis)
        slices = [slice(None)] * img.ndim
        slices[idx] = slice(None, None, -1)
        out = img[tuple(slices)]
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_flip


@register_function(
    title="Zoom ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def zoom(model: WidgetDataModel) -> Parametric:
    @configure_gui(preview=True, dimension={"choices": make_dims_annotation(model)})
    def run_zoom(
        factor: float,
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        same_shape: bool = False,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.zoom(
            factor,
            mode=mode,
            cval=cval,
            same_shape=same_shape,
            dims=dimension,
        )
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_zoom


@register_function(
    title="Bin ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def bin(model: WidgetDataModel) -> Parametric:
    @configure_gui(preview=True, dimension={"choices": make_dims_annotation(model)})
    def run_bin(
        bin_size: Literal[2, 3, 4, 5, 6, 7, 8],
        method: str = "mean",
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.binning(bin_size, method=method, check_edges=False, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_bin
