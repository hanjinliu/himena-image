from typing import Annotated, Literal

from himena import WidgetDataModel, Parametric
from himena.consts import StandardType
from himena.plugins import register_function, configure_gui
from himena_image.consts import PaddingMode, InterpolationOrder
from himena_image.utils import make_dims_annotation, image_to_model, model_to_image

MENUS = ["image/transform", "/model_menu/transform"]


@register_function(
    title="Shift ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:shift",
)
def shift(model: WidgetDataModel) -> Parametric:
    shape = model.value.shape
    if len(shape) < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    max_size = max(shape)

    @configure_gui(
        preview=True,
        shift={
            "options": {
                "widget_type": "FloatSpinBox",
                "min": -max_size,
                "max": max_size,
            }
        },
        run_async=True,
    )
    def run_shift(
        shift: tuple[float, float],
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.shift(shift, mode=mode, cval=cval, dims=2)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_shift


@register_function(
    title="Rotate ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:rotate",
)
def rotate(model: WidgetDataModel) -> Parametric:
    @configure_gui(preview=True, run_async=True)
    def run_rotate(
        degree: Annotated[float, {"min": -90, "max": 90, "widget_type": "FloatSlider"}],
        order: InterpolationOrder = 3,
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        if abs(degree) < 1e-4:
            return model
        img = model_to_image(model, is_previewing)
        out = img.rotate(degree, mode=mode, cval=cval, order=order)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_rotate


@register_function(
    title="Flip ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:flip",
)
def flip(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        preview=True, dimension={"choices": make_dims_annotation(model)}, run_async=True
    )
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
    command_id="himena-image:zoom",
)
def zoom(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        preview=True, dimension={"choices": make_dims_annotation(model)}, run_async=True
    )
    def run_zoom(
        factor: float,
        order: InterpolationOrder = 3,
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        same_shape: bool = False,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.zoom(
            factor,
            order=order,
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
    command_id="himena-image:bin",
)
def bin(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        preview=True, dimension={"choices": make_dims_annotation(model)}, run_async=True
    )
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


@register_function(
    title="Drift correction ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:drift_correction",
)
def drift_correction(model: WidgetDataModel) -> Parametric:
    """Correct drift in the image."""
    img = model_to_image(model)
    along_choices = [str(a) for a in img.axes]
    if "t" in along_choices:
        along_default = "t"
    elif "z" in along_choices:
        along_default = "z"
    else:
        along_default = along_choices[0]

    @configure_gui(
        along={"choices": along_choices, "value": along_default},
        dimension={"choices": make_dims_annotation(model)},
        run_async=True,
    )
    def run_drift_correction(
        along: str,
        reference: str = "",
        zero_ave: bool = True,
        max_shift: float | None = None,
        order: InterpolationOrder = 1,
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        dimension: int = 2,
    ) -> WidgetDataModel:
        img = model_to_image(model)
        out = img.drift_correction(
            ref=reference or None,
            zero_ave=zero_ave,
            along=along,
            max_shift=max_shift,
            mode=mode,
            cval=cval,
            order=order,
            dims=dimension,
        )
        return image_to_model(out, orig=model)

    return run_drift_correction
