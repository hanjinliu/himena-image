from typing import Annotated, Literal
import impy as ip

from himena import WidgetDataModel, Parametric, AppContext
from himena.plugins import register_function, configure_gui
from himena.consts import StandardType
from himena_image.utils import image_to_model

MENU = "image/calculate"


@register_function(
    title="Add images ...",
    menus=MENU,
    enablement=AppContext.num_sub_windows > 1,
)
def add() -> Parametric:
    def run_add(
        image_1: Annotated[
            WidgetDataModel[ip.ImgArray], {"types": [StandardType.IMAGE]}
        ],
        image_2: Annotated[
            WidgetDataModel[ip.ImgArray], {"types": [StandardType.IMAGE]}
        ],
    ) -> WidgetDataModel[ip.ImgArray]:
        out = image_1.value + image_2.value
        return image_1.with_value(out)

    return run_add


@register_function(
    title="Subtract images ...",
    menus=MENU,
    enablement=AppContext.num_sub_windows > 1,
)
def subtract() -> Parametric:
    def run_subtract(
        image_1: Annotated[
            WidgetDataModel[ip.ImgArray], {"types": [StandardType.IMAGE]}
        ],
        image_2: Annotated[
            WidgetDataModel[ip.ImgArray], {"types": [StandardType.IMAGE]}
        ],
    ) -> WidgetDataModel[ip.ImgArray]:
        out = image_1.value - image_2.value
        return image_1.with_value(out)

    return run_subtract


@register_function(
    title="Multiply images ...",
    menus=MENU,
    enablement=AppContext.num_sub_windows > 1,
)
def multiply() -> Parametric:
    def run_multiply(
        image_1: Annotated[
            WidgetDataModel[ip.ImgArray], {"types": [StandardType.IMAGE]}
        ],
        image_2: Annotated[
            WidgetDataModel[ip.ImgArray], {"types": [StandardType.IMAGE]}
        ],
    ) -> WidgetDataModel[ip.ImgArray]:
        out = image_1.value * image_2.value
        return image_1.with_value(out)

    return run_multiply


@register_function(
    title="Projection ...",
    menus=MENU,
    types=[StandardType.IMAGE],
)
def projection(model: WidgetDataModel[ip.ImgArray]) -> Parametric:
    """Project the image along an axis."""
    img = ip.asarray(model.value)
    axis_choices = [str(a) for a in img.axes]
    if "z" in axis_choices:
        value = "z"
    elif "t" in axis_choices:
        value = "t"
    else:
        value = axis_choices[0]

    @configure_gui(
        axis={"choices": axis_choices, "value": value},
    )
    def run_projection(
        axis: str,
        method: Literal["mean", "max", "min", "sum", "std"],
    ) -> WidgetDataModel[ip.ImgArray]:
        out = model.value.proj(axis=axis, method=method)
        return image_to_model(out, title=model.title)

    return run_projection


@register_function(
    title="Invert",
    menus=MENU,
    types=[StandardType.IMAGE],
)
def invert(model: WidgetDataModel[ip.ImgArray]) -> WidgetDataModel[ip.ImgArray]:
    """Invert the image."""
    out = -model.value
    return model.with_value(out)
