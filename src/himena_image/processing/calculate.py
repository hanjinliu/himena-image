from typing import Annotated, Literal
import impy as ip

from himena import WidgetDataModel, Parametric, AppContext
from himena.plugins import register_function, configure_gui
from himena.consts import StandardType
from himena.standards.model_meta import ImageMeta
from himena.standards import roi
import numpy as np
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


@register_function(
    title="Profile line",
    menus=MENU,
    types=[StandardType.IMAGE],
)
def profile_line(model: WidgetDataModel[ip.ImgArray]) -> Parametric:
    """Profile line."""
    img = ip.asarray(model.value)
    if not isinstance(meta := model.metadata, ImageMeta):
        raise ValueError("Metadata is missing.")
    img_slice = img[meta.current_indices]
    if isinstance(r := meta.current_roi, roi.LineRoi):
        points = [[r.y1, r.x1], [r.y2, r.x2]]
    elif isinstance(r := meta.current_roi, roi.SegmentedLineRoi):
        points = np.stack([r.ys, r.xs], axis=-1)
    else:
        raise TypeError(f"Cannot get profile line from {type(r)}.")
    sliced = img_slice.reslice(points)
    # TODO: convert x, y to dataframe
    return WidgetDataModel(
        value=sliced,
        type=StandardType.DATAFRAME,
        title=model.title,
    )
