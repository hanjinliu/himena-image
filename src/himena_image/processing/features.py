from typing import Annotated
import impy as ip

from himena import WidgetDataModel, Parametric
from himena.plugins import register_function
from himena_image.utils import image_to_model


@register_function(
    title="Label ...",
    menus="image/features",
    types=["image.binary"],
)
def label(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.Label]:
    def run_label(
        dimension: int = 2,
        connectivity: int = 1,
    ) -> WidgetDataModel[ip.ImgArray]:
        img = model.value
        out = img.label(connectivity=connectivity, dims=dimension)
        del img.labels
        return model.with_value(out)

    return run_label


@register_function(
    title="Region Properties ...",
    menus="image/features",
    types=["image.label"],
)
def region_properties(model: WidgetDataModel[ip.ImgArray]) -> Parametric:
    def run_region_properties(
        label: WidgetDataModel[ip.Label],
        properties: Annotated[
            list[str], {"choices": ["area", "centroid", "intensity_mean"]}
        ],
    ) -> WidgetDataModel:
        img = model.value
        img.labels = label
        table = img.regionprops(properties=properties)
        dict_ = {}
        for key, prop in table.items():
            dict_[key] = prop.ravel()
        return image_to_model(table)

    return run_region_properties
