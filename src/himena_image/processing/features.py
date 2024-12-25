from typing import Annotated
import impy as ip

from himena import StandardType, WidgetDataModel, Parametric
from himena.plugins import register_function, configure_gui
from himena_image.utils import image_to_model, make_dims_annotation

MENUS = ["image/features", "/model_menu/features"]


@register_function(
    title="Label ...",
    menus=MENUS,
    types=[StandardType.IMAGE_BINARY],
)
def label(model: WidgetDataModel[ip.ImgArray]) -> Parametric:
    @configure_gui(
        connectivity={"choices": [1, 2, 3]},
        dimension={"choices": make_dims_annotation(model)},
        run_async=True,
    )
    def run_label(
        connectivity: int = 1,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.Label]:
        img = model.value
        out = img.label(connectivity=connectivity, dims=dimension)
        del img.labels
        return model.with_value(out, type=StandardType.IMAGE_LABELS)

    return run_label


@register_function(
    title="Peak local maxima ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def peak_local_max(model: WidgetDataModel[ip.ImgArray]) -> Parametric:
    @configure_gui(
        labels={"types": [StandardType.IMAGE_LABELS]},
        dimension={"choices": make_dims_annotation(model)},
        run_async=True,
    )
    def run_peak_local_max(
        min_distance: float = 1.0,
        percentile: float | None = None,
        topn: int | None = None,
        exclude_border: bool = True,
        labels: WidgetDataModel | None = None,
        topn_per_label: int | None = None,
        dimension: int = 2,
    ) -> WidgetDataModel:
        img = model.value
        if labels is not None:
            img.labels = labels
        try:
            out = img.peak_local_max(
                min_distance=min_distance,
                percentile=percentile,
                topn=topn if topn is not None else float("inf"),
                topn_per_label=topn_per_label
                if topn_per_label is not None
                else float("inf"),
                exclude_border=exclude_border,
                use_labels=labels is not None,
                dims=dimension,
            )
        finally:
            del img.labels
        return model.with_value(
            out.values,
            type=StandardType.COORDINATES,
            title=f"Peaks of {model.title}",
        )

    return run_peak_local_max


@register_function(
    title="Region Properties ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def region_properties(model: WidgetDataModel[ip.ImgArray]) -> Parametric:
    @configure_gui(
        labels={"types": [StandardType.IMAGE_LABELS]},
        run_async=True,
    )
    def run_region_properties(
        labels: WidgetDataModel[ip.Label],
        properties: Annotated[
            list[str], {"choices": ["area", "centroid", "intensity_mean"]}
        ],
    ) -> WidgetDataModel:
        img = model.value
        img.labels = labels
        table = img.regionprops(properties=properties)
        dict_ = {}
        for key, prop in table.items():
            dict_[key] = prop.ravel()
        return image_to_model(table)

    return run_region_properties
