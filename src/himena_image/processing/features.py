import impy as ip

from himena import StandardType, WidgetDataModel, Parametric
from himena.plugins import register_function, configure_gui
from himena_image.utils import (
    label_to_model,
    make_dims_annotation,
    model_to_image,
    norm_dims,
)

MENUS = ["image/analyze/features", "/model_menu/analyze/features"]


@register_function(
    title="Label ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    run_async=True,
    command_id="himena-image:label",
)
def label(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        connectivity={"choices": [1, 2, 3]},
        dimension={"choices": make_dims_annotation(model)},
    )
    def run_label(
        connectivity: int = 1,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.Label]:
        img = model_to_image(model)
        out = img.label(connectivity=connectivity, dims=norm_dims(dimension, img.axes))
        return label_to_model(out, orig=model)

    return run_label


@register_function(
    title="Peak local maxima ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    run_async=True,
    command_id="himena-image:peak_local_max",
)
def peak_local_max(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        labels={"types": [StandardType.IMAGE_LABELS]},
        dimension={"choices": make_dims_annotation(model)},
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
        img = model_to_image(model)
        if labels is not None:
            img.labels = labels
        out = img.peak_local_max(
            min_distance=min_distance,
            percentile=percentile,
            topn=topn or float("inf"),
            topn_per_label=topn_per_label or float("inf"),
            exclude_border=exclude_border,
            use_labels=labels is not None,
            dims=norm_dims(dimension, img.axes),
        )
        df = {k: out[k].to_numpy() for k in out.columns}
        return WidgetDataModel(
            value=df, type=StandardType.DATAFRAME, title=f"Peaks of {model.title}"
        )

    return run_peak_local_max


REGIONPROPS_CHOICES = [
    "area", "area_bbox", "area_convex", "area_filled", "axis_major_length",
    "axis_minor_length", "bbox", "centroid", "centroid_local", "centroid_weighted",
    "centroid_weighted_local", "coords_scaled", "coords", "eccentricity",
    "equivalent_diameter_area", "euler_number", "extent", "feret_diameter_max",
    "image_intensity", "inertia_tensor", "inertia_tensor_eigvals", "intensity_max",
    "intensity_mean", "intensity_min", "intensity_std", "label", "moments",
    "moments_central", "moments_hu", "moments_normalized", "moments_weighted",
    "moments_weighted_central", "moments_weighted_hu", "moments_weighted_normalized",
    "num_pixels", "orientation", "perimeter", "perimeter_crofton", "solidity",
]  # fmt: skip


@register_function(
    title="Region Properties ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    run_async=True,
    command_id="himena-image:region_properties",
)
def region_properties(model: WidgetDataModel) -> Parametric:
    """Measure region properties of an image."""

    @configure_gui(
        labels={"types": [StandardType.IMAGE_LABELS]},
        properties={"choices": REGIONPROPS_CHOICES, "widget_type": "Select"},
    )
    def run_region_properties(
        labels: WidgetDataModel[ip.Label],
        properties: list[str] = ["intensity_mean"],
    ) -> WidgetDataModel:
        img = model_to_image(model)
        img.labels = labels.value
        table = img.regionprops(properties=properties)
        dict_ = {}
        for key, prop in table.items():
            dict_[key] = prop.ravel()
        return WidgetDataModel(
            value=dict_,
            type=StandardType.DATAFRAME,
            title=f"Properties of {model.title}",
        )

    return run_region_properties
