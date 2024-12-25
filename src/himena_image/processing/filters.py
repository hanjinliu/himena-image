from typing import Annotated, Literal
import impy as ip

from himena import WidgetDataModel, Parametric
from himena.consts import StandardType
from himena.plugins import register_function, configure_gui
from himena.standards.model_meta import ImageMeta
from himena_image.consts import PaddingMode
from himena_image.utils import make_dims_annotation, model_to_image, image_to_model

MENUS = ["image/filter", "/model_menu/filter"]


@register_function(
    title="Gaussian Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def gaussian_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)},
        preview=True,
        run_async=True,
    )
    def run_gaussian_filter(
        sigma: Annotated[float, {"min": 0.0}] = 1.0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.gaussian_filter(sigma=sigma, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_gaussian_filter


@register_function(
    title="Median Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def median_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)},
        preview=True,
        run_async=True,
    )
    def run_median_filter(
        radius: Annotated[float, {"min": 0.0}] = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.median_filter(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_median_filter


@register_function(
    title="Mean Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def mean_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_mean_filter(
        radius: Annotated[float, {"min": 0.0}],
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.mean_filter(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_mean_filter


@register_function(
    title="STD Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def std_filter(model: WidgetDataModel) -> Parametric:
    """Standard deviation filter."""

    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_std_filter(
        radius: Annotated[float, {"min": 0.0}],
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.std_filter(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_std_filter


@register_function(
    title="Coef Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def coef_filter(model: WidgetDataModel) -> Parametric:
    """Coefficient of variation filter."""

    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_coef_filter(
        radius: Annotated[float, {"min": 0.0}],
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.coef_filter(radius, mode=mode, cval=cval, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_coef_filter


@register_function(
    title="Difference of Gaussian (DoG) Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def dog_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_dog_filter(
        sigma_low: Annotated[float, {"min": 0.0}] = 1.0,
        sigma_high: Annotated[float, {"min": 0.0}] = 1.6,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.dog_filter(sigma_low, sigma_high, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_dog_filter


@register_function(
    title="Laplacian Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def laplacian_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_laplacian_filter(
        radius: Annotated[int, {"min": 1}] = 1,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.laplacian_filter(radius=radius, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_laplacian_filter


@register_function(
    title="Difference of Hessian (DoH) Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def doh_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_doh_filter(
        sigma: Annotated[float, {"min": 0.0}] = 1.0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.doh_filter(sigma, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_doh_filter


@register_function(
    title="Laplacian of Gaussian (LoG) Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def log_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_log_filter(
        sigma: Annotated[float, {"min": 0.0}] = 1.0,
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.log_filter(sigma, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_log_filter


@register_function(
    title="Rolling ball ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def rolling_ball(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)},
        run_async=True,
    )
    def run_rolling_ball(
        radius: Annotated[float, {"min": 0.0}] = 30.0,
        dimension: int = 2,
        return_background: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model)
        out = img.rolling_ball(radius, return_bg=return_background, dims=dimension)
        return image_to_model(out, orig=model)

    return run_rolling_ball


@register_function(
    title="Threshold ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def threshold(model: WidgetDataModel) -> Parametric:
    from skimage.filters import threshold_yen

    img = ip.asarray(model.value)
    if img.dtype.kind == "f":
        wdgt = "FloatSlider"
    elif img.dtype.kind in "ui":
        wdgt = "Slider"
    else:
        raise ValueError(f"Unsupported dtype: {img.dtype}")
    if isinstance(meta := model.metadata, ImageMeta):
        if inds := meta.current_indices:
            value = threshold_yen(img.value[inds], nbins=128)
        else:
            value = img.value.mean()

    thresh_options = {
        "min": img.min(),
        "max": img.max(),
        "value": value,
        "widget_type": wdgt,
    }

    @configure_gui(threshold=thresh_options, preview=True, run_async=True)
    def run_threshold(
        threshold,
        dark_background: bool = True,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.threshold(threshold)
        if not dark_background:
            out = ~out
        model_out = image_to_model(out, orig=model, is_previewing=is_previewing)
        model_out.type = StandardType.IMAGE_BINARY
        return model_out

    return run_threshold


@register_function(
    title="Edge Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def edge_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        dimension={"choices": make_dims_annotation(model)}, preview=True, run_async=True
    )
    def run_edge_filter(
        method: Literal["sobel", "prewitt", "scharr", "farid"],
        dimension: int = 2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.edge_filter(method, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_edge_filter


@register_function(
    title="Smooth mask ...",
    menus=MENUS,
    types=[StandardType.IMAGE_BINARY],
)
def smooth_mask(model: WidgetDataModel) -> Parametric:
    @configure_gui(
        title="Smooth mask",
        dimension={"choices": make_dims_annotation(model)},
        preview=True,
        run_async=True,
    )
    def run_smooth_mask(
        sigma: Annotated[float, {"min": 0.0}] = 1.0,
        dilate_radius: Annotated[float, {"min": 0.0}] = 1.0,
        dark_background: bool = True,
        dimension: int = 2,
    ) -> WidgetDataModel:
        out = model_to_image(model).smooth_mask(
            sigma=sigma,
            dilate_radius=dilate_radius,
            mask_light=not dark_background,
            dims=dimension,
        )
        return image_to_model(out, orig=model)

    return run_smooth_mask
