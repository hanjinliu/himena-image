from typing import Literal
import impy as ip

from himena import WidgetDataModel, Parametric
from himena.plugins import register_function, configure_gui
from himena_image.consts import PaddingMode
from himena_image.utils import make_dims_annotation


@register_function(
    title="Gaussian Filter ...",
    menus="image/filter",
    types=["image"],
)
def gaussian_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_gaussian_filter(
        sigma: float, dimension: int
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).gaussian_filter(sigma=sigma, dims=dimension)
        return model.with_value(out)

    return run_gaussian_filter


@register_function(
    title="Median Filter ...",
    menus="image/filter",
    types=["image"],
)
def median_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_median_filter(
        radius: float = 1.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).median_filter(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_median_filter


@register_function(
    title="Mean Filter ...",
    menus="image/filter",
    types=["image"],
)
def mean_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_mean_filter(
        radius: float,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).mean_filter(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_mean_filter


@register_function(
    title="Top-hat Filter ...",
    menus="image/filter",
    types=["image"],
)
def tophat(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_tophat(
        radius: float = 30.0,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).tophat(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_tophat


@register_function(
    title="Difference of Gaussian (DoG) Filter ...",
    menus="image/filter",
    types=["image"],
)
def dog_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_dog_filter(
        sigma_low: float, sigma_high: float, dimension: int
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).dog_filter(sigma_low, sigma_high, dims=dimension)
        return model.with_value(out)

    return run_dog_filter


@register_function(
    title="Laplacian Filter ...",
    menus="image/filter",
    types=["image"],
)
def laplacian_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_laplacian_filter(
        radius: int = 1,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).laplacian_filter(radius=radius, dims=dimension)
        return model.with_value(out)

    return run_laplacian_filter


@register_function(
    title="Laplacian of Gaussian (LoG) Filter ...",
    menus="image/filter",
    types=["image"],
)
def log_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_log_filter(sigma: float, dimension: int) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).log_filter(sigma, dims=dimension)
        return model.with_value(out)

    return run_log_filter


@register_function(
    title="Dilation ...",
    menus="image/filter",
    types=["image"],
)
def dilation(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_dilation(
        radius: float,
        mode: PaddingMode,
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).dilation(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_dilation


@register_function(
    title="Erosion ...",
    menus="image/filter",
    types=["image"],
)
def erosion(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_erosion(
        radius: float,
        mode: PaddingMode,
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).erosion(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_erosion


@register_function(
    title="Opening ...",
    menus="image/filter",
    types=["image"],
)
def opening(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_opening(
        radius: float,
        mode: PaddingMode,
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).opening(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_opening


@register_function(
    title="Closing ...",
    menus="image/filter",
    types=["image"],
)
def closing(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_closing(
        radius: float,
        mode: PaddingMode,
        cval: float = 0,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).closing(
            radius, mode=mode, cval=cval, dims=dimension
        )
        return model.with_value(out)

    return run_closing


@register_function(
    title="Threshold ...",
    menus="image/filter",
    types=["image"],
    preview=True,
)
def threshold(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    img = model.value
    if img.dtype.kind == "f":
        wdgt = "FloatSlider"
    elif img.dtype.kind in "ui":
        wdgt = "Slider"
    else:
        raise ValueError(f"Unsupported dtype: {img.dtype}")

    @configure_gui(threshold={"min": img.min(), "max": img.max(), "widget_type": wdgt})
    def run_threshold(
        threshold: float,
        dark_background: bool = True,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).threshold(threshold)
        if not dark_background:
            out = ~out
        return model.model_copy(update={"value": out, "type": "image.binary"})

    return run_threshold


@register_function(
    title="Edge Filter ...",
    menus="image/filter",
    types=["image"],
)
def edge_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_edge_filter(
        method: Literal["sobel", "prewitt", "scharr", "farid"],
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).edge_filter(method, dims=dimension)
        return model.with_value(out)

    return run_edge_filter
