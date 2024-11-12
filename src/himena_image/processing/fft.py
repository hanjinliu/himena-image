import impy as ip

from himena import WidgetDataModel, Parametric
from himena.plugins import register_function, configure_gui
from himena_image.utils import make_dims_annotation

MENU = "image/fft"


@register_function(
    title="FFT ...",
    menus=MENU,
    types=["image"],
)
def fft(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_fft(
        origin_in_center: bool = True,
        double_precision: bool = False,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).fft(
            shift=origin_in_center,
            double_precision=double_precision,
            dims=dimension,
        )
        return model.with_value(out)

    return run_fft


@register_function(
    title="IFFT ...",
    menus=MENU,
    types=["image"],
)
def ifft(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_ifft(
        return_real: bool = True,
        origin_in_center: bool = True,
        double_precision: bool = False,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).ifft(
            real=return_real,
            shift=origin_in_center,
            double_precision=double_precision,
            dims=dimension,
        )
        return model.with_value(out)

    return run_ifft


@register_function(
    title="Low-pass Filter ...",
    menus=MENU,
    types=["image"],
    preview=True,
)
def lowpass_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_lowpass_filter(
        cutoff: float = 0.2,
        order: int = 2,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).lowpass_filter(
            cutoff=cutoff, order=order, dims=dimension
        )
        return model.with_value(out)

    return run_lowpass_filter


@register_function(
    title="High-pass Filter ...",
    menus=MENU,
    types=["image"],
    preview=True,
)
def highpass_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_highpass_filter(
        cutoff: float = 0.2,
        order: int = 2,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).highpass_filter(
            cutoff=cutoff, order=order, dims=dimension
        )
        return model.with_value(out)

    return run_highpass_filter


@register_function(
    title="Band-pass Filter ...",
    menus=MENU,
    types=["image"],
    preview=True,
)
def bandpass_filter(model: WidgetDataModel[ip.ImgArray]) -> Parametric[ip.ImgArray]:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_bandpass_filter(
        low_cutoff: float = 0.2,
        high_cutoff: float = 0.5,
        order: int = 2,
        dimension: int = 2,
    ) -> WidgetDataModel[ip.ImgArray]:
        out = ip.asarray(model.value).bandpass_filter(
            low_cutoff=low_cutoff, high_cutoff=high_cutoff, order=order, dims=dimension
        )
        return model.with_value(out)

    return run_bandpass_filter
