from himena import WidgetDataModel, Parametric
from himena.plugins import register_function, configure_gui, configure_submenu
from himena_image.utils import make_dims_annotation, image_to_model, model_to_image
from himena.consts import StandardType

MENUS = ["image/fft", "/model_menu/fft"]

configure_submenu("image/fft", title="Fourier transform")
configure_submenu("/model_menu/fft", title="Fourier transform")


@register_function(
    title="FFT ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def fft(model: WidgetDataModel) -> Parametric:
    """Fast Fourier transformation of an image."""

    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_fft(
        origin_in_center: bool = True,
        double_precision: bool = False,
        dimension=2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.fft(
            shift=origin_in_center,
            double_precision=double_precision,
            dims=dimension,
        )
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_fft


@register_function(
    title="IFFT ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def ifft(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)})
    def run_ifft(
        return_real: bool = True,
        origin_in_center: bool = True,
        double_precision: bool = False,
        dimension=2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        img = model_to_image(model, is_previewing)
        out = img.ifft(
            real=return_real,
            shift=origin_in_center,
            double_precision=double_precision,
            dims=dimension,
        )
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_ifft


@register_function(
    title="Low-pass Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def lowpass_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)}, preview=True)
    def run_lowpass_filter(
        cutoff: float = 0.2,
        order: int = 2,
        dimension=2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        # TODO: lazy
        img = model_to_image(model)
        out = img.lowpass_filter(cutoff=cutoff, order=order, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_lowpass_filter


@register_function(
    title="High-pass Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def highpass_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)}, preview=True)
    def run_highpass_filter(
        cutoff: float = 0.2,
        order: int = 2,
        dimension=2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        # TODO: lazy
        img = model_to_image(model)
        out = img.highpass_filter(cutoff=cutoff, order=order, dims=dimension)
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_highpass_filter


@register_function(
    title="Band-pass Filter ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
)
def bandpass_filter(model: WidgetDataModel) -> Parametric:
    @configure_gui(dimension={"choices": make_dims_annotation(model)}, preview=True)
    def run_bandpass_filter(
        cuton: float = 0.2,
        cutoff: float = 0.5,
        order: int = 2,
        dimension=2,
        is_previewing: bool = False,
    ) -> WidgetDataModel:
        # TODO: lazy
        img = model_to_image(model)
        out = img.bandpass_filter(
            cuton=cuton, cutoff=cutoff, order=order, dims=dimension
        )
        return image_to_model(out, orig=model, is_previewing=is_previewing)

    return run_bandpass_filter
