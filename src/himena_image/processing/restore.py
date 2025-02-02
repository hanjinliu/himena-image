from himena import WidgetDataModel, Parametric
from himena.consts import StandardType
from himena.plugins import register_function, configure_gui
from himena_image.consts import PaddingMode, InterpolationOrder
from himena_image.utils import make_dims_annotation, image_to_model, model_to_image

MENUS = ["image/restore", "/model_menu/restore"]


@register_function(
    title="Drift correction ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:drift_correction",
    run_async=True,
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


@register_function(
    title="Richardson-Lucy deconvolution ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:restore-deconv:lucy-deconv",
    run_async=True,
)
def lucy(model: WidgetDataModel) -> Parametric:
    """Restore image using poin spread function by Richardson-Lucy's method"""

    @configure_gui(
        psf={"types": [StandardType.IMAGE]},
        dimension={"choices": make_dims_annotation(model)},
    )
    def run_lucy(
        psf: WidgetDataModel,
        niter: int = 50,
        dimension: int = 2,
        eps: float = 1e-5,
    ) -> WidgetDataModel:
        img = model_to_image(model)
        out = img.lucy(psf, niter=niter, eps=eps, dims=dimension)
        return image_to_model(out, orig=model)

    return run_lucy


@register_function(
    title="Richardson-Lucy TV deconvolution ...",
    menus=MENUS,
    types=[StandardType.IMAGE],
    command_id="himena-image:restore-deconv:lucy-tv-deconv",
    run_async=True,
)
def lucy_tv(model: WidgetDataModel) -> Parametric:
    """Restore image using poin spread function by Richardson-Lucy's method with total
    variance regularization."""

    @configure_gui(
        psf={"types": [StandardType.IMAGE]},
        dimension={"choices": make_dims_annotation(model)},
    )
    def run_lucy_tv(
        psf: WidgetDataModel,
        niter: int = 50,
        dimension: int = 2,
        lmd: float = 1.0,
        tol: float = 1e-3,
        eps: float = 1e-5,
    ) -> WidgetDataModel:
        img = model_to_image(model)
        out = img.lucy_tv(psf, niter=niter, lmd=lmd, tol=tol, eps=eps, dims=dimension)
        return image_to_model(out, orig=model)

    return run_lucy_tv
