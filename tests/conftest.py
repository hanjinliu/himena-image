import numpy as np
from himena import WidgetDataModel, StandardType
from himena.standards.model_meta import ImageMeta, ArrayAxis
from himena.testing import install_plugin
import pytest

@pytest.fixture(scope="session", autouse=True)
def init_pytest(request):
    install_plugin("himena-image")

@pytest.fixture(scope="function")
def image_data():
    rng = np.random.default_rng(0)
    return WidgetDataModel(
        value=rng.normal(size=(4, 5, 2, 6, 5)),
        type=StandardType.IMAGE,
        metadata=ImageMeta(
            axes=[
                ArrayAxis(name="t", scale=0.2, unit="sec"),
                ArrayAxis(name="z", scale=1.5, unit="um"),
                ArrayAxis(name="c", labels=["green", "red"]),
                ArrayAxis(name="y", scale=0.9, unit="um"),
                ArrayAxis(name="x", scale=0.9, unit="um"),
            ],
            channel_axis=2,
            is_rgb=False,
        ),
    )
