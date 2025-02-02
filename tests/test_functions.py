import pytest
from himena.widgets import MainWindow


@pytest.mark.parametrize(
    "command",
    [
        "himena-image:0-filter-basic:gaussian_filter",
        "himena-image:0-filter-basic:median_filter",
        "himena-image:0-filter-basic:mean_filter",
        "himena-image:1-filter-variance:std_filter",
        "himena-image:1-filter-variance:coef_filter",
        "himena-image:2-filter-comp:dog_filter",
        "himena-image:2-filter-comp:doh_filter",
        "himena-image:2-filter-comp:log_filter",
        "himena-image:2-filter-comp:laplacian_filter",
    ],
)
def test_filter(himena_ui: MainWindow, image_data, command: str):
    win = himena_ui.add_data_model(image_data)
    himena_ui.exec_action(
        command,
        model_context=win.to_model(),
        with_params={},
    )
