import pytest
from himena.widgets import MainWindow


@pytest.mark.parametrize(
    "command",
    [
        "himena-image:0-filter-basic:gaussian-filter",
        "himena-image:0-filter-basic:median-filter",
        "himena-image:0-filter-basic:mean-filter",
        "himena-image:1-filter-variance:std-filter",
        "himena-image:1-filter-variance:coef-filter",
        "himena-image:2-filter-comp:dog-filter",
        "himena-image:2-filter-comp:doh-filter",
        "himena-image:2-filter-comp:log-filter",
        "himena-image:2-filter-comp:laplacian-filter",
    ],
)
def test_filter(make_himena_ui, image_data, command: str):
    ui: MainWindow = make_himena_ui(backend="mock")
    win = ui.add_data_model(image_data)
    ui.exec_action(
        command,
        model_context=win.to_model(),
        with_params={},
    )
