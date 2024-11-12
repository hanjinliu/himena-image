from __future__ import annotations
from ndv import NDViewer
from himena.types import WidgetDataModel
from himena.model_meta import ImageMeta


class HimenaImageViewer(NDViewer):
    def __init__(self):
        super().__init__(None)

    def update_model(self, model: WidgetDataModel):
        if isinstance(meta := model.additional_data, ImageMeta):
            if (inds := meta.current_indices) is not None:
                self.set_current_index(inds)

        self.set_data(model.value)

    def to_model(self) -> WidgetDataModel:
        return WidgetDataModel(value=self.data, type="image", additional_data={})

    def size_hint(self) -> tuple[int, int]:
        return (320, 400)

    def model_type(self) -> str:
        return "image"
