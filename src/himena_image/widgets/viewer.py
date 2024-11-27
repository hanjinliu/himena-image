from __future__ import annotations

from collections.abc import Hashable
from typing import Any, TYPE_CHECKING
from enum import Enum, auto
import cmap
from qtpy import QtWidgets as QtW, QtCore
import numpy as np
import impy as ip
from impy.arrays.axesmixin import AxesMixin

from ndv import NDViewer, DataWrapper
from superqt import QEnumComboBox
from himena.consts import StandardType
from himena.types import WidgetDataModel
from himena.model_meta import ImageMeta
from himena.plugins import protocol_override

if TYPE_CHECKING:
    from ndv import Indices


class ComplexConversionRule(Enum):
    ABS = auto()
    REAL = auto()
    IMAG = auto()
    PHASE = auto()
    LOG_ABS = auto()

    def apply(self, data: np.ndarray) -> np.ndarray:
        if self == ComplexConversionRule.ABS:
            return np.abs(data)
        elif self == ComplexConversionRule.REAL:
            return data.real
        elif self == ComplexConversionRule.IMAG:
            return data.imag
        elif self == ComplexConversionRule.PHASE:
            return np.angle(data)
        elif self == ComplexConversionRule.LOG_ABS:
            return np.log(np.abs(data) + 1e-10)
        raise ValueError(f"Unknown complex conversion rule: {self}")


class ModelDataWrapper(DataWrapper[AxesMixin]):
    def __init__(self, model: WidgetDataModel):
        if isinstance(img := model.value, np.ndarray):
            img = ip.asarray(img)
        if not isinstance(img := model.value, AxesMixin):
            raise ValueError(f"Expected an impy MetaArray object, got {type(img)}.")
        if not isinstance(model.metadata, ImageMeta):
            raise ValueError("Expected image meta data.")
        super().__init__(img)
        self._meta = model.metadata
        self._type = model.type
        self._dtype = img.dtype
        self._complex_conversion = ComplexConversionRule.ABS

    @classmethod
    def supports(cls, obj: Any) -> bool:
        return isinstance(obj, WidgetDataModel)

    def guess_channel_axis(self) -> Hashable | None:
        return self._meta.channel_axis

    def isel(self, indexers: Indices) -> np.ndarray:
        """Select a slice from a data store using (possibly) named indices."""

        slices = self._indices_to_slice(indexers)
        out = self._data[slices]
        if isinstance(out, ip.LazyImgArray):
            out = out.compute()
        if self._dtype.kind == "b":
            return out.astype(np.uint8)
        elif self._dtype.kind == "c":
            return self._complex_conversion.apply(out)
        return out

    def _indices_to_slice(self, indexers: Indices) -> tuple[int | slice, ...]:
        slices = [slice(None)] * self._data.ndim
        for k, v in indexers.items():
            if isinstance(k, str):
                idx = self._meta.axes.index(k)
            else:
                idx = k
            slices[idx] = v
        return tuple(slices)


class HimenaImageViewer(NDViewer):
    _data_wrapper: ModelDataWrapper

    def __init__(self):
        super().__init__()
        self._control_widget = QtW.QWidget()
        layout = QtW.QHBoxLayout(self._control_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        spacer = QtW.QWidget()
        spacer.setSizePolicy(
            QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Expanding
        )
        layout.addWidget(spacer)
        self._complex_conversion_rule_cbox = QEnumComboBox(
            enum_class=ComplexConversionRule
        )
        self._complex_conversion_rule_cbox.currentEnumChanged.connect(
            self._on_complex_conversion_rule_changed
        )
        layout.addWidget(self._complex_conversion_rule_cbox)
        layout.addWidget(self._channel_mode_btn)
        layout.addWidget(self._ndims_btn)
        layout.addWidget(self._set_range_btn)
        layout.addWidget(self._add_roi_btn)

    @protocol_override
    def update_model(self, model: WidgetDataModel):
        self.set_data(ModelDataWrapper(model))
        is_complex = self._data_wrapper._dtype.kind == "c"
        self._complex_conversion_rule_cbox.setVisible(is_complex)
        if model.type == StandardType.IMAGE_LABELS:
            ...  # TODO: cyclic colormap
        if is_complex:
            self._complex_conversion_rule_cbox.setCurrentEnum(ComplexConversionRule.ABS)
        else:
            self.refresh()

    @protocol_override
    def to_model(self) -> WidgetDataModel:
        return WidgetDataModel(
            value=self.data,
            type=self.model_type(),
            metadata=ImageMeta(
                current_indices=self._data_wrapper._indices_to_slice(
                    self._dims_sliders.value()
                ),
                axes=[str(a) for a in self.current_indices().keys()],
            ),
        )

    @protocol_override
    def size_hint(self) -> tuple[int, int]:
        return (320, 400)

    @protocol_override
    def model_type(self) -> str:
        return self._data_wrapper._type

    def current_indices(self) -> dict[str, int]:
        return self._dims_sliders.value()

    @protocol_override
    def control_widget(self):
        return self._control_widget

    def _on_complex_conversion_rule_changed(self, enum_: ComplexConversionRule):
        self._data_wrapper._complex_conversion = enum_
        if enum_ is ComplexConversionRule.PHASE:
            cmap_name = "cmocean:phase"
        else:
            cmap_name = "inferno"
        for ctrl in self._lut_ctrls.values():
            ctrl._cmap.setCurrentColormap(cmap.Colormap(cmap_name))
        self.refresh()
