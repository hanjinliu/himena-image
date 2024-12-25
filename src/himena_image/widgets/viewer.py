from __future__ import annotations

from typing import Any, TYPE_CHECKING
from enum import Enum, auto
import cmap
from qtpy import QtWidgets as QtW, QtCore
import numpy as np

from ndv import NDViewer, DataWrapper
from superqt import QEnumComboBox
from himena.consts import StandardType
from himena.types import WidgetDataModel
from himena.standards.model_meta import ImageMeta
from himena.plugins import validate_protocol

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


class ModelDataWrapper(DataWrapper):
    def __init__(self, model: WidgetDataModel):
        super().__init__(model.value)
        if not isinstance(meta := model.metadata, ImageMeta):
            raise ValueError("Invalid metadata")
        self._meta = meta
        self._type = model.type
        self._complex_conversion = ComplexConversionRule.ABS

    @classmethod
    def supports(cls, obj: Any) -> bool:
        return isinstance(obj, WidgetDataModel)

    def isel(self, indexers: Indices) -> np.ndarray:
        """Select a slice from a data store using (possibly) named indices."""
        import dask.array as da

        slices = self._indices_to_slice(indexers)
        out = self._data[slices]
        if isinstance(out, da.Array):
            out = out.compute()
        assert isinstance(out, np.ndarray)
        if out.dtype.kind == "b":
            return out.astype(np.uint8)
        elif out.dtype.kind == "c":
            return self._complex_conversion.apply(out)
        return out

    def _indices_to_slice(self, indexers: Indices) -> tuple[int | slice, ...]:
        slices = [slice(None)] * self._data.ndim
        if self._meta.axes is not None:
            axis_names = [a.name for a in self._meta.axes]
        else:
            axis_names = list(range(len(self._data.shape)))
        for k, v in indexers.items():
            if isinstance(k, str):
                idx = axis_names.index(k)
            else:
                idx = k
            slices[idx] = v
        return tuple(slices)

    def sizes(self):
        if axes := self._meta.axes:
            names = [a.name for a in axes]
        else:
            names = list(range(len(self._data.shape)))
        return dict(zip(names, self._data.shape))


class NDImageViewer(NDViewer):
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

    @validate_protocol
    def update_model(self, model: WidgetDataModel):
        self.set_data(ModelDataWrapper(model))
        is_complex = model.value.dtype.kind == "c"
        self._complex_conversion_rule_cbox.setVisible(is_complex)
        if model.type == StandardType.IMAGE_LABELS:
            ...  # TODO: cyclic colormap
        self.set_ndim(min(3, len(model.value.shape)))
        if is_complex:
            self._complex_conversion_rule_cbox.setCurrentEnum(ComplexConversionRule.ABS)
        else:
            self.refresh()

    @validate_protocol
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

    @validate_protocol
    def size_hint(self) -> tuple[int, int]:
        return (320, 400)

    @validate_protocol
    def model_type(self) -> str:
        return self._data_wrapper._type

    def current_indices(self) -> dict[str, int]:
        return self._dims_sliders.value()

    @validate_protocol
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
