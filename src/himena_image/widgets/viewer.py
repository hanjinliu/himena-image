from __future__ import annotations

from typing import Any, TYPE_CHECKING, Hashable, Mapping, Sequence
from enum import Enum, auto
from cmap import Colormap
from qtpy import QtWidgets as QtW, QtCore
import numpy as np

from ndv import DataWrapper, ArrayViewer
from superqt import QEnumComboBox
from himena.types import WidgetDataModel
from himena.standards.model_meta import ImageMeta, ArrayAxis
from himena.plugins import validate_protocol

if TYPE_CHECKING:
    from ndv.views._qt._array_view import _QArrayViewer


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

    @property
    def dims(self) -> tuple[str, ...]:
        if axes := self._meta.axes:
            return tuple(a.name for a in axes)
        return tuple(f"axis_{i}" for i in range(len(self._data.shape)))

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        return {d: range(s) for d, s in zip(self.dims, self.data.shape)}

    def isel(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        """Select a slice from a data store using (possibly) named indices."""
        import dask.array as da

        sl = [slice(None)] * len(self._data.shape)
        for k, v in indexers.items():
            sl[k] = v
        out = self._data[tuple(sl)]
        if isinstance(out, da.Array):
            out = out.compute()
        assert isinstance(out, np.ndarray)
        if out.dtype.kind == "b":
            return out.astype(np.uint8)
        elif out.dtype.kind == "c":
            return self._complex_conversion.apply(out)
        return out

    def sizes(self):
        if axes := self._meta.axes:
            names = [a.name for a in axes]
        else:
            names = list(range(len(self._data.shape)))
        return dict(zip(names, self._data.shape))


class NDImageViewer(ArrayViewer):
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

    @validate_protocol
    def update_model(self, model: WidgetDataModel):
        self.data = model
        is_complex = model.value.dtype.kind == "c"
        self._complex_conversion_rule_cbox.setVisible(is_complex)
        meta = model.metadata
        if isinstance(meta, ImageMeta):
            if meta.is_rgb:
                raise ValueError("RGB images are not supported yet")
            for ch, lut in zip(meta.channels, self.display_model.luts.values()):
                lut.cmap = ch.colormap or "gray"
                lut.clims = ch.contrast_limits
        if is_complex:
            self._complex_conversion_rule_cbox.setCurrentEnum(ComplexConversionRule.ABS)

    @validate_protocol
    def to_model(self) -> WidgetDataModel:
        indices = [
            None if isinstance(v, slice) else v
            for v in self.display_model.current_index.values()
        ]
        self.display_model.luts
        return WidgetDataModel(
            value=self.data,
            type=self.model_type(),
            metadata=ImageMeta(
                current_indices=indices,
                axes=[ArrayAxis(name=a) for a in self.data_wrapper.dims],
            ),
        )

    @validate_protocol
    def size_hint(self) -> tuple[int, int]:
        return (320, 400)

    @validate_protocol
    def model_type(self) -> str:
        return self.data_wrapper._type

    @validate_protocol
    def native_widget(self) -> _QArrayViewer:
        return self.widget()

    @validate_protocol
    def control_widget(self):
        return self._control_widget

    def _on_complex_conversion_rule_changed(self, enum_: ComplexConversionRule):
        self.data_wrapper._complex_conversion = enum_
        if enum_ is ComplexConversionRule.PHASE:
            cmap_name = "cmocean:phase"
        else:
            cmap_name = "inferno"
        # for ctrl in self._lut_ctrls.values():
        #     ctrl._cmap.setCurrentColormap(cmap.Colormap(cmap_name))
        # self.refresh()
        for val in self.display_model.luts.values():
            val.cmap = Colormap(cmap_name)
