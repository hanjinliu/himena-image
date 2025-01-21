from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence
import zipfile
import impy as ip
import numpy as np
from roifile import ROI_SUBTYPE, ImagejRoi, roiread, roiwrite, ROI_TYPE

from himena import StandardType, WidgetDataModel
from himena.standards.model_meta import ImageMeta
from himena.standards import roi as _roi
from himena.plugins import register_reader_provider, register_writer_provider
from himena_image.utils import image_to_model


_SUPPORTED_EXT = frozenset(
    [".tif", ".tiff",
     ".mrc", ".rec", ".st", ".map", ".mrc.gz", ".map.gz",
     ".nd2",
     ]
)  # fmt: skip


def read_image(path: Path):
    img = ip.imread(path)
    is_rgb = "c" in img.axes and path.suffix in [".png", ".jpg", ".jpeg"]
    model = image_to_model(img, is_rgb=is_rgb)
    model.extension_default = path.suffix
    return model


def write_image(model: WidgetDataModel, path: Path):
    img = model.value
    _axes = None
    _scales = {}
    _units = {}
    if isinstance(meta := model.metadata, ImageMeta):
        if axes := meta.axes:
            _axes = [a.name for a in axes]
            _scales = {a.name: a.scale for a in axes}
            _units = {a.name: a.unit for a in axes}
    img = ip.asarray(img, axes=_axes)

    for a in img.axes:
        a.scale = _scales.get(str(a))
        a.unit = _units.get(str(a))
    img.imsave(path)


def _get_coords(ijroi: ImagejRoi) -> np.ndarray:
    if ijroi.subpixelresolution:
        return ijroi.subpixel_coordinates - 1
    return ijroi.integer_coordinates - 1 + [ijroi.left, ijroi.top]


def _to_standard_roi(ijroi: ImagejRoi) -> tuple[tuple[int, ...], _roi.RoiModel]:
    p = ijroi.position
    c = ijroi.c_position
    t = ijroi.t_position
    z = ijroi.z_position
    name = ijroi.name

    if ijroi.subtype == ROI_SUBTYPE.UNDEFINED:
        if ijroi.roitype == ROI_TYPE.RECT:
            out = _roi.RectangleRoi(
                x=ijroi.left - 1,
                y=ijroi.top - 1,
                width=ijroi.right - ijroi.left,
                height=ijroi.bottom - ijroi.top,
                name=name,
            )
        elif ijroi.roitype == ROI_TYPE.LINE:
            out = _roi.LineRoi(
                x1=ijroi.x1 - 1,
                y1=ijroi.y1 - 1,
                x2=ijroi.x2 - 1,
                y2=ijroi.y2 - 1,
                name=name,
            )
        elif ijroi.roitype == ROI_TYPE.POINT:
            coords = _get_coords(ijroi)
            if coords.shape[0] == 1:
                out = _roi.PointRoi2D(x=coords[0, 0], y=coords[0, 1], name=name)
            else:
                out = _roi.PointsRoi2D(xs=coords[:, 0], ys=coords[:, 1], name=name)
        elif ijroi.roitype in (ROI_TYPE.POLYGON, ROI_TYPE.FREEHAND):
            coords = _get_coords(ijroi)
            out = _roi.PolygonRoi(xs=coords[:, 0], ys=coords[:, 1], name=name)
        elif ijroi.roitype in (ROI_TYPE.POLYLINE, ROI_TYPE.FREELINE):
            coords = _get_coords(ijroi)
            out = _roi.SegmentedLineRoi(xs=coords[:, 0], ys=coords[:, 1], name=name)
        elif ijroi.roitype == ROI_TYPE.OVAL:
            out = _roi.EllipseRoi(
                x=ijroi.left - 1,
                y=ijroi.top - 1,
                width=ijroi.right - ijroi.left,
                height=ijroi.bottom - ijroi.top,
                name=name,
            )
        else:
            raise ValueError(f"Unsupported ROI type: {ijroi.roitype!r}")
    elif ijroi.subtype == ROI_SUBTYPE.ROTATED_RECT:
        coords = _get_coords(ijroi)
        start = (coords[0] + coords[1]) / 2
        end = (coords[2] + coords[3]) / 2
        width = np.hypot(coords[0, 0] - coords[1, 0], coords[0, 1] - coords[1, 1])
        out = _roi.RotatedRectangleRoi(
            start=tuple(start),
            end=tuple(end),
            width=width,
            name=name,
        )
    elif ijroi.subtype == ROI_SUBTYPE.ELLIPSE:
        # ImageJ rotated ellipse is just a freehand line
        coords = _get_coords(ijroi)
        out = _roi.PolygonRoi(xs=coords[:, 0], ys=coords[:, 1], name=name)
    else:
        raise ValueError(f"Unsupported ROI subtype: {ijroi.subtype}")
    return (p, t, z, c), out


def _from_standard_roi(
    roi: _roi.RoiModel,
    multi_dims: dict[str, int],
) -> ImagejRoi:
    if isinstance(roi, _roi.RectangleRoi):
        return ImagejRoi(
            roitype=ROI_TYPE.RECT,
            name=roi.name,
            top=roi.y + 1,
            left=roi.x + 1,
            bottom=roi.y + roi.height + 1,
            right=roi.x + roi.width + 1,
            **multi_dims,
        )
    elif isinstance(roi, _roi.EllipseRoi):
        return ImagejRoi(
            roitype=ROI_TYPE.OVAL,
            name=roi.name,
            top=roi.y + 1,
            left=roi.x + 1,
            bottom=roi.y + roi.height + 1,
            right=roi.x + roi.width + 1,
            **multi_dims,
        )
    elif isinstance(roi, _roi.LineRoi):
        return ImagejRoi(
            roitype=ROI_TYPE.LINE,
            name=roi.name,
            x1=roi.x1 + 1,
            y1=roi.y1 + 1,
            x2=roi.x2 + 1,
            y2=roi.y2 + 1,
            **multi_dims,
        )
    elif isinstance(roi, _roi.PointRoi2D):
        return ImagejRoi(
            roitype=ROI_TYPE.POINT,
            name=roi.name,
            subpixelresolution=True,
            subpixel_coordinates=np.array([[roi.y + 1, roi.x + 1]]),
            **multi_dims,
        )
    elif isinstance(roi, _roi.PointsRoi2D):
        return ImagejRoi(
            roitype=ROI_TYPE.POINT,
            name=roi.name,
            subpixelresolution=True,
            subpixel_coordinates=np.stack([roi.ys + 1, roi.xs + 1], axis=1),
            **multi_dims,
        )
    elif isinstance(roi, _roi.PolygonRoi):
        return ImagejRoi(
            roitype=ROI_TYPE.POLYGON,
            name=roi.name,
            subpixelresolution=True,
            subpixel_coordinates=np.stack([roi.ys + 1, roi.xs + 1], axis=1),
            **multi_dims,
        )
    elif isinstance(roi, _roi.SegmentedLineRoi):
        return ImagejRoi(
            roitype=ROI_TYPE.POLYLINE,
            name=roi.name,
            subpixelresolution=True,
            subpixel_coordinates=np.stack([roi.ys + 1, roi.xs + 1], axis=1),
            **multi_dims,
        )
    elif isinstance(roi, _roi.RotatedRectangleRoi):
        coords = np.stack(roi._get_vertices(), axis=0) + 1
        return ImagejRoi(
            roitype=ROI_TYPE.FREEHAND,
            subtype=ROI_SUBTYPE.ROTATED_RECT,
            name=roi.name,
            subpixelresolution=True,
            subpixel_coordinates=coords,
            **multi_dims,
        )
    raise ValueError(f"Unsupported ROI type: {type(roi)}")


def read_roi(path: Path):
    out = roiread(path)
    if isinstance(out, ImagejRoi):
        ijrois = [out]
    else:
        ijrois = out
    indices: list[Sequence[int]] = []
    rois = []
    for ijroi in ijrois:
        ind, sroi = _to_standard_roi(ijroi)
        indices.append(ind)
        rois.append(sroi)
    axis_names = ["p", "t", "z", "c"]
    indices = np.array(indices, dtype=np.int32)
    val = _roi.RoiListModel(rois, indices=indices, axis_names=axis_names).simplified()
    return WidgetDataModel(value=val, type=StandardType.ROIS, title=path.name)


def write_roi(path: Path, model: WidgetDataModel):
    if not isinstance(rlist := model.value, _roi.RoiListModel):
        raise ValueError(f"Must be a RoiListModel, got {type(rlist)}")
    _ij_position_getter = partial(
        _to_ij_position, rlist.indices, axis_names=rlist.axis_names
    )
    p_s = _ij_position_getter(["p", "position"])
    t_s = _ij_position_getter(["t", "time"])
    z_s = _ij_position_getter(["z", "slice"])
    c_s = _ij_position_getter(["c", "channel"])
    ijrois: list[ImagejRoi] = []
    for p, t, z, c, roi in zip(p_s, t_s, z_s, c_s, rlist.items):
        multi_dims = {
            "position": p,
            "t_position": t,
            "z_position": z,
            "c_position": c,
        }
        ijrois.append(_from_standard_roi(roi, **multi_dims))
    roiwrite(path, ijrois)
    return None


def _to_ij_position(
    indices: np.ndarray,
    candidates: list[str],
    axis_names: list[str],
) -> np.ndarray:
    for cand in candidates:
        if cand in axis_names:
            return indices[:, axis_names.index(cand)]
    return np.full(indices.shape[0], None, dtype=np.object_)


@register_reader_provider
def read_image_provider(path: Path):
    ext = "".join(path.suffixes)
    if ext in _SUPPORTED_EXT:
        return read_image
    return None


@register_writer_provider
def write_image_provider(model: WidgetDataModel, path: Path):
    ext = "".join(path.suffixes)
    if ext in _SUPPORTED_EXT:
        return write_image
    return None


@register_reader_provider
def read_roi_provider(path: Path):
    ext = "".join(path.suffixes)
    if ext == ".roi":
        return read_roi
    elif ext == ".zip":
        with zipfile.ZipFile(path) as z:
            if names := z.namelist():
                if names[0].endswith(".roi"):
                    return read_roi
    return None


@register_writer_provider
def write_roi_provider(model: WidgetDataModel, path: Path):
    ext = "".join(path.suffixes)
    if ext == ".zip":
        return write_roi
    return None
