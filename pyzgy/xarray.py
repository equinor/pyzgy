import itertools
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
from enum import Enum

import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint, BackendArray
from xarray.core import indexing
import os

from openzgy.api import ZgyReader, SampleDataType, UnitDimension

from .read import SeismicReader


@dataclass(order=True, frozen=True)
class ZgyDataTypes:

    _mapping = {
        SampleDataType.float: np.dtype(np.float32),
        SampleDataType.int16: np.dtype(np.int16),
        SampleDataType.int8: np.dtype(np.int8),
    }

    @classmethod
    def __getitem__(cls, dtype: SampleDataType) -> np.typename:
        return cls._mapping[dtype]


@dataclass(order=True, frozen=True)
class AttrKeyField:
    # match segysak if possible
    ns: str = "ns"
    sample_rate: str = "sample_rate"
    text: str = "text"
    measurement_system: str = "measurement_system"
    d3_domain: str = "d3_domain"
    epsg: str = "epsg"
    corner_points: str = "corner_points"
    corner_points_xy: str = "corner_points_xy"
    source_file: str = "source_file"
    srd: str = "srd"
    datatype: str = "datatype"
    percentiles: str = "percentiles"
    coord_scalar: str = "coord_scalar"


@dataclass(order=True, frozen=True)
class CoordKeyField:
    # match segysak if possible
    cdp_x: str = "cdp_x"
    cdp_y: str = "cdp_y"


class ZgyBackendArray(BackendArray):
    def __init__(
        self,
        shape,
        dtype,
        lock,
        zgy_file,
    ):
        self.shape = shape
        self.dtype = dtype
        self.zgy_file = zgy_file
        self.lock = lock

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        shape = []
        out_shape = []
        block_program = []

        for dim, end in zip(key, self.shape):
            dim_bp = []
            if isinstance(dim, int):
                shape.append(1)
                dim_bp.append((dim, slice(None, None, None)))
            else:
                start = 0 if dim.start is None else dim.start
                end = end if dim.stop is None else dim.stop
                step = 1 if dim.step is None else dim.step
                indexes = tuple(
                    i for i in itertools.islice(range(end), start, end, step)
                )
                shape.append(len(indexes))
                out_shape.append(shape[-1])

                if step == 1:
                    dim_bp.append((start, slice(None, None, None)))
                else:
                    for out_ind, ind in enumerate(indexes):
                        dim_bp.append((ind, slice(out_ind, out_ind + 1, None)))

            block_program.append(dim_bp)

        with ZgyReader(str(self.zgy_file)) as reader:
            volume = np.zeros(shape, dtype=self.dtype)

            # Zgy is Always 3D
            for iline, xline, vert in itertools.product(*tuple(block_program)):
                block_start = (iline[0], xline[0], vert[0])
                reader.read(block_start, volume[iline[1], xline[1], vert[1]])

        return volume.reshape(out_shape)


class PyzgyBackendEntrypoint(BackendEntrypoint):

    def _create_dimension(self, start, step, size, hunitfactor, as_int=True):
        # create actual data dimensions from meta data
        dim = hunitfactor * (start + np.arange(0, size) * step)
        if as_int:
            return dim.astype(np.int32)
        return dim

    def _get_dtype(self, filename_or_obj):
        with ZgyReader(str(filename_or_obj)) as reader:
            header = reader.meta
            return ZgyDataTypes()[header["datatype"]]

    def _create_geometry_ds(self, filename_or_obj):
        # creating dimensions and new dataset
        with SeismicReader(filename_or_obj) as reader:
            zgy_shape = reader.filehandle.size

            # create dimensions
            dims = {
                "iline": reader.ilines,
                "xline": reader.xlines,
            }
            dims_vert = {"samples": reader.samples}

            ds = xr.Dataset(coords=dims | dims_vert)

            # load attributes
            ds.attrs[AttrKeyField.source_file] = str(filename_or_obj)
            ds.attrs[AttrKeyField.text] = "PyZgy/OpenZgy"
            ds.attrs[AttrKeyField.ns] = reader.n_samples
            ds.attrs[AttrKeyField.coord_scalar] = reader.filehandle.hunitfactor

            indices = np.indices(reader.filehandle.size[:2])

            ds[CoordKeyField.cdp_x] = (
                ("iline", "xline"),
                xr.apply_ufunc(reader.gen_cdp_x, *indices),
            )
            ds[CoordKeyField.cdp_y] = (
                ("iline", "xline"),
                xr.apply_ufunc(reader.gen_cdp_y, *indices),
            )

        return zgy_shape, ds

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike,
        drop_variables: tuple[str] | None = None,
    ):
        zgy_shape, ds = self._create_geometry_ds(filename_or_obj)
        backend_array = ZgyBackendArray(
            zgy_shape,
            np.dtype(np.float32),
            None,
            filename_or_obj,
        )
        data = indexing.LazilyIndexedArray(backend_array)
        encoding = dict(
            # zgy preferred chunk size, from openzgy
            preferred_chunks = {
            "iline": 64,
            "xline": 64,
            "samples": 64
        })
        ds["data"] = xr.Variable(ds.dims, data, encoding=encoding)
        return ds

    def guess_can_open(self, filename_or_obj: str | os.PathLike):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".zgy"}
