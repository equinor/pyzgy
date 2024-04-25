import itertools
from dataclasses import dataclass
from typing import Union

import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint, BackendArray
from xarray.core import indexing
import os

from openzgy.api import ZgyReader, SampleDataType, UnitDimension

from .read import SeismicReader
from .write import SeismicWriter


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
            dims.update(dims_vert)
            ds = xr.Dataset(coords=dims)

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
        filename_or_obj: Union[str, os.PathLike],
        drop_variables: Union[tuple[str], None] = None,
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

    def guess_can_open(self, filename_or_obj: Union[str, os.PathLike]):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".zgy"}

@xr.register_dataset_accessor("pyzgy")
class PyZGY:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._corners = None

    def _check_variable_has_dims(self, data_variable, dims):
        # test for malformed dataset
        assert data_variable in self._obj
        assert len(self._obj[data_variable].dims) == len(dims)
        for dim in dims:
            assert dim in self._obj.dims
            assert dim in self._obj[data_variable].dims

    def _check_regular_dim(self, dim):
        # check that the dimensions increment regularly, ZGY doesn't support
        # irregular sampling
        assert np.ptp(self._obj[dim].diff(dim).values) == 0.0

    def corners(self):
        """Get the corner coordinates of the dataset from variables `cdp_x` and
        `cdp_y`.

        The order of the list is corners or per index of figure.
            First inline / first crossline,
            last inline / first crossline,
            first inline / last crossline,
            last inline / last crossline.

        (1, iln, xl1)        (3, iln, xln)
            ^
            |
            |
            |
            |
            |
            |______________________>
        (0, il1, xl1)          (2, il1, xln)

        Returns:
        --------
        corners: Tuple[Tuple[2]][4]
        """
        for coord in (CoordKeyField.cdp_x, CoordKeyField.cdp_y):
            try:
                self._check_variable_has_dims(coord, ("iline", "xline"))
            except AssertionError as err:
                err.args += ("Malformed coordinates variables", coord, )
                raise

        # create corners
        il0, iln = self._obj["iline"][0].item(), self._obj["iline"][-1].item()
        xl0, xln = self._obj["xline"][0].item(), self._obj["xline"][-1].item()
        corner_list = ((il0, xl0), (iln, xl0), (il0, xln), (iln, xln))
        return tuple(
            (
                self._obj[CoordKeyField.cdp_x].sel(iline=il, xline=xl).values.item(),
                self._obj[CoordKeyField.cdp_y].sel(iline=il, xline=xl).values.item()
            )
            for (il, xl) in corner_list
        )

    def increment(self, dim):
        """Calculate the increment of a dimension

        Parameters
        ----------
        dim: str
            The dimension to return the increment on.
        """
        self._check_regular_dim(dim)
        # check for length 1 dimensions
        if self._obj[dim].size > 2:
            inc = self._obj[dim][1] - self._obj[dim][0]
        else:
            inc = 0
        return inc

    def to_zgy(self, filename, data_variable="data"):
        """Output single variable to ZGY file.

        Expects dimension names of `iline`, `xline` and `samples` for a 3D
        volume to export. If `cdp_x` and `cdp_y` variables are available, they
        will be used to determine the corner points of the ZGY volume.

        If modifying the size of your dataset chunks, use multiples of 64. This is
        the preferred block/brick size for ZGY.

        Parameters
        ----------
        filename: str
            The filename and path to write to.

        data_variable: str
            The name of the data volume variable to export, defaults to "data".
        """
        # test for malformed dataset
        assert data_variable in self._obj
        assert len(self._obj[data_variable].dims) == 3
        for dim in ("iline", "xline", "samples"):
            assert dim in self._obj.dims
            assert dim in self._obj[data_variable].dims

        zgy_dv = self._obj[data_variable].transpose("iline", "xline", "samples")
        shape = zgy_dv.shape

        # generate corners from cdp variables if they exist
        if (
            CoordKeyField.cdp_x in self._obj
            and CoordKeyField.cdp_y in self._obj
            ):
            self._corners = self.corners()

        # assumes that dimensions are not length 1
        zinc = self.increment("samples")
        annotinc = (self.increment("iline"), self.increment("xline"))

        with SeismicWriter(
            filename,
            shape,
            self._obj["samples"][0].item(),
            zinc,
            (self._obj["iline"][0].item(), self._obj["xline"][0].item()),
            annotinc,
            corners=self._corners
        ) as writer:
            if zgy_dv.chunks:
                il_cx, xl_cx, s_cx = zgy_dv.chunks
                for i, j, k in itertools.product(
                    range(0, len(il_cx)), # iterate iline
                    range(0, len(xl_cx)), # iterate xline
                    range(0, len(s_cx)), # iterate samples
                    ):
                    istart = sum(il_cx[:i])
                    jstart = sum(xl_cx[:j])
                    kstart = sum(s_cx[:k])
                    irange = np.arange(istart, istart + il_cx[i])
                    jrange = np.arange(jstart, jstart + xl_cx[j])
                    krange = np.arange(kstart, kstart + s_cx[k])
                    writer.write_subvolume(
                        zgy_dv.isel(
                            iline=irange,
                            xline=jrange,
                            samples=krange
                        ).values, istart, jstart, kstart
                    )
            else:
                writer.write_subvolume(
                    zgy_dv.values,
                    0, 0, 0
                )
