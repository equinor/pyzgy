import numpy as np
import xarray as xr
import pytest
import segyio

from pyzgy.read import SeismicReader
from openzgy.exception import ZgyUserError


ZGY_SGY_FILE_PAIRS = [
    ("test_data/small-{}bit.zgy".format(n), "test_data/small-{}bit.sgy".format(n))
    for n in [32, 16, 8]
]


@pytest.mark.parametrize(
    "zgyfile_pairs",
    ZGY_SGY_FILE_PAIRS,
)
def test_xarray_geometry(zgyfile_pairs):
    ZGY_FILE, SGY_FILE = zgyfile_pairs
    ds = xr.open_dataset(ZGY_FILE)

    with segyio.open(SGY_FILE) as sgyfile:
        # check labelling
        assert np.all(ds.iline == sgyfile.ilines)
        assert np.all(ds.xline == sgyfile.xlines)
        assert np.all(ds.samples == sgyfile.samples)

        # check dtypes
        assert ds.iline.dtype == sgyfile.ilines.dtype
        assert ds.xline.dtype == sgyfile.xlines.dtype
        assert ds.samples.dtype == sgyfile.samples.dtype


@pytest.mark.parametrize(
    "zgyfile_pairs",
    ZGY_SGY_FILE_PAIRS,
)
def test_xarray_data(zgyfile_pairs):
    ZGY_FILE, SGY_FILE = zgyfile_pairs
    ds = xr.open_dataset(ZGY_FILE)

    with segyio.open(SGY_FILE) as sgyfile:
        # check ilines
        for line_number in ds.iline:
            slice_zgy = ds.sel(iline=line_number).data
            slice_segy = sgyfile.iline[int(line_number.data)]
            assert np.allclose(slice_zgy.data, slice_segy, rtol=1e-5)

    with segyio.open(SGY_FILE) as sgyfile:
        # check xlines
        for line_number in ds.xline:
            slice_zgy = ds.sel(xline=line_number).data
            slice_segy = sgyfile.xline[int(line_number.data)]
            assert np.allclose(slice_zgy.data, slice_segy, rtol=1e-5)


@pytest.mark.parametrize(
    "zgyfile_pairs",
    ZGY_SGY_FILE_PAIRS,
)
def test_preferred_chunks(zgyfile_pairs):
    ZGY_FILE, SGY_FILE = zgyfile_pairs
    ds = xr.open_dataset(ZGY_FILE, chunks={})

    # small files reads all data
    assert ds.data.chunks == ((5, ), (5, ), (50, ))
