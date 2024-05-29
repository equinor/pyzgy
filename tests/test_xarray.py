import pathlib
import numpy as np
import xarray as xr
import pytest
import segyio

from openzgy.api import ZgyReader


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


def test_xarray_data(zgy_sgy_file_pairs):
    ZGY_FILE, SGY_FILE = zgy_sgy_file_pairs
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


def test_preferred_chunks(zgy_sgy_file_pairs):
    ZGY_FILE, SGY_FILE = zgy_sgy_file_pairs
    ds = xr.open_dataset(ZGY_FILE, chunks={})

    # small files reads all data
    assert ds.data.chunks == ((5,), (5,), (50,))


def test_xarray_corners(zgy_sgy_file_pairs):
    ZGY_FILE, SGY_FILE = zgy_sgy_file_pairs
    ds = xr.open_dataset(ZGY_FILE, chunks={})

    ds.pyzgy.corners()

    with ZgyReader(ZGY_FILE) as reader:
        assert ds.pyzgy.corners() == reader.corners

    ds = ds.drop_vars(("cdp_x", "cdp_y"))
    with pytest.raises(AssertionError):
        ds.pyzgy.corners()


def test_xarray_to_zgy(zgy_sgy_file_pairs, temp_dir):
    ZGY_FILE, SGY_FILE = zgy_sgy_file_pairs
    ds = xr.open_dataset(ZGY_FILE, chunks={})

    ZGY_FILE_OUT = pathlib.Path(ZGY_FILE).name

    ds.pyzgy.to_zgy(temp_dir / ZGY_FILE_OUT)

    ds_out = xr.open_dataset(str(temp_dir / ZGY_FILE_OUT))

    assert np.allclose(ds.iline, ds_out.iline)
    assert np.allclose(ds.xline, ds_out.xline)
    assert np.allclose(ds.data, ds_out.data)
    assert np.allclose(ds.cdp_x, ds_out.cdp_x)
    assert np.allclose(ds.cdp_y, ds_out.cdp_y)


@pytest.mark.parametrize("dims,fname", (
        ((50, 50, 50), "sm"),
        ((50, 50, 1000), "med"),
        ((200, 200, 1000), "lrg")
))
def test_xarray_to_zgy_chunking(dims, fname, temp_dir):

    ni, nx, ns = dims
    ZGY_OUT_FILE = temp_dir / f"zgy_chunked_{fname}.zgy"

    ilines = np.arange(ni, dtype=int)
    xlines = np.arange(nx, dtype=int)
    samples = np.arange(ns, dtype=int)

    cube = xr.Dataset(
        coords={
            "iline": (["iline"], ilines),
            "xline": (["xline"], xlines),
            "samples": (["samples"], samples),
        },
        data_vars={"data": (("iline", "xline", "samples"), np.random.random(dims))},
    ).chunk({"iline": 64, "xline": 64, "samples": 64})
    cube.pyzgy.to_zgy(str(ZGY_OUT_FILE))
    assert ZGY_OUT_FILE.exists()

    # read back data
    ds_out = xr.open_dataset(str(ZGY_OUT_FILE))
    assert np.allclose(ds_out.iline, ilines)
    assert np.allclose(ds_out.xline, xlines)
    assert np.allclose(ds_out.samples, samples)
    assert np.allclose(ds_out.data, cube.data)
