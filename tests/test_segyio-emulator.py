import numpy as np
import segyio
import pyzgy

ZGY_FILE = 'test_data/small-32bit.zgy'
SGY_FILE = 'test_data/small-32bit.sgy'


def compare_inline_ordinal(zgy_filename, sgy_filename, lines_to_test, tolerance):
    with pyzgy.open(zgy_filename) as zgyfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_ordinal in lines_to_test:
                slice_segy = segyfile.iline[segyfile.ilines[line_ordinal]]
                slice_zgy = zgyfile.iline[zgyfile.ilines[line_ordinal]]
                assert np.allclose(slice_zgy, slice_segy, rtol=tolerance)

def compare_inline_number(zgy_filename, sgy_filename, lines_to_test, tolerance):
    with pyzgy.open(zgy_filename) as zgyfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_number in lines_to_test:
                slice_segy = segyfile.iline[line_number]
                slice_zgy = zgyfile.iline[line_number]
                assert np.allclose(slice_zgy, slice_segy, rtol=tolerance)

def compare_inline_slicing(zgy_filename):
    slices = [slice(1, 5, 2), slice(1, 2, None), slice(1, 3, None), slice(None, 3, None), slice(3, None, None)]
    with pyzgy.open(zgy_filename) as zgyfile:
        for slice_ in slices:
            slices_slice = np.asarray(zgyfile.iline[slice_])
            start = slice_.start if slice_.start is not None else 1
            stop = slice_.stop if slice_.stop is not None else 6
            step = slice_.step if slice_.step is not None else 1
            slices_concat = np.asarray([zgyfile.iline[i] for i in range(start, stop, step)])
            assert np.array_equal(slices_slice, slices_concat)

def test_inline_accessor():
    compare_inline_ordinal(ZGY_FILE, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-5)
    compare_inline_number(ZGY_FILE, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-5)
    compare_inline_slicing(ZGY_FILE)


def compare_crossline_ordinal(zgy_filename, sgy_filename, lines_to_test, tolerance):
    with pyzgy.open(zgy_filename) as zgyfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_ordinal in lines_to_test:
                slice_segy = segyfile.xline[segyfile.xlines[line_ordinal]]
                slice_zgy = zgyfile.xline[zgyfile.xlines[line_ordinal]]
                assert np.allclose(slice_zgy, slice_segy, rtol=tolerance)

def compare_crossline_number(zgy_filename, sgy_filename, lines_to_test, tolerance):
    with pyzgy.open(zgy_filename) as zgyfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_number in lines_to_test:
                slice_segy = segyfile.xline[line_number]
                slice_zgy = zgyfile.xline[line_number]
                assert np.allclose(slice_zgy, slice_segy, rtol=tolerance)

def compare_crossline_slicing(zgy_filename):
    slices = [slice(20, 21, 2), slice(21, 23, 1), slice(None, 22, None), slice(22, None, None)]
    with pyzgy.open(zgy_filename) as zgyfile:
        for slice_ in slices:
            slices_slice = np.asarray(zgyfile.xline[slice_])
            start = slice_.start if slice_.start is not None else 20
            stop = slice_.stop if slice_.stop is not None else 25
            step = slice_.step if slice_.step is not None else 1
            slices_concat = np.asarray([zgyfile.xline[i] for i in range(start, stop, step)])
            assert np.array_equal(slices_slice, slices_concat)

def test_crossline_accessor():
    compare_crossline_ordinal(ZGY_FILE, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-5)
    compare_crossline_number(ZGY_FILE, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-5)
    compare_crossline_slicing(ZGY_FILE)


def compare_zslice(zgy_filename, tolerance):
    with pyzgy.open(zgy_filename) as zgyfile:
        with segyio.open(SGY_FILE) as segyfile:
            for line_number in range(50):
                slice_zgy = zgyfile.depth_slice[line_number]
                slice_segy = segyfile.depth_slice[line_number]
                assert np.allclose(slice_zgy, slice_segy, rtol=tolerance)


def test_zslice_accessor():
    compare_zslice(ZGY_FILE, tolerance=1e-5)


def test_trace_accessor():
    with pyzgy.open(ZGY_FILE) as zgyfile:
        with segyio.open(SGY_FILE) as segyfile:
            for trace_number in range(-5, 25, 1):
                zgy_trace = zgyfile.trace[trace_number]
                segy_trace = segyfile.trace[trace_number]
                assert np.allclose(zgy_trace, segy_trace, rtol=1e-5)


def test_read_trace_header():
    with pyzgy.open(ZGY_FILE) as zgyfile:
        with segyio.open(SGY_FILE) as sgyfile:
            for trace_number in range(-5, 25, 1):
                zgy_header = zgyfile.header[trace_number]
                sgy_header = sgyfile.header[trace_number]
                assert zgy_header[181] == sgy_header[181]
                assert zgy_header[185] == sgy_header[185]
                assert zgy_header[189] == sgy_header[189]
                assert zgy_header[193] == sgy_header[193]


def compare_cube(zgy_filename, sgy_filename, tolerance):
    vol_sgy = segyio.tools.cube(sgy_filename)
    vol_zgy = pyzgy.tools.cube(zgy_filename)
    assert np.allclose(vol_zgy, vol_sgy, rtol=tolerance)

def test_cube_func():
    compare_cube(ZGY_FILE, SGY_FILE, tolerance=1e-5)
