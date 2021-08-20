#!/usr/bin/env python3

"""
White-box unit tests for OpenZGY.
"""

import numpy as np
import os
import io
import sys
import math

from ..impl import bulk as impl_bulk
from ..impl import meta as impl_meta
from ..impl import file as impl_file
from ..impl import stats as impl_stats
from ..impl import enum as impl_enum
from ..impl import histogram as impl_histogram
from .. import api as openzgy
from ..exception import *
from ..test.utils import SDCredentials, LocalFileAutoDelete

def test_ScaleToStorage():
    """
    Unit test for ZgyInternalBulk._scaleToStorage.
    """
    fn = impl_bulk.ZgyInternalBulk._scaleToStorage
    codingrange = (-10.0, +30.0)

    # Converting the min- and max of the coding range should yield
    # the first and last valid storage number.
    assert fn(np.float32(-10.0), codingrange, np.int8) == -128
    assert fn(np.float32(+30.0), codingrange, np.int8) == 127
    assert fn(np.float32(-10.0), codingrange, np.uint16) == 0
    assert fn(np.float32(+30.0), codingrange, np.uint16) == 65535

    # Outside valid range should silently clip.
    assert fn(np.float32(-15.0), codingrange, np.int8) == -128
    assert fn(np.float32(+35.0), codingrange, np.int8) == 127

    # Conversion should round to nearest. For this coding range
    # each slot covers a bit more than +/- 0.0783
    assert fn(np.float32(-10.0 + 0.0783), codingrange, np.int8) == -128
    assert fn(np.float32(-10.0 + 0.0786), codingrange, np.int8) == -127
    assert fn(np.float32(-10.0 + 3*0.0783), codingrange, np.int8) == -127
    assert fn(np.float32(-10.0 + 3*0.0786), codingrange, np.int8) == -126

def test_LookupTables():
    """
    Unit test for the following related functions:
        _get{Brick,Alpha}FilePosition()
        _get{Brick,Alpha}LookupIndex()
        _validatePosition()
            lod out of range
            ijk out of range for lod 0
            ijk out of range for lod N, would not have been if lod == 0

    Calls these functions to get mocked data; these are assumed tested
    elsewhere. Or that they are so trivial that they don't need testing.
    """

    #test all-constant brick (both types) and missing brick
    # Arrange

    class MockInternalMeta: pass
    class MockInfoHeader: pass

    mock = MockInternalMeta()
    mock._ih = MockInfoHeader()
    mock._ih._size = (112, 64, 176) # bricks: (2, 1, 3)
    mock._ih._bricksize = (64, 64, 64)
    mock._ih._lodsizes = impl_meta._CalcLodSizes(mock._ih._size, mock._ih._bricksize)
    mock._ih._nlods = len(mock._ih._lodsizes)
    mock._ih._brickoffsets = impl_meta._CalcLutOffsets(mock._ih._lodsizes, False)
    mock._ih._alphaoffsets = impl_meta._CalcLutOffsets(mock._ih._lodsizes, True)
    mock._blup = [0] * mock._ih._brickoffsets[mock._ih._nlods]
    mock._alup = [0] * mock._ih._alphaoffsets[mock._ih._nlods]

    assert mock._ih._nlods == 3
    assert type(mock._ih._lodsizes) == tuple
    assert type(mock._ih._lodsizes[0]) == tuple
    assert tuple(mock._ih._lodsizes[0]) == (2, 1, 3)
    assert tuple(mock._ih._lodsizes[1]) == (1, 1, 2)
    assert tuple(mock._ih._lodsizes[2]) == (1, 1, 1)

def test_throws(e, f):
    try:
        f(None)
        return False
    except e:
        return True

def test_MapEnums():
    """
    Unit test for conversion between various internal and api enums.
    """
    from ..impl import enum as internal

    # (internal) RawDataType to (public) SampleDataType
    # Errors returned as unknown because the code should
    # be tolerant with regards to garbage in the file.
    assert openzgy._map_DataTypeToSampleDataType(internal.RawDataType.SignedInt8) == openzgy.SampleDataType.int8
    assert openzgy._map_DataTypeToSampleDataType(internal.RawDataType.SignedInt16) == openzgy.SampleDataType.int16
    assert openzgy._map_DataTypeToSampleDataType(internal.RawDataType.Float32) == openzgy.SampleDataType.float
    # Maps to unknown because I have not added this one to the enum
    assert openzgy._map_DataTypeToSampleDataType(internal.RawDataType.UnsignedInt8) == openzgy.SampleDataType.unknown

    # (public) SampleDataType to (internal) RawDataType
    # Errors should raise exceptions.
    assert openzgy._map_SampleDataTypeToDataType(openzgy.SampleDataType.int8) == internal.RawDataType.SignedInt8
    assert openzgy._map_SampleDataTypeToDataType(openzgy.SampleDataType.int16) == internal.RawDataType.SignedInt16
    assert openzgy._map_SampleDataTypeToDataType(openzgy.SampleDataType.float) == internal.RawDataType.Float32

    # (public) UnitDimension to (internal) Horizontal- and Vertical dim
    # Errors should raise exceptions.
    assert openzgy._map_UnitDimensionToHorizontalDimension(openzgy.UnitDimension.length) == internal.RawHorizontalDimension.Length
    assert openzgy._map_UnitDimensionToHorizontalDimension(openzgy.UnitDimension.arcangle) == internal.RawHorizontalDimension.ArcAngle
    # time is not allowed horizontally.
    assert test_throws(openzgy.ZgyUserError, lambda x: openzgy._map_UnitDimensionToHorizontalDimension(openzgy.UnitDimension.time))
    assert openzgy._map_UnitDimensionToVerticalDimension(openzgy.UnitDimension.length) == internal.RawVerticalDimension.Depth
    assert openzgy._map_UnitDimensionToVerticalDimension(openzgy.UnitDimension.time) == internal.RawVerticalDimension.SeismicTWT
    # arcangle not allowed vertically.
    assert test_throws(openzgy.ZgyUserError, lambda x: openzgy._map_UnitDimensionToVerticalDimension(openzgy.UnitDimension.arcangle))

    # (internal) Horizontal- and Vertical dim to (public) UnitDimension
    # Errors should return unknown, but if there was a bad tag it should
    # have been mapped to unknown already.
    assert openzgy._map_VerticalDimensionToUnitDimension(internal.RawVerticalDimension.SeismicTWT) == openzgy.UnitDimension.time
    assert openzgy._map_VerticalDimensionToUnitDimension(internal.RawVerticalDimension.SeismicOWT) == openzgy.UnitDimension.time
    assert openzgy._map_VerticalDimensionToUnitDimension(internal.RawVerticalDimension.Depth) == openzgy.UnitDimension.length
    assert openzgy._map_VerticalDimensionToUnitDimension(internal.RawVerticalDimension.Unknown) == openzgy.UnitDimension.unknown

    assert openzgy._map_HorizontalDimensionToUnitDimension(internal.RawHorizontalDimension.Length) == openzgy.UnitDimension.length
    assert openzgy._map_HorizontalDimensionToUnitDimension(internal.RawHorizontalDimension.ArcAngle) == openzgy.UnitDimension.arcangle
    assert openzgy._map_HorizontalDimensionToUnitDimension(internal.RawHorizontalDimension.Unknown) == openzgy.UnitDimension.unknown

    del internal


def test_FormatDescription():
    for verbose in (0, 1, 2):
        with io.StringIO() as stream:
            impl_meta.checkAllFormats(verbose=verbose, file=stream)
            # The part of checkAllFormats that would dump the definitions
            # in C++ or HTML have been moved to openzgy/tools/cppmeta.py.
            # The checks and now not expected to output anything.
            if verbose == 0: assert len(stream.getvalue()) == 0
            if verbose >  0: assert len(stream.getvalue()) == 0
            # No test on the result, only see that it doesn't crash.
            #print("FORMATS TYPE", verbose, stream.getvalue(), "@END")

def test_Statistics():
    stats = impl_stats.StatisticData()
    # adding scalars
    # -NOT. StatisticData() changed to only accept numpy arrays.
    stats += np.array(42)
    stats += np.array(2)
    stats += np.array(np.inf)
    stats += np.array(-np.inf)
    stats += np.array(-np.nan)
    #print(repr(stats))
    assert stats._cnt == 2
    assert stats._inf == 3
    assert stats._sum == 42 + 2
    assert stats._ssq == 42*42 + 2*2
    assert stats._min == 2
    assert stats._max == 42

    # adding arrays
    a = np.array([42, 2, np.inf, -np.inf, np.nan])
    stats2 = impl_stats.StatisticData()
    stats2 += a
    assert stats == stats2

    # Multiply by integral factor
    stats *= 3
    stats2 += a
    stats2 += a
    assert stats == stats2

    # Add two instances
    stats = stats + 3 * (impl_stats.StatisticData() + a)
    stats2 = stats2 + stats2

    # Scaling
    a = np.array([1, 0, 2, 3, 42], dtype=np.int8)
    stats = impl_stats.StatisticData()
    stats2 = impl_stats.StatisticData()
    stats += a
    stats2 += np.array([3.14*x + 2.72 for x in a])
    # old signature: stats.scale(0, 1, 2.72, 3.14+2.72)
    stats.scale(3.14, 2.72)
    #print(repr(stats))
    #print(repr(stats2))
    assert stats._cnt == stats2._cnt
    assert stats._inf == stats2._inf
    assert math.isclose(stats._sum, stats2._sum)
    assert math.isclose(stats._ssq, stats2._ssq)
    assert math.isclose(stats._min, stats2._min)
    assert math.isclose(stats._max, stats2._max)


def alltiles(reader):
    for lod in range(reader.nlods):
        size = reader.brickcount[lod]
        for ii in range(size[0]):
            for jj in range(size[1]):
                yield ii, jj, lod

def allbricks(reader):
    for lod in range(reader.nlods):
        size = reader.brickcount[lod]
        for ii in range(size[0]):
            for jj in range(size[1]):
                for kk in range(size[2]):
                    yield ii, jj, kk, lod

def _typename(t):
    return t.__name__ if t.__module__ is None or t.__module__ == type.__module__ else t.__module__ + "." + t.__name__

def test_bricksize(filename):
    with openzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        if False:
            print("Opened file size", reader.size, "bricks", reader.brickcount)
            print(list(allbricks(reader)))
        for ii, jj, kk, lod in allbricks(reader):
            ix = reader._accessor._getBrickLookupIndex(ii, jj, kk, lod)
            raw_beg = reader._accessor._metadata._blup._lookup[ix]
            raw_end = reader._accessor._metadata._blup._lookend[ix]
            if False:
                print("begin: {0}({1}), end: {2}({3})".format(
                      _typename(type(raw_beg)), hex(raw_beg),
                      _typename(type(raw_end)), hex(raw_end)))
            beg, size = reader._accessor._getBrickBegAndSize(ix, sane = False)
            if False:
                print("ix {ix} lod {lod} brickpos ({ii},{jj},{kk}) beg 0x{beg:x} size 0x{size:x}".format(
                    ix=ix, lod=lod, ii=ii, jj=jj, kk=kk, beg=beg, size=size))
            # Size will be > 64^3 sometimes due to interleaved alpha tiles
            # and it will be very large for the last brick since we skipped
            # the test for EOF. For Salt2-v3.zgy, lowres alpha tiles exist
            # between 0x440000 and 0x480000 so the brick at 0x400000 will
            # appear twice as large as it ought to be.
            assert beg == 0 or size >= 64*64*64

        for ii, jj, lod in alltiles(reader):
            ix = reader._accessor._getAlphaLookupIndex(ii, jj, lod)
            raw_beg = reader._accessor._metadata._alup._lookup[ix]
            raw_end = reader._accessor._metadata._alup._lookend[ix]
            if False:
                print("begin: {0}({1}), end: {2}({3})".format(
                      _typename(type(raw_beg)), hex(raw_beg),
                      _typename(type(raw_end)), hex(raw_end)))
            beg, size = reader._accessor._getAlphaBegAndSize(ix, sane = False)
            if False:
                print("ix {ix} lod {lod} alphapos ({ii},{jj}) beg 0x{beg:x} size 0x{size:x}".format(
                    ix=ix, lod=lod, ii=ii, jj=jj, beg=beg, size=size))
            assert beg == 0 or size >= 64*64

def test_PaddingOutsideSurvey(filename):
    with openzgy.ZgyWriter(filename,
                           iocontext = SDCredentials(),
                           size = (7, 13, 17),
                           datatype = openzgy.SampleDataType.int16,
                           datarange = (-32768,+32767)) as writer:
        data = np.arange(1, 7*13*17+1, dtype=np.int16).reshape((7,13,17))
        data = np.pad(data, ((0,64-7),(0,64-13),(0,64-17)),
                             mode='constant', constant_values=-999)
        writer.write((0,0,0), data)
        (brickstatus, fileoffset, constvalue, bricksize) = (
            writer._accessor._getBrickFilePosition(0, 0, 0, 0))

    # With ZgyReader the padding area should be filled with a default value,
    # which is zero in this simple case where there is no scale/offset.
    # So the only nonzero data should be our real samples.
    with openzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        readback = np.full((64, 64, 64), 42, dtype=np.int16)
        reader.read((0,0,0), readback)
        assert np.count_nonzero(readback) == 7 * 13 * 17

    # Reading using basic file I/O to get the data that the compressor saw.
    with open(filename, "rb") as f:
        f.seek(fileoffset, 0)
        check = f.read(64*64*64*2)
        check = np.frombuffer(check, dtype=np.int16).reshape((64, 64, 64))

    samples_in_survey = 7 * 13 * 17
    expect_nonzero_samples = 8 * 16 * 20
    actual_nonzero_samples = np.count_nonzero(check)

    assert np.all(data[:7,:13,:17] == check[:7,:13,:17])
    assert brickstatus == impl_enum.BrickStatus.Normal
    assert fileoffset == 64*64*64*2 # Not really a problem if it fails
    assert actual_nonzero_samples == expect_nonzero_samples
    assert check[7,12,16] == data[6,12,16] # one past lower corner
    assert check[6,13,16] == data[6,12,16]
    assert check[6,12,17] == data[6,12,16]
    assert np.all(check[7,:13,:17] == check[6,:13,:17])
    assert np.all(check[:7,13,:17] == check[:7,12,:17])
    assert np.all(check[:7,14,:17] == check[:7,12,:17])
    assert np.all(check[:7,15,:17] == check[:7,12,:17])
    assert np.all(check[:7,:13,17] == check[:7,:13,16])
    assert np.all(check[:7,:13,18] == check[:7,:13,16])
    assert np.all(check[:7,:13,19] == check[:7,:13,16])

def test_histogram16():
    hist = impl_histogram.HistogramData(dtype=np.int16)
    assert hist.vv_range == (-32768, +32767)
    if hist.bins.shape[0] != 65536:
        # I might not have enabled the feature of using oversize histograms.
        # But I can fake it by resizing the new, empty histogram.
        hist.resize(65536)
        assert hist.vv_range == (-32768, +32767)
        assert hist.bins.shape[0] == 65536
    # Using only 4096 of the possible 65536 values
    hist.add(1000+np.arange(4096, dtype=np.int16))
    assert np.all(hist.bins[32768+1000:32768+1000+4096] == 1)
    # The first number, 1000, ended up in bin 32768+1000.
    assert np.isclose(1000, hist.binvalue(32768+1000))
    hist.resize(256)
    # Current implementation reserves slot 0 for zero-centric
    # adjustment which is not implemented yet. The factor will
    # end up as 17, so we will have 4096/17 = 240 bins with count
    # 17 and the 16 left over in the bin immediately after those.
    assert len(hist.bins) == 256
    assert hist.bins[0] == 0
    assert np.all(hist.bins[1:241] == 17)
    assert hist.bins[241] == 16
    assert np.all(hist.bins[242:] == 0)
    #print(hist._hmin, hist._hmax, hist._bins, sep="\n")
    # The first number, 1000, should have moved to bin 1.
    assert np.isclose(1000, hist.binvalue(1))

def test_histogram8():
    hist = impl_histogram.HistogramData(dtype=np.int8)
    assert hist.vv_range == (-128, +127)
    # Using only 90 of the possible 256 values
    hist.add(10+np.arange(80, dtype=np.int8))
    assert np.all(hist.bins[128+10:128+10+80] == 1)
    # The first number, 10, ended up in bin 128+10.
    assert np.isclose(10, hist.binvalue(128+10))
    hist.resize(32)
    # Current implementation reserves slot 0 for zero-centric
    # adjustment which is not implemented yet. The factor will
    # end up as 3, so we will have 80/3 = 26 bins with count
    # 3 and the 2 left over in the bin immediately after those.
    assert len(hist.bins) == 32
    assert hist.bins[0] == 0
    assert np.all(hist.bins[1:1+26] == 3)
    assert hist.bins[27] == 2
    assert np.all(hist.bins[28:] == 0)
    #print(hist._hmin, hist._hmax, hist._bins, sep="\n")
    # The first number, 1000, should have moved to bin 1.
    assert np.isclose(10, hist.binvalue(1))
    # Now resize it larger. All the bins end up clumped together
    # in the lower past; the resize will simply add more empty
    # bins at the end.
    hist.resize(1000)
    assert len(hist.bins) == 1000
    assert hist.bins[0] == 0
    assert np.all(hist.bins[1:1+26] == 3)
    assert hist.bins[27] == 2
    assert np.all(hist.bins[28:] == 0)
    #print(hist._hmin, hist._hmax, hist._bins, sep="\n")
    # The first number, 1000, should have moved to bin 1.
    assert np.isclose(10, hist.binvalue(1))

if __name__ == "__main__":
    np.seterr(all='raise')
    test_histogram16()
    test_histogram8()
    test_ScaleToStorage()
    test_LookupTables()
    test_MapEnums()
    test_FormatDescription()
    test_Statistics()
    # TODO-High re-enable, use test data reachable from the cloud.
    #test_bricksize("/home/paal/git/Salmon/UnitTestData/Salmon/UnitTest/Salt2-v3.zgy")
    #test_bricksize("/home/paal/git/Salmon/UnitTestData/Salmon/UnitTest/Salt2-32.zgy")
    with LocalFileAutoDelete("padding.zgy") as fn:
        test_PaddingOutsideSurvey(fn.name)

# Copyright 2017-2020, Schlumberger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
