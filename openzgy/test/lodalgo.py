#!/usr/bin/env python3

"""
Test the low level decimation algorithms.
"""

import numpy as np
from enum import Enum
from ..impl.lodalgo import decimate, decimate8, DecimationType
from ..impl.histogram import HistogramData

def _make8(brick, special = None):
    """
    Expand the input brick to double size, with each sample in the input
    ending up in a 2x2x2 neighborhod with similar values (in the range
    sample_in..saple_in+6m average sample_in+3. Then chop the result into
    8 bricks having the same size as the original.
    Note, if using int8 then make sure the input values are at most 121.
    """
    half = brick.shape
    full = (half[0]*2, half[1]*2, half[2]*2)
    tmp = np.zeros(full, dtype=brick.dtype)
    tmp[0::2, 0::2, 0::2] = brick
    tmp[0::2, 0::2, 1::2] = brick + 1
    tmp[0::2, 1::2, 0::2] = brick + 2
    tmp[0::2, 1::2, 1::2] = brick + 3
    tmp[1::2, 0::2, 0::2] = brick + 4
    tmp[1::2, 0::2, 1::2] = brick + 5
    tmp[1::2, 1::2, 0::2] = brick + 6
    tmp[1::2, 1::2, 1::2] = brick + 3 if special is None else special
    bricks = []
    bricks.append(tmp[:half[0], :half[1], :half[2]].copy())
    bricks.append(tmp[:half[0], :half[1], half[2]:].copy())
    bricks.append(tmp[:half[0], half[1]:, :half[2]].copy())
    bricks.append(tmp[:half[0], half[1]:, half[2]:].copy())
    bricks.append(tmp[half[0]:, :half[1], :half[2]].copy())
    bricks.append(tmp[half[0]:, :half[1], half[2]:].copy())
    bricks.append(tmp[half[0]:, half[1]:, :half[2]].copy())
    bricks.append(tmp[half[0]:, half[1]:, half[2]:].copy())
    return bricks

def testLodAlgorithmsWithData(b0):
    """
    This will be run once with int8 data and once with float32
    data including NaNs. For most of the algorithms the expected
    result will be the same. In the int8 case the NaN values are
    replaced with a sample equal to the average of the other
    samples in that 2x2x2 region.
    """
    is_float = not np.issubdtype(b0.dtype, np.integer)
    bricks = _make8(b0, np.nan if is_float else None)
    # Make a variant of the test data that has one zero in each 2x2x2
    # neighborhood instead of a NaN, and that has no other zeros.
    b0x = b0.copy()
    b0x[b0x <= 0] = 1
    bricks_with_0 = _make8(b0x, 0)
    histogram = HistogramData(dtype=b0.dtype)
    histogram.add(bricks)
    result_LowPass  = decimate8(bricks, DecimationType.LowPass)
    result_WeightedAverage = decimate8(bricks, DecimationType.WeightedAverage, histogram = histogram, defaultvalue = np.nan)
    result_Average  = decimate8(bricks, DecimationType.Average)
    result_Median   = decimate8(bricks, DecimationType.Median)
    result_Minimum  = decimate8(bricks, DecimationType.Minimum)
    result_Maximum  = decimate8(bricks, DecimationType.Maximum)
    result_Decimate = decimate8(bricks, DecimationType.Decimate)
    result_AllZero  = decimate8(bricks, DecimationType.AllZero)
    result_AvgNon0  = decimate8(bricks_with_0, DecimationType.AverageNon0)
    assert np.all(result_Average  == b0 + 3)
    assert np.all(result_Median   == b0 + 3)
    assert np.all(result_Minimum  == b0)
    assert np.all(result_Maximum  == b0 + 6)
    assert np.all(result_Decimate == b0)
    assert np.all(result_AllZero  == np.zeros_like(bricks[0]))
    assert np.all(result_AvgNon0  == b0x + 3)
    # Expected result for LowPass is too difficult to compute if
    # a random cube is input. For slowly varying data it ought to
    # be similar to the Decimate case. And for all-constant data
    # the output should be that same constant.
    # The same reasoning applies to WeightedAverage.

def testLodAlgorithms():
    bricksize = (64, 64, 64)
    b0 = (np.random.random(bricksize) * 201 - 100).astype(np.int32)
    #print(b0[0,0,:])
    testLodAlgorithmsWithData(b0.astype(np.int8))
    testLodAlgorithmsWithData(b0.astype(np.float32))

def testLowpassLodAlgorithm():
    bricksize = (64, 64, 64)
    b0 = np.arange(np.product(bricksize), dtype=np.float64).reshape(bricksize)
    #b0.fill(42)
    bricks = _make8(b0, None) # TODO-Test also test NaN handling?
    lowpass = decimate8(bricks, DecimationType.LowPass)
    delta = lowpass - b0
    error = np.sqrt(np.average(delta * delta))
    #print("input  ", bricks[0][0,0,:])
    #print("lowpass", lowpass[0,0,:])
    #print("error  ", error)
    assert error < 0.1

    b0.fill(42)
    for b in bricks: b.fill(42)
    lowpass = decimate8(bricks, DecimationType.LowPass)
    delta = lowpass - b0
    error = np.sqrt(np.average(delta * delta))
    #print("input  ", bricks[0][0,0,:])
    #print("lowpass", lowpass[0,0,:])
    #print("error  ", error)
    assert error < 0.0001

    brick = [0.5, 0.1, 0.01] + 23*[0] + [1] + 37*[0]
    brick = np.array(4*brick)
    brick = brick.reshape((2, 2, len(brick)//4))
    d = decimate(brick, DecimationType.LowPass)
    # This input is simple enough that it makes sense to just
    # eyeball it. Not adding any asserts here.
    #print(brick[0,0,:])
    #print(d[0,0,:])

def testSpecial(dtype=np.float32, last=0):
    """
    There is some overlap with testLodAlgorithmsWithData(), but this
    test also does the newly added WeightedAverage algorithm which is
    important for Petrel.
    """
    cast = np.dtype(dtype).type
    cube = np.zeros((6, 10, 20), dtype=dtype)
    nibble = np.array([250, 42, 42, -1, -1, -1, 0, last], dtype=dtype)
    cube[:2,:2,:2] = nibble.reshape((2,2,2))
    # I want a total of 1*250, 5*42, 100*-1, 1094*0
    cube[2,0,:3] = 42
    cube[3,:10,:10] = -1 # add 100
    cube[3,0,:3] = 0 # and remove 3
    histogram = HistogramData(range_hint=(-3,+252), dtype=dtype)
    if dtype == np.int16:
        # Need a 1:1 mapping of inregral input data to bins
        histogram._size = 65536
        histogram._bins = np.zeros(histogram._size, dtype=np.int64)
    histogram.add(cube)

    # Expected weighted average computed by hand.
    s = 250/1 + 2*42/5 - 3*1/100 + 2*0/1094
    w = 1/1 + 2/5 + 3/100 + 2/1094
    if not np.isfinite(last): w -= 1/1094
    expect_wavg = s / w

    # Expected lowpass was just extracted from a previous run.
    expect_lowpass = 141.996475

    expect = {
        DecimationType.LowPass:          expect_lowpass,
        DecimationType.WeightedAverage:  expect_wavg,
        DecimationType.Average:          np.nanmean(nibble),
        DecimationType.Median:           0,
        DecimationType.Minimum:          -1,
        DecimationType.Maximum:          max(250, last),
        DecimationType.MinMax:           None,
        DecimationType.Decimate:         250,
        DecimationType.DecimateSkipNaN:  None,
        DecimationType.DecimateRandom:   None,
        DecimationType.AllZero:          0,
        DecimationType.WhiteNoise:       None,
        DecimationType.MostFrequent:     None,
        DecimationType.MostFrequentNon0: None,
        DecimationType.AverageNon0:      None, # Set below
    }

    # expect0 is the output when all 8 inputs are zero.
    expect0 = { k: (None if v is None else 0) for k, v in expect.items() }
    expect0[DecimationType.AverageNon0] = 99 # defaultvalue because all 0
    expect[DecimationType.AverageNon0] = np.mean([250, 42, 42, -1, -1, -1])
    if last == np.inf:
        # last entry is now in upper part instead of lower part.
        # The two elements at the center are 42 and 0, so the median
        # is reported as the average of those two.
        expect[DecimationType.Median] = 21

    actual = {}
    actual0 = {}
    for mode in expect:
        if expect[mode] is not None:
            if mode == DecimationType.WeightedAverage:
                a = decimate(cube, mode, histogram=histogram, defaultvalue=0)
            elif mode == DecimationType.AverageNon0:
                a = decimate(cube, mode, defaultvalue = 99)
            else:
                a = decimate(cube, mode)
            actual[mode] = a[0,0,0]
            actual0[mode] = a[2,0,0]

    errors = 0
    for mode in expect:
        if expect[mode] is not None:
            if not np.isclose(cast(expect[mode]), cast(actual[mode])):
                print("Error in decimate {0}: expect {1} got {2}".format(
                    mode.name, cast(expect[mode]), cast(actual[mode])))
                errors += 1
            elif not np.isclose(cast(expect0[mode]), cast(actual0[mode])):
                print("Error in decimate#2 {0}: expect {1} got {2}".format(
                    mode.name, cast(expect0[mode]), cast(actual0[mode])))
                errors += 1
            #else:
            #    print("Decimation {0}: got {1} and {2}".format(
            #        mode.name, cast(actual[mode]), cast(actual0[mode])))
    assert not errors

def numpyBugs():
    """
    I am fairly sure this is a bug in numpy. In operations on masked
    arrays a value of np.inf is sometimes treated as masked and sometimes
    not. The behavior is repeatable but changing an expression in a very
    minor fashion may change the result. Bottom line, try to avoid code
    that depends on this "feature".

    The unit test contains asserts cagainst the currently observed
    behavior in numpy 1.18.2. If the test starts failing after a numpy
    upgrade then this might mean that the bug has been fixed.
    """

    data1d = np.array([250, 42, 42, -1, -1, -1, 0, np.inf])
    data1d = np.ma.masked_equal(data1d, 0)
    data2d = data1d.reshape(-1,1)

    out1d = np.mean(data1d)
    out2d = np.mean(data2d, 0)[0]
    weird = np.mean(data2d, 1)

    def typename(x): return type(x).__module__ + "." + type(x).__name__
    #print(out1d, typename(out1d))
    #print(out2d, typename(out2d))
    #print(weird)

    # mean of 1d array with both masked elements and +inf.
    # Expected and actual results are positive infinity.
    assert str(out1d) == "inf"

    # mean over a single dimension of a 2d array, where the
    # other dimenzion is 1. Effectively the exact same data
    # except the result will be an array of one element.
    # This is NOT the case. The resulting value is masked.
    assert str(out2d) == "--"

    # Trying to do a mean over the other dimension, giving
    # 8 "mean" values where the input for each of them is
    # just a single value. Expected result is the original
    # data. Observed result is that the np.inf has been
    # replaced with a masked value.
    assert str(weird) == "[250.0 42.0 42.0 -1.0 -1.0 -1.0 -- --]"

    # To add to the confusion, if the out1d array is still a
    # masked array but starts out with nothing masked then the
    # behavior of the first case will change. Now this will
    # also return a masked value instead of inf.
    data1d = np.array([250, 42, 42, -1, -1, -1, 2, np.inf])
    data1d = np.ma.masked_equal(data1d, 0)
    out1d = np.mean(data1d)
    #print(out1d)
    assert str(out1d) == "--"

if __name__ == "__main__":
    np.seterr(all='raise')
    testLodAlgorithms()
    testLowpassLodAlgorithm()
    testSpecial(np.int16)
    testSpecial(np.float32)
    testSpecial(np.float32, np.nan)
    testSpecial(np.float32, np.inf)
    numpyBugs()

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
