#!/usr/bin/env python3

"""
Decimation algorithms to output low resolution bricks.
"""
##@package openzgy.impl.lodalgo
#@brief Decimation algorithms to output low resolution bricks.

import numpy as np
from enum import Enum
import warnings

from ..exception import ZgyUserError

##@brief Possible algorithms to generate LOD bricks.
class DecimationType(Enum):
    """
    Possible algorithms to generate LOD bricks.
    We might trim this list later to what is actually in use.
    The "classic" ZGY only uses the first two.

    CAVEAT: Many of these might be expensive to port and/or not
    possible to implement efficiently in Python.

    TODO-Low: Avoid exposing this enum to the public API.
    """
    LowPass = 0,           # Lowpass Z / decimate XY.
    WeightedAverage = 1,   # Weighted averaging (depends on global stats).
    Average = 2,           # Simple averaging.
    Median = 3,            # Somewhat more expensive averaging.
    Minimum = 4,           # Minimum value.
    Maximum = 5,           # Maximum value.
    MinMax = 6,            # Checkerboard of minimum and maximum values.
    Decimate = 7,          # Simple decimation, use first sample.
    DecimateSkipNaN = 8,   # Use first sample that is not NaN.
    DecimateRandom = 9,    # Random decimation using a fixed seed.
    AllZero = 10,          # Just fill the LOD brick with zeroes.
    WhiteNoise = 11,       # Fill with white noise, hope nobody notices.
    MostFrequent = 12,     # The value that occurs most frequently.
    MostFrequentNon0 = 13, # The non-zero value that occurs most frequently.
    AverageNon0 = 14,      # Average value, but treat 0 as NaN.

def _reorder(brick):
    """
    Reorder the input brick, decimating it by a factor 2 but keeping
    each of the 2x2x2 samples that are to be combined in the last
    dimension. E.g. an input shape (64, 64, 64) returns a (32, 32, 32, 8).
    """
    shape = (brick.shape[0]//2, brick.shape[1]//2, brick.shape[2]//2, 8)
    tmp = np.zeros(shape, dtype=brick.dtype)
    tmp[...,0] = brick[0::2,0::2,0::2]
    tmp[...,1] = brick[0::2,0::2,1::2]
    tmp[...,2] = brick[0::2,1::2,0::2]
    tmp[...,3] = brick[0::2,1::2,1::2]
    tmp[...,4] = brick[1::2,0::2,0::2]
    tmp[...,5] = brick[1::2,0::2,1::2]
    tmp[...,6] = brick[1::2,1::2,0::2]
    tmp[...,7] = brick[1::2,1::2,1::2]
    return tmp

_filter = np.array([
    +0.07996591908598821,
    -0.06968050331585399,
    -0.10596589191287473,
    +0.12716426995479813,
    +0.44963820678761110,
    +0.44963820678761110,
    +0.12716426995479813,
    -0.10596589191287473,
    -0.06968050331585399,
    +0.07996591908598821], dtype=np.float64) * 1.0392374478340252

def _decimate_LowPass(a):
    # TODO-Test, this needs very careful testing. I am sort of guessing here.
    # TODO-Medium, NaN and Inf handling compatible with what the C++ impl does.
    # TODO-Worry, if the algoritm doesn't have precisely zero DC response
    # I might need to explicitly test for all-constant traces to avoid
    # an empty fullres brick turning into a non-empty LOD brick.
    # See Salmon/Zgy/ArrayBasic/ArrayTile.cpp
    # See Salmon/Zgy/Common/Sampling.cpp
    # See numpy.convolve
    dtype = a.dtype
    a = a[::2,::2,:].astype(np.float32)
    a = np.pad(a, ((0,0), (0,0), (5,5)), mode = 'reflect')
    a = np.convolve(a.flatten('C'), _filter, mode = 'same').reshape(a.shape)
    a = a[..., 5:-5:2]
    return a.astype(dtype)

def _decimate_WeightedAverage(data, *, histogram, defaultvalue):
    """
    Decimate the input by weighted average, with the weights being the inverse
    of how common this sample value is in the entire survey. This means that
    the algorithm needs access to the survey histogram. Which might be a
    challenge if the histogram is being computed in parallel with the low
    resolution blocks.

    The computation can be run either in "storage" mode or in "float" mode.
    The data buffer and the histogram obviously need to be in the same mode.

    defaultvalue is used if all 8 inputs for one output sample are nan/inf.
    For float data this would almost always be 0.0 and for integral data
    the input samples cannot be nan/inf. So it might as well be 0 always.

    This fuction looks very different from the C++ version in the old ZGY
    accessor. The reason is that in C++ it is no big deal to have a tight
    inner loop processing one sample at a time. In Python that approach is
    a joke. On the other hand, Python has the fantastic numpy module.

    Note that I might choose to remove all NaN and Inf values in the
    _decimate_LowPass method that is typically used to generate lod 1.
    In that case the extra testung for NaN might be redundant.
    """

    if histogram is None or np.sum(histogram.bins) == 0:
        raise ZgyInternalError("The WeightedAverage algorithm needs a histogram.")

    # Linear transform from value to bin number.
    # Should round to nearest: bin = np.rint(value * factor + offset)
    r = histogram.vv_range
    n = histogram.bins.size
    factor = (n - 1)/ (r[1] - r[0])
    offset = -r[0] * factor

    # Better to be safe. All computation in double, to avoid surprises
    # e.g. with huge sample counts causing numeric underflow. Also handle
    # NaN / Inf values explicitly to avoid a lot of testing.
    input_dtype = data.dtype
    integral = np.issubdtype(input_dtype, np.integer)
    data = data.astype(np.float64, copy=True)
    if not integral:
        ugly = np.logical_not(np.isfinite(data))
        if np.any(ugly):
            data[ugly] = 0
        else:
            ugly = None
    else:
        ugly = None

    # Get histogram bin number for every input sample.
    bin_numbers = np.rint(data * factor + offset)
    np.clip(bin_numbers, 0, histogram.bins.size - 1, out = bin_numbers)
    bin_numbers = bin_numbers.astype(np.int32)

    # Get the sample's frequency for every input sample.
    src_frequency = histogram.bins[bin_numbers]

    # If frequency is reported as 0, this is an inconsistency.
    # Since we know it occurs at least one time.
    np.clip(src_frequency, 1, None, out=src_frequency)

    tmp_weight = np.float64(1) / src_frequency
    if ugly is not None:
        tmp_weight[ugly] = 0

    tmp_wsum = data * tmp_weight

    # Now sum each 2x2x2 set of samples.
    tmp_wsum = np.nansum(_reorder(tmp_wsum), 3)
    tmp_weight = np.nansum(_reorder(tmp_weight), 3)

    # Zero weights means no valid samples found among the 8 inputs.
    # This can only happen if non-finite data was present in the input,
    # causing weights to be set to zero earlier in this function.
    if ugly is not None:
        ugly2 = tmp_weight == 0
        tmp_wsum[ugly2] = defaultvalue
        tmp_weight[ugly2] = 1

    # Final result is the weighted sum of 8 input samples per output sample.
    # If all 8 samples were NaN we instead want to substitute an explicit
    # default value, or possibly set them back to to NaN.
    tmp_wsum /= tmp_weight

    # Convert back to user's type, protecting against integer overflow.
    if not integral:
        tmp_wsum = tmp_wsum.astype(np.float32, copy=False)
    else:
        cliprange = (np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
        np.clip(tmp_wsum, cliprange[0], cliprange[1], out=tmp_wsum)
        tmp_wsum = np.rint(tmp_wsum).astype(input_dtype, copy=False)

    return tmp_wsum

def _decimate_Average(brick):
    tmp = _reorder(brick.astype(np.double))
    # All NaN will return NaN, ditto for average of +inf, -inf.
    # I have not seen this documented explicitly but it should be
    # a safe assumption. Also the consequences if wrong are slight.
    # So, ignore the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(tmp, 3).astype(brick.dtype)

def _decimate_Median(brick):
    tmp = _reorder(brick.astype(np.double))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(tmp, 3).astype(brick.dtype)

def _decimate_Minimum(brick):
    tmp = _reorder(brick)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(tmp, 3)

def _decimate_Maximum(brick):
    tmp = _reorder(brick)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(tmp, 3)

def _decimate_Decimate(brick):
    return brick[0::2,0::2,0::2]

def _decimate_AllZero(brick):
    shape = (brick.shape[0]//2, brick.shape[1]//2, brick.shape[2]//2)
    return np.zeros(shape, dtype=brick.dtype)

def _decimate_AverageNon0(brick, *, defaultvalue = 0):
    """Average not considering 0, nan, inf. Use defaultvalue if all invalid."""
    tmp = _reorder(brick.astype(np.double))
    mask = np.isfinite(tmp)
    np.logical_not(mask, out=mask)
    np.logical_or(mask, tmp == 0, out=mask)
    tmp = np.ma.array(tmp, mask=mask, fill_value = defaultvalue)
    tmp = np.mean(tmp, 3)
    tmp.fill_value = defaultvalue # mean() doesn't preserve it
    return tmp.filled().astype(brick.dtype)

_decimation_algo = {
    DecimationType.LowPass:          _decimate_LowPass,
    DecimationType.WeightedAverage:  _decimate_WeightedAverage,
    DecimationType.Average:          _decimate_Average,
    DecimationType.Median:           _decimate_Median,
    DecimationType.Minimum:          _decimate_Minimum,
    DecimationType.Maximum:          _decimate_Maximum,
    #DecimationType.MinMax:           _decimate_MinMax,
    DecimationType.Decimate:         _decimate_Decimate,
    #DecimationType.DecimateSkipNaN:  _decimate_DecimateSkipNaN,
    #DecimationType.DecimateRandom:   _decimate_DecimateRandom,
    DecimationType.AllZero:          _decimate_AllZero,
    #DecimationType.WhiteNoise:       _decimate_WhiteNoise,
    #DecimationType.MostFrequent:     _decimate_MostFrequent,
    #DecimationType.MostFrequentNon0: _decimate_MostFrequentNon0,
    DecimationType.AverageNon0:      _decimate_AverageNon0,
}

def _is_power_of_two_ge_2(n):
    for shift in range(1, 32):
        if n == 1<<shift:
            return True
    return False

def _is_all_power_of_two_ge_2(nn):
    return np.all([_is_power_of_two_ge_2(n) for n in nn])

def _combine_eight_bricks(bricks):
    """
    Paste together 8 equal-sized bricks. Called only from decimate8(),
    which is not called from production code. Shoud perhaps be moved
    to the unit test module.
    """
    half = bricks[0].shape
    full = (half[0]*2, half[1]*2, half[2]*2)
    result = np.zeros(full, dtype=bricks[0].dtype)
    result[:half[0], :half[1], :half[2]] = bricks[0]
    result[:half[0], :half[1], half[2]:] = bricks[1]
    result[:half[0], half[1]:, :half[2]] = bricks[2]
    result[:half[0], half[1]:, half[2]:] = bricks[3]
    result[half[0]:, :half[1], :half[2]] = bricks[4]
    result[half[0]:, :half[1], half[2]:] = bricks[5]
    result[half[0]:, half[1]:, :half[2]] = bricks[6]
    result[half[0]:, half[1]:, half[2]:] = bricks[7]
    return result

def decimate(brick, algo, **kwargs):
    """
    Decimate a single input brick to produce one smaller output brick.
    """
    # The following test is only a requirement for bricksize.
    # decimate() can be called e.g. for entire brick columns,
    # and in that case the only requirement is that the sizes
    # are all even.
    #if not _is_all_power_of_two_ge_2(brick.shape):
    #    raise ZgyUserError("Brick size must be >= 2 and a power of 2")
    if not all([(n%2) == 0 for n in brick.shape]):
         raise ZgyUserError("Decimation can only be run on even-sized bricks")
    try:
        # TODO-Worry ... Remove the need for this kludge ...
        # TODO-Worry ... NOT THREADSAFE ... NOT THREADSAFE ... NOT THREADSAFE ...
        np_errors = np.seterr(all='print')
        if algo in _decimation_algo:
            return _decimation_algo[algo](brick, **kwargs)
        else:
            raise NotImplementedError(str(algo))
    finally:
        np.seterr(**np_errors)

def decimate8(bricks, algo, **kwargs):
    """
    Decimate 8 equal-sized input bricks to produce one output.

    Currently not used in production code. Only in the unit test.
    The genlod module prefers to split up and paste its data itself.
    This function shold perhaps be moved to the unit test module.

    Most of the decimation algorithms operate om a local 2x2x2 region
    for each output sample. So it makes no difference whether we
    decimate each input brick and then combine them, or the other
    way around. Combining last *might* use less memory if lazy
    evaluation works. Combining first might help the algorithm perform
    better (LowPass) or faster (WeightedAverage?)
    """
    test1 = decimate(_combine_eight_bricks(bricks), algo, **kwargs)
    test2 = _combine_eight_bricks([decimate(b, algo, **kwargs) for b in bricks])
    if algo != DecimationType.LowPass:
        assert np.all(test1 == test2)
    return test1

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
