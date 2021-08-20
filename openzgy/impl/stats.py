#!/usr/bin/env python3

"""
Code to compute statistics.

    impl.stats.StatisticData:

        * Accumulate statistics: count, sum, sum of squares, and value range.
"""
##@package openzgy.impl.stats
#@brief Code to compute statistics.

import numpy as np

class StatisticData:
    """
    Accumulate statistics: count, sum, sum of squares, and value range.
    """
    def __init__(self, other = None):
        self._cnt = other._cnt if other else 0 # Number of added samples.
        self._inf = other._inf if other else 0 # Number of rejected samples.
        self._sum = other._sum if other else 0 # Sum of added samples.
        self._ssq = other._ssq if other else 0 # Sum-of-squares of samples.
        self._min = other._min if other else 1 # Minimum sample value.
        self._max = other._max if other else -1 # Maximum  sample value.
        # Note for min and max: These initial values are only interesting
        # if storing an empty file. As soon as real data is seen they get
        # replaced (due to min > max) and the value type may also change.

    def __repr__(self):
        return "StatisticData(cnt={_cnt}, inf={_inf}, sum={_sum}, ssq={_ssq}, min={_min}, max={_max})".format(**self.__dict__)

    def __str__(self):
        return "StatisticData({_cnt} samples)".format(**self.__dict__)

    def _add_numpy_array(self, value):
        """
        Add the data in the provided numpy array.
        Technically this function should also work for Python lists or
        even scalars since all the numpy functions called are robust
        enough to handle that. But that complication isn't really needed
        so I don't want to worry about testing it.
        TODO-Low make a simpler and more efficient version for integral types.
        TODO-Low performance boost to include a "factor" argument.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Only numpy arrays accepted when adding statistics.")
        valid = np.isfinite(value).astype(np.bool)
        scnt = np.count_nonzero(valid)
        sinf = np.count_nonzero(valid == False)
        if sinf == 0: valid = True
        # Obscure problem: initial is required for smin, smax when
        # where is used, in spite of me knowing that there is at least
        # one valid value. But, if initial=np.inf and value is an
        # integral type this goes rather badly. Using value[0] as
        # the initial is also problematic because it might be NaN.
        # To solve this I need to compute the range as float.
        # TODO-Low: or I could use int64 for all integral types but do I
        # really gain anything? Slightly more accurate results, yes,
        # but more corner cases to test and might I theoretically
        # overflow even an int64? The old code just used double.
        # TODO-Worry the existing schema stores value range as float32,
        # which is inaccurate if we ever decide to support int32 data.
        value = value.astype(np.float64, copy=False)
        if scnt:
            ssum = np.sum(value, where=valid, dtype=np.float64)
            sssq = np.sum(np.square(value, dtype=np.float64), where=valid, dtype=np.float64)
            smin = np.amin(value, where=valid, initial=np.inf)
            smax = np.amax(value, where=valid, initial=-np.inf)
            self._cnt += int(scnt)
            self._sum += ssum
            self._ssq += sssq
            if self._min is None or self._max is None or self._min > self._max:
                self._min = smin
                self._max = smax
            else:
                self._min = min(self._min, smin)
                self._max = max(self._max, smax)
        self._inf += int(sinf)
        return self

    def _add_other(self, other):
        """
        Add more samples to an existing StatisticData.
        other is also allowed to hold negative counts,
        this will cause samples to be removed. But the
        min/max range will still be expanded.
        """
        # if other.cnt_ == 0 then cnt, sum, ssq should also be zero
        # and min, max should be uninitialized. Don't trust that.
        if other._cnt != 0:
            if self._cnt != 0:
                # already have data, so the range needs to be combined.
                self._min = min(self._min, other._min)
                self._max = max(self._max, other._max)
            else:
                # our own min/max is bogus since we don't have any samples yet.
                self._min = other._min
                self._max = other._max
            self._cnt += int(other._cnt)
            self._sum += other._sum
            self._ssq += other._ssq
        self._inf += int(other._inf)
        return self

    def _multiply_scalar(self, factor):
        """
        Multiply StatisticsData with a constant N, equivalent to creating
        a new instance and adding the old one to it N times. N can also be
        negative. The min/max range is not affected.
        """
        if factor != int(factor):
            raise TypeError("Multiplication by integers only.")
        factor = int(factor) # In case it is np.int32 now.
        self._cnt *= factor;
        self._inf *= factor;
        self._sum *= factor;
        self._ssq *= factor;
        return self

    def _equal(self, other):
        if other is None:
            return False
        elif not isinstance(other, type(self)):
            raise TypeError()
        else:
            return (self._cnt == other._cnt and
                    self._inf == other._inf and
                    self._sum == other._sum and
                    self._ssq == other._ssq and
                    self._min == other._min and
                    self._max == other._max)

    def add(self, data, factor):
        if factor == 1:
            self._add_numpy_array(data)
        else:
            tmp = StatisticData()
            tmp._add_numpy_array(data)
            if factor != 1:
                tmp._multiply_scalar(factor)
            self._add_other(tmp)

    def scale(self, slope, intercept):
        """
        Calculate the linear transform needed to convert from one range
        (typically the natural data range of the integral storage type)
        to the data range that the application wants to see.
        Then update the statistics in place so they look like the transform
        had been done on every single data point before adding it.

        The decoded value Y is given by a linear transform of the coded value X:

          Y = intercept + slope*X

        where intercept and slope are given by the coding range and the value range
        of type T (see below). The statistics of Y are then:

          SUM_Y = SUM(intercept + slope*x)
                = n*intercept + slope*SUM(x) = n*intercept + slope*SUM_X

          SSQ_Y = SUM((intercept + slope*x)^2)
                = SUM(intercept^2 + 2*intercept*slope*x + slope^2*x^2)
                = n*intercept^2 + 2*intercept*slope*SUM(x) + slope^2*SUM(x^2)
                = n*intercept^2 + 2*intercept*slope*SUM_X + slope^2*SSQ_X

          MIN_Y = MIN(intercept + slope*x)
                = intercept + slope*MIN(x)
                = intercept + slope*MIN_X

          MAX_Y = MAX(intercept + slope*x)
                = intercept + slope*MAX(x)
                = intercept + slope*MAX_X
        """
        #slope = (newmax - newmin) / (oldmax - oldmin)
        #intercept = newmin - oldmin * slope
        self._ssq = self._cnt*intercept*intercept + 2*intercept*slope*self._sum + slope*slope*self._ssq;
        self._sum = self._cnt*intercept + slope*self._sum;
        self._min = intercept + slope*self._min;
        self._max = intercept + slope*self._max;
        return self

    def __eq__(self, other):
        return self._equal(other)

    def __ne__(self, other):
        return not self._equal(other)

    def __add__(self, other):
        if isinstance(other, StatisticData):
            return StatisticData(self)._add_other(other)
        elif isinstance(other, np.ndarray):
            return StatisticData(self)._add_numpy_array(other)
        else:
            raise TypeError("Can only add numpy arrays and other StatisticData instances")

    def __iadd__(self, other):
        if isinstance(other, StatisticData):
            return self._add_other(other)
        elif isinstance(other, np.ndarray):
            return self._add_numpy_array(other)
        else:
            raise TypeError("Can only add numpy arrays and other StatisticData instances")

    def __mul__(self, factor):
        return StatisticData(self)._multiply_scalar(factor)

    def __rmul__(self, factor):
        return StatisticData(self)._multiply_scalar(factor)

    def __imul__(self, factor):
        return self._multiply_scalar(factor)

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
