#!/usr/bin/env python3

##@package openzgy.impl.histogram

import numpy as np

class HistogramData:
    def __init__(self, range_hint=None, dtype=np.float32):
        self._hmin, self._hmax = self._suggestHistogramRange(range_hint, dtype)
        self._dtype = dtype
        self._size = 256
        # Uncomment the next lines to enable the temporary large histogram.
        # One problem is that the histogram may lose the zero centric
        # property that the user specified range might have, making the
        # result look bad.
        #if np.issubdtype(dtype, np.integer):
        #    self._size = max(256, min(65536, self._hmax - self._hmin + 1))
        self._bins = np.zeros(self._size, dtype=np.int64)
        if False:
            print("@ Histogram ", self._hmin, self._hmax, "for data", range_hint)

    @staticmethod
    def _suggestHistogramRange(range_hint, dtype):
        """
        Choose the histogram range to use.

        The histogram range is normally set to the coding range for
        integral data and the actual data range for float data.
        This method takes care of corner cases and (future) subtle
        tewaks such as making the range zero centric.
        """
        # This logic in C++ is found in GenLodImpl::suggestHistogramRange()
        # and has a much more detailed explanation of what is going on.
        bogus = (-128, +127)
        if np.issubdtype(dtype, np.integer):
            # The histogram is built from storage values so its maximum
            # possible range is already known. For int8 there will be one
            # bin for every possible value. For int16 we might do the same,
            # temporarily producing a histogram with 65,536 values and then
            # whittling it down to 256 bins before storing it. But for now
            # just map to the user provided coding range and hope that the
            # user didn't decide to use just a small part of the available
            # integer storage values.
            return (np.iinfo(dtype).min, np.iinfo(dtype).max)
        else:
            # Choose histogram range based on the min/max value collected
            # while writing lod 0. Always end up with a sane interval with
            # min < max to avoid problems generating the histogram and also
            # for applications reading the file later. For completely empty
            # files just use (-1,+1) which is as good a default as any.
            if (not range_hint or
                not np.isfinite(range_hint[0]) or
                not np.isfinite(range_hint[1]) or
                range_hint[0] > range_hint[1]):
                return bogus # nothing written or error.
            elif range_hint[0] < range_hint[1]:
                # This is the normal case for floating point data.
                # Don't return numpy types. They have weird rules.
                return (float(range_hint[0]), float(range_hint[1]))
            elif range_hint[0] > 0: # At this point, hint[0] == hint[1]
                return (0, range_hint[0]) # single positive value
            elif range_hint[0] < 0:
                return (range_hint[0], 0) # single negative value
            else:
                return bogus # all zero

    def _histogram_data(self, data):
        if not np.issubdtype(self._dtype, np.integer):
            # numpy.histogram is documented to ignore values outside range.
            # Handling of NaN is undocumented and currently reports errors
            # from low level code. So, map NaN to +Inf just in case.
            # TODO-Performance: This is not really a good idea.
            data = np.copy(data)
            data[np.isnan(data)] = np.inf
        return np.histogram(data, bins=self._size, range=self.np_range)[0]

    def add(self, data, factor = 1):
        tmp = self._histogram_data(data)
        if factor != 1:
            tmp *= factor
        self._bins += tmp

    def scale(self, a, b):
        self._hmin = a * self._hmin + b
        self._hmax = a * self._hmax + b

    def resize(self, newsize):
        binwidth = (self._hmax - self._hmin) / (self._size - 1)
        oldbins = self._bins
        oldsize = self._size
        self._size = newsize
        self._bins = np.zeros(self._size, dtype=np.int64)
        if np.count_nonzero(oldbins) == 0:
            return
        if newsize >= oldsize:
            self._bins[:oldsize] = oldbins
            self._hmax = self._hmin + binwidth * (self._size - 1)
            return
        skiplo = np.argmax(oldbins[::1] != 0)
        skiphi = np.argmax(oldbins[::-1] != 0)
        factor = max(1, (oldsize-skiplo-skiphi + (newsize-1)) // (newsize-2))
        factor = ((factor // 2) * 2) + 1 # Round up to make it odd.

        # Very minor issue: I reserve the first and last bin to hold
        # data from the misaligned part. If enerything ends up aligned
        # those two end up unused. I am absolutely sure no one will
        # notice. *except possibly* when running unit tests.

        # Adjust skiplo and skiphi upwards so that (a) neither moves
        # more than "factor", (b) neither becomes negative, (c) the
        # remaining size - skiphi - skiplo is a multiple of "factor",
        # and (d) any zero-centric property is preserved by making
        # sure the "zero" bin in the input ends up in the middle of
        # one of the output bins. The last one is where it gets really
        # tricky and TODO-High must be implemented. Or YAGNI, remove
        # the capability to resize.

        # Combine "factor" input bins into each output bin
        center_count = ((oldsize-skiphi-skiplo)//factor)*factor
        skiphi = oldsize - skiplo - center_count
        partial = np.sum(oldbins[skiplo:oldsize-skiphi].reshape(-1,factor),axis=1)
        # Mop up the ends that might have fewer than "factor" entries.
        head = np.sum(oldbins[:skiplo])
        tail = np.sum(oldbins[oldsize-skiphi:])
        self._bins[1:(center_count//factor)+1] = partial
        self._bins[0] = head
        self._bins[(center_count//factor)+1] = tail
        # The new binwidth must be binwidth*factor.
        # The new bin[1] corresponds to old bin[skiplo], so new bin[0]
        # must be new binwidth less than that.
        self._hmin = (self._hmin + binwidth * skiplo) - (binwidth*factor)
        self._hmax = self._hmin + (binwidth*factor) * (self._size-1)

    @property
    def bins(self):
        return self._bins

    @property
    def vv_range(self):
        """
        Histogram range, voxelvision and zgy style, with numbers
        representing the center value of the first and last bin.
        """
        return (self._hmin, self._hmax)

    @property
    def np_range(self):
        """
        Histogram range, numpy and salmon style, with numbers
        representing the edges of the first and last bin.
        """
        binwidth = (self._hmax - self._hmin) / (self._size - 1)
        return (self._hmin - binwidth/2, self._hmax + binwidth/2)

    def binvalue(self, bin_number):
        """
        Convert a single bin number to the center value of this bin.
        Note that in ZGY this will refer to storage values, so you
        may need to explicitly convert the result.
        """
        binwidth = (self._hmax - self._hmin) / (self._size - 1)
        return self._hmin + bin_number * binwidth

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
