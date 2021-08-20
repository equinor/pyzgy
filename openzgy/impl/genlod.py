#!/usr/bin/env python3

##@package openzgy.impl.genlod

# TODO-Low: Several tweaks are possible but might not offer
# much value. See a detailed discussion in doc/lowres.html.

# Testing notes:
#   test.black.checkLodContents(), checkStatistics(), checkHistogram()
#   is part of a reasonably complete end to end test checking this module.
#   A new unit test testFinalizeProgress() has been written to verify the
#   progress mechanism. Both that the done and total end up precisely
#   identical at the end and that the client can abort the calculation.

import sys
import argparse
import numpy as np
from collections import namedtuple

from . import enum as impl_enum
from ..exception import *
from .stats import StatisticData
from .histogram import HistogramData
from .lodalgo import decimate, DecimationType
from .bulk import ScalarBuffer

def _round_down_to_power_of_2(n):
    exp = -1
    while n > 0:
        n >>= 1
        exp += 1
    return 1<<exp

def _make_blocksizes(bricksize, surveysize, nlods, dtype, factor=(1,1,1), verbose=None):
    """
    CURRENTLY NOT USED.

    Calculate the minimum blocksize to read at each lod level. Clip to
    the survey size. Also compute the memory needed to hold one buffer
    for each lod. Note that the genlod algorithm currently assumes
    that the block size is the same for all levels except for the
    clipping. And it currently handles clipping itself and might not
    like us to do so. Currently this function is not very useful.
    """
    blocksizes = np.zeros((nlods, 3), dtype=np.int64)
    ss = np.array(surveysize, dtype=np.int64)
    bs = np.array([2*factor[0]*bricksize[0],
                   2*factor[1]*bricksize[1],
                   ss[2]], dtype=np.int64)
    iterations = 0
    for lod in range(nlods):
        bs = np.minimum(bs, ss)
        blocksizes[lod] = bs
        iterations += np.product((ss+bs-1) // bs)
        ss = (ss + 1) // 2
    bytesused = np.sum(np.product(blocksizes, axis=1)) * int(np.dtype(dtype).itemsize)
    returntype = namedtuple("BlockSizeInfo", "blocksizes bytesused iterations")
    result = returntype(blocksizes, bytesused, iterations)
    print(result)
    return result

def _choose_blocksizes(bricksize, surveysize, nlods, dtype, maxmem=512*1024*1024):
    """
    CURRENTLY NOT USED.

    Plan B and C are rather memory intensive already. So we might not
    be able to increase the buffer sizes very much.

    Calculate the actual blocksize to read at each lod level. If there
    is enough memory available we can multiply the basic block size by
    some factor in the j direction. And if there is still more memory
    left it can be multiplied in the J directon as well. There is a
    benefit when the file resides on the cloud because the code will
    read larger blocks from lod 0. There is also a benefit both for
    cloud and on-prem files in that low resolutun bricks end up more
    contiguous.

    Beware that the current genlod algorithm assumes the block size is
    the same at each level. It will do exactly 4 reads at lod-1 to
    produce one brick at lod. There is also a requirement that the
    facors are a power of 2. Neither of those limitations apply to the
    non-recursive plan B. Which is currently not implemented.

    This means that the factor we multipy with should be the same for
    all lods. Had it not been for this, it would probably have made
    sense to make the blocks larger in (i,j) since they will be
    smaller in k.

    For very small surveys there is another caveat. The granularity
    of calls to the progress callback will be the block size.
    If the buffer gets too large then a progress bar won't be
    very useful.
    """
    result  = _make_blocksizes(bricksize, surveysize, nlods, dtype,
                               factor=(1,1,1))
    jfactor = max(1, maxmem // result.bytesused)
    jfactor = _round_down_to_power_of_2(jfactor)
    result  = _make_blocksizes(bricksize, surveysize, nlods, dtype,
                               factor=(1,jfactor,1))
    ifactor = max(1, maxmem // result.bytesused)
    ifactor = _round_down_to_power_of_2(ifactor)
    result = _make_blocksizes(bricksize, surveysize, nlods, dtype,
                              factor=(ifactor,jfactor,1), verbose=None)
    return result

class GenLodBase:
    """
    Abstract class for generating low resolution bricks, histogram,
    and statistics. At this level only define virtual methods for I/O.
    The implementation can be used as-is when mocking the class.
    The optional nlods parameter is only used as a consistency check.

    Note that the WeightedAverage algorithm requires a histogram to work.
    If no histogram was provided then the current contents of the
    accumulated histogram will be used. This is unfortunate and might
    cause brick artifacts. Especially in the first few bricks that are
    generated. With a non-recursive algorithm (plan C) and with only
    lod2 and above uses weighted average then this is unproblematic.
    Because in that case we will be done with the histogram once we
    need it. TODO-Low consider doing an initial pass with a statistical
    sampling of the lod0 data, only for use with weighted average.
    There will be a minor issue with some values appearing to have zero
    frequency, but this should not cause any trouble. (assume "1").

    Note that the WeightedAverage and AverageNon0 algorithms expect a
    defaultvalue to use when all inputs are inf/nan or (for AverageNon0)
    zero. Only relevant for integral types, to ensure that the default
    is whatever will produce the value closest to 0 after conversion.
    And integral data can neither be inf nor nan, so this is a pretty
    academic issue. For AverageNon0 that algorithm is currently not
    used. So it isn't even clear what the desired behavior should be.
    """
    def __init__(self, size, *, bricksize = (64, 64, 64), dtype = np.float32,
                 range_hint = None, nlods = None,
                 decimation = None, histogram = None, defaultvalue = None,
                 progress = None, verbose = None):
        _nlods = 1 # The loop stops before counting final level
        _total = 1 # Total number of bricks in all levels.
        bs = np.array(bricksize, dtype=np.int64)
        sz = np.array(size, dtype=np.int64)
        while np.any(sz > bs):
            _nlods += 1
            _total += np.product((sz + bs - 1) // bs)
            sz = (sz + 1) // 2
        assert nlods is None or nlods == _nlods
        self._surveysize = np.array(size, dtype=np.int64)
        self._bricksize = np.array(bricksize, dtype=np.int64)
        self._dtype = dtype
        self._range_hint = range_hint or (-1, 1)
        self._progress = progress
        self._verbose = verbose or (lambda *args, **kw: False)
        self._done = 0
        self._total = _total
        self._nlods = _nlods
        self._surveysize.flags.writeable = False
        self._bricksize.flags.writeable = False
        self._decimation_type = decimation or [DecimationType.LowPass,
                                               DecimationType.WeightedAverage]
        self._wa_histogram = histogram # Might become self._histo later.
        self._wa_defaultvalue = defaultvalue or 0
        self._verbose("@survey {0}".format(tuple(self._surveysize)))
        #_choose_blocksizes(bricksize, size, nlods, dtype)
        self._report(None)

    def _report(self, data):
        """
        Invoke the user's progress callback if any.
        Keep track of how many bricks we have processed. Both reads and
        writes increment the same counter. For plan C the reads will cover
        all blocks in lod 0 and the writes will cover all blocks in lod > 0.
        For plan D all blocks are written which means the computation of
        _total done in __init__ might need to change.
        """
        if data is not None:
            count = np.product((np.array(data.shape, dtype=np.int64) +
                                self._bricksize - 1) // self._bricksize)
            self._done += count
        if self._progress and not self._progress(self._done, self._total):
            raise ZgyAborted("Computation of low resolution data was aborted")

    def _read(self, lod, pos, size):
        """
        This is a stub that must be redefined except for low level unit tests.
        Read a block from the ZGY file (plans B and C) or the application
        (plan D). The lod parameter will always be 0 for plans C and D.
        Returns a ScalarBuffer if all constant, else a 3D numpy array.
        """
        result = ScalarBuffer(size, 0, self._dtype)
        self._report(result)
        return result

    def _write(self, lod, pos, data):
        """
        This is a stub that must be redefined except for low level unit tests.
        Write a block to the ZGY file. Silently ignore writes of data that
        is known to have been read directly from the file. For plans B and C
        this means ignoring all writes to lod 0.
        """
        self._report(data)

    def _savestats(self):
        """
        This is a stub that must be redefined except for low level unit tests.
        Finalize and write the computed statistics and histogram to the file.
        """
        pass

    def _prefix(self, lod):
        """For debugging and logging only."""
        return "  " * (self._nlods-1 - lod)

    @staticmethod
    def _format_result(result):
        """For debugging and logging only."""
        if isinstance(result, np.ndarray):
            return "array" + str(tuple(result.shape))
        else:
            return repr(result)

class GenLodImpl(GenLodBase):
    """
    Abstract class for generating low resolution bricks, histogram,
    and statistics. The inherited methods for I/O are still stubs.
    See doc/lowres.html for details. This class implements plan C or D
    which is good for compressed data and acceptable for uncompressed.
    The ordering of low resolution bricks in the file will not be optimal.
    For optimal ordering but working only for uncompressed data consider
    implementing plan B in addition to the plan C already implemented.
    The implementation can be used as-is in a unit test with mocked I/O.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #print("GenLodImpl", args, kwargs)
        self._stats = StatisticData()
        self._histo = HistogramData(range_hint=self._range_hint,
                                    dtype=self._dtype)
        # See base constructor. Better than no histogram at all.
        self._wa_histogram = self._wa_histogram or self._histo

    def __call__(self):
        """
        Generate and store statistics, histogram, and all low resolution
        bricks. Works for plans C and D. If we also need an implementation
        of plan B then this method wold need to iterate over all bricks
        and lods, and _accumulate would not make any recursive calls.
        """
        self._calculate((0,0,0), self._nlods-1)
        self._savestats() # Remove?
        return self._stats, self._histo

    def _accumulate(self, data):
        """
        Keep a running tally of statistics and histogram.
        """
        if data is None:
            return
        factor = 1
        if isinstance(data, ScalarBuffer):
            factor = np.product(data.shape, dtype=np.int64)
            data = np.array([data.value], dtype=data.dtype)
        self._stats.add(data, factor)
        self._histo.add(data, factor)

    def _calculate(self, readpos, readlod):
        """
        Read data from the specified (readpos, readlod) and store it back.
        The function will itself decide how much to read. But with several
        constraints. Always read full traces. Size in i and j needs to be
        2* bs * 2^N  where bs is the file's brick size in that dimension,
        Clipped to the survey boundaries. This might give an empty result.

        When readlod is 0 and the data was read from the ZGY file then the
        writing part is skipped. Since the data is obviously there already.

        In addition to reading and writing at the readlod level, the
        method will compute a single decimated buffer at readlod+1 and
        return it. As with the read/write the buffer might be smaller at
        the survey edge. Note that the caller is responsible for storing
        the decimated data.

        Full resolution data (lod 0) will be read from file (plan C) or the
        application (plan D). Low resolution is computed by a recursive call
        to this function (plans C and D) or by reading the file (plan B).
        Note that currently only plan C is fully implemented.

        For plans B and C a single call needs to be made to read the brick
        (there is by definition just one) at the highest level of detail.
        This will end up computing all possible low resolution bricks and
        storing them. For plan B the caller must iterate.
        """
        surveysize = (self._surveysize + (1<<readlod) - 1) // (1<<readlod)
        readpos = np.array((readpos[0], readpos[1], 0), dtype=np.int64)

        if readpos[0] >= surveysize[0] or readpos[1] >= surveysize[1]:
            return None

        readsize = np.minimum(2*self._bricksize, surveysize - readpos)
        readsize[2] = surveysize[2] # Always read full traces.
        writesize = (readsize + 1) // 2

        self._verbose("@{0}calculate(lod={1}, pos={2})".format(
            self._prefix(readlod), readlod, tuple(readpos)))
        if readlod == 0:
            data = self._read(lod=readlod, pos=readpos, size=readsize)
            self._accumulate(data)
        else:
            offsets = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
            offsets = np.array(offsets, dtype=np.int64) * self._bricksize
            hires = [None, None, None, None]
            for i in range(4):
                hires[i] = self._calculate(2*readpos + 2*offsets[i], readlod-1)
            data = self._paste4(hires[0], hires[1], hires[2], hires[3])

        self._write(lod=readlod, pos=readpos, data=data)

        if readlod == self._nlods - 1:
            result = None # Caller will discard it anyway.
            assert self._done == self._total
        elif not isinstance(data, np.ndarray):
            result = ScalarBuffer(writesize, data.value, data.dtype)
        else:
            result = self._decimate(data, lod=readlod+1)
            assert tuple(result.shape) == tuple(writesize)

        self._verbose("@{0}calculate returns(lod={1}, pos={2}, data={3})".format(
            self._prefix(readlod), readlod+1, tuple(readpos//2),
            self._format_result(result)))
        return result

    def _decimate(self, data, lod):
        """
        Return a decimated version of the input buffer with half the size
        (rounded up) in each dimension. In total the result will be ~1/8
        the size of the input.
        """
        if data is None:
            return None
        elif isinstance(data, ScalarBuffer):
            size = ((data.shape[0]+1)//2,
                    (data.shape[1]+1)//2,
                    (data.shape[2]+1)//2)
            return ScalarBuffer(size, data.value, data.dtype)
        else:
            di = data.shape[0] % 2
            dj = data.shape[1] % 2
            dk = data.shape[2] % 2
            if di != 0 or dj != 0 or dk != 0:
                data = np.pad(data, ((0, di), (0, dj), (0, dk)), mode='edge')
            #return data[::2,::2,::2]
            dt = self._decimation_type[min(lod, len(self._decimation_type)-1)]
            if dt == DecimationType.WeightedAverage:
                return decimate(data, dt,
                                histogram = self._wa_histogram,
                                defaultvalue = self._wa_defaultvalue)
            elif dt == DecimationType.AverageNon0:
                return decimate(data, dt,
                                defaultvalue = self._wa_defaultvalue)
            else:
                return decimate(data, dt)

    def _paste1(self, result, more, ioff, joff):
        """
        See _paste4() for details.
        """
        if more is not None:
            value = more.value if isinstance(more, ScalarBuffer) else more
            result[ioff:ioff+more.shape[0], joff:joff+more.shape[1], :] = value

    def _paste4(self, d00, d01, d10, d11):
        """
        Combine 4 buffers into one. Input buffers may be None (do not
        paste) or ScalarBuffer (paste a constant value). If all not-None
        buffers are just scalars then the return from this function
        will also be a scalar. d01 adds more data in the J direction,
        so it starts at i=0, j>0 in the target. Similarly d10 adds
        more in the J direction. And d11 in the diagonal.
        """
        if d01 is None and d10 is None and d11 is None:
            return d00 # Nothing to paste. Also works for None or scalar.

        assert d00 is not None # Empty d00 + non-empty others is not good.
        assert d01 is None or d01.shape[0] == d00.shape[0]
        assert d10 is None or d10.shape[1] == d00.shape[1]
        if d01 is not None and d10 is not None:
            # The "diagonal" brick must exist with the right size.
            assert d11 is not None
            assert d11.shape[1] == d01.shape[1] and d11.shape[0] == d10.shape[0]
        else:
            # The "diagonal" brick should not be needed in this case.
            assert d11 is None

        ni = d00.shape[0] + (d10.shape[0] if d10 is not None else 0)
        nj = d00.shape[1] + (d01.shape[1] if d01 is not None else 0)
        nk = d00.shape[2]

        all_same = True
        for e in (d00, d01, d10, d11):
            if all_same and e is not None:
                if not isinstance(e, ScalarBuffer) or e.value != d00.value:
                    all_same = False
        if all_same:
            result = ScalarBuffer((ni, nj, nk), d00.value, d00.dtype)
        else:
            result = np.zeros((ni, nj, nk), dtype=d00.dtype)
            self._paste1(result, d00, 0, 0)
            self._paste1(result, d01, 0, d00.shape[1])
            self._paste1(result, d10, d00.shape[0], 0)
            self._paste1(result, d11, d00.shape[0], d00.shape[1])
        return result

class GenLodC(GenLodImpl):
    """
    Generate and store low resolution bricks, histogram, and statistics.
    See doc/lowres.html for details. I/O is done via ZgyInternalBulk.
    Use this class as part as finalize().
    The implementation uses plan C, which means the full resolution data
    will be read from the ZGY file. To implement plan D, make a derived
    class that redefines _read() to query the client for the required full
    resolution data. _read() must then also call _write() to store the
    data it just received.
    """
    def __init__(self, accessor, *,
                 compressor = None, decimation = None,
                 progress = None, verbose = None):
        # Could add defaultvalue. And add a unit test for correct operation.
        super().__init__(size = accessor._metadata._ih._size,
                         bricksize = accessor._metadata._ih._bricksize,
                         dtype = impl_enum._map_DataTypeToNumpyType(
                             accessor._metadata._ih._datatype),
                         range_hint = (accessor._sample_min,
                                       accessor._sample_max),
                         nlods = accessor._metadata._ih._nlods,
                         decimation = decimation,
                         progress = progress, verbose = verbose)
        self._accessor = accessor
        self._compressor = compressor

    def _read(self, lod, pos, size):
        """See base class for detils."""
        as_float = bool(self._dtype == np.float32)
        verbose  = False
        data = self._accessor.readConstantValue(
            pos, size, lod=lod, as_float=as_float, verbose=verbose)
        if data is not None:
            data = ScalarBuffer(size, data, self._dtype)
        else:
            data = np.zeros(size, dtype=self._dtype)
            self._accessor.readToExistingBuffer(
                data, pos, lod=lod, as_float=as_float, verbose=verbose)
        self._verbose("@{0}read(lod={1}, pos={2}, size={3}) => {4}".format(
            self._prefix(lod-1), lod, tuple(pos), tuple(size),
            self._format_result(data)))
        assert tuple(size) == tuple(data.shape)
        self._report(data)
        return data

    def _write(self, lod, pos, data):
        """See base class for detils."""
        self._verbose("@{0}write(lod={1}, pos={2}, data={3})".format(
            self._prefix(lod-1), lod, tuple(pos),
            self._format_result(data)))
        is_storage = bool(data.dtype != np.float32)
        verbose  = False
        if lod > 0:
            self._accessor._writeRegion(
                data, pos, lod=lod, compressor=self._compressor,
                is_storage=is_storage, verbose=verbose)
            self._report(data)

    def _savestats(self):
        #print("STATS", repr(self._stats), str(type(self._stats._min)))
        pass

def main(filename, snr):
    """
    Create or re-create all statistics, histogram, and low resolution data
    in the provided open ZGY file. This method is for testing only; normally
    class GenLodC will be invoked from inside the api layer as part of
    finalizing a newly created file. Normally the genlod module should not
    use the api layer itself. Also note that ZGY technically doesn't allow
    updating a file once it has been written and closed. A kludge makes
    it work, sort of, but not for seismic store and not for versions < 2.
    Also, only the lowres data is actually written to the file.
    Now that the new code has been hooked up to the rest of OpenZGY there
    isn't really much need for this stand alone test; the code should be
    covered better by the regular unit tests.
    """
    from ..api import ZgyReader, ZgyCompressFactory, ProgressWithDots
    from ..test.utils import SDCredentials
    with ZgyReader(filename, iocontext=SDCredentials(), _update=True) as reader:
        GenLodC(accessor   = reader._accessor,
                progress   = ProgressWithDots(),
                verbose    = lambda *args, **kw: None, # print(*args, **kw),
                compressor = ZgyCompressFactory("ZFP", snr = snr))()

if __name__ == "__main__":
    """
    Stand alone app for testing the genlod module. See main() for details.
    """
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description="Generate and store low resolution bricks, histogram, and statistics. See doc/lowres.html for details.")
    parser.add_argument('input', help='ZGY input cube, local or sd://')
    parser.add_argument('--snr', default=0, type=int,
                        help='Pass 10..70 for lossy compression, 99 for lossless, omit for uncompressed.')
    args = parser.parse_args()
    if not args.input:
        print("File names cannot be empty.", file=sys.stderr)
        sys.exit(1)
    main(args.input, args.snr)

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
