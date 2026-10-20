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
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor

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
        iterations += np.prod((ss+bs-1) // bs)
        ss = (ss + 1) // 2
    bytesused = np.sum(np.prod(blocksizes, axis=1)) * int(np.dtype(dtype).itemsize)
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

def _ordered_map(func, items, nthreads):
    """Yield func(item) for each item, in order. At most 2*nthreads items are in flight."""
    with ThreadPoolExecutor(nthreads) as pool:
        pending = deque()
        try:
            for item in items:
                pending.append(pool.submit(func, item))
                if len(pending) >= 2 * nthreads:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for f in pending:
                f.cancel()

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
            _total += np.prod((sz + bs - 1) // bs)
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
            count = np.prod((np.array(data.shape, dtype=np.int64) +
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
    def __init__(self, *args, num_threads = 1, **kwargs):
        super().__init__(*args, **kwargs)
        #print("GenLodImpl", args, kwargs)
        self._num_threads = num_threads
        self._stats = StatisticData()
        self._histo = HistogramData(range_hint=self._range_hint,
                                    dtype=self._dtype)

    def __call__(self):
        """
        Generate and store statistics, histogram, and all low resolution
        bricks. The lod0 blocks are summarized in parallel, and their
        statistics are merged in depth first order. Then each lod1 frame is
        calculated with its four lod0 blocks, in parallel. The levels from 2
        up, and all writes, are done by the calling thread in the same order
        as the recursive calculation this replaces
        """
        leaves, last = self._traverse()
        self._last = last
        # cum[k] is the histogram after leaves[0..k], needed for weighted average.
        self._cum = np.empty((len(leaves), self._histo._bins.size), dtype=np.int64)
        for k, (data, s, bins) in enumerate(_ordered_map(self._leaf, leaves, self._num_threads)):
            self._report(data)
            self._stats._add_other(s)
            self._histo._bins += bins
            self._cum[k] = self._histo._bins
        if self._nlods > 1:
            frames = [pos for (lod, pos) in last if lod == 1]
            subtrees = _ordered_map(self._subtree, frames, self._num_threads)
            self._combine((0, 0, 0), self._nlods - 1, subtrees)
        self._savestats() # Remove?
        return self._stats, self._histo

    def _accumulate(self, stats, histo, data):
        """
        Keep a running tally of statistics and histogram.
        """
        if data is None:
            return
        factor = 1
        if isinstance(data, ScalarBuffer):
            factor = np.prod(data.shape, dtype=np.int64)
            data = np.array([data.value], dtype=data.dtype)
        stats.add(data, factor)
        histo.add(data, factor)

    def _children(self, pos, lod):
        """The four frames one level below pos, or None where outside the survey."""
        bs = self._bricksize
        ss = (self._surveysize + (1 << (lod-1)) - 1) >> (lod-1)
        result = []
        for oi, oj in ((0, 0), (0, 1), (1, 0), (1, 1)):
            child = (2*pos[0] + 2*oi*bs[0], 2*pos[1] + 2*oj*bs[1], 0)
            inside = child[0] < ss[0] and child[1] < ss[1]
            result.append(child if inside else None)
        return result

    def _traverse(self):
        """
        Visit the lod0 blocks depth first. Returns (leaves, last): the lod0
        block positions in order, and for each frame (lod, pos) the index of
        the last leaf in its subtree.
        """
        leaves, last = [], {}
        def visit(pos, lod):
            if lod == 0:
                leaves.append(pos)
            else:
                for child in self._children(pos, lod):
                    if child is not None:
                        visit(child, lod-1)
            last[(lod, pos)] = len(leaves) - 1
        visit((0, 0, 0), self._nlods - 1)
        return leaves, last

    def _block_size(self, lod, pos):
        """Size of the block of 2x2 bricks at this position, full trace length."""
        ss = (self._surveysize + (1 << lod) - 1) >> lod
        bs2 = 2 * self._bricksize
        return np.array((min(bs2[0], ss[0] - pos[0]), min(bs2[1], ss[1] - pos[1]), ss[2]),
                        dtype=np.int64)

    def _snapshot(self, index):
        """The histogram as it was after leaves[0..index]."""
        histo = HistogramData(range_hint=self._range_hint, dtype=self._dtype)
        histo._bins = self._cum[index]
        return histo

    def _leaf(self, pos):
        """Read one lod0 block and summarize it. Runs in a worker thread."""
        pos = np.array(pos, dtype=np.int64)
        data = self._read(0, pos, self._block_size(0, pos))
        stats = StatisticData()
        histo = HistogramData(range_hint=self._range_hint, dtype=self._dtype)
        self._accumulate(stats, histo, data)
        return data, stats, histo._bins

    def _subtree(self, pos):
        """
        Calculate one lod1 frame from its four lod0 blocks. Runs in a worker
        thread. Returns the lod1 data, which the caller writes, and the lod2
        decimation of it, which the parent frame needs.
        """
        hires = []
        for leaf in self._children(pos, 1):
            if leaf is None:
                hires.append(None)
            else:
                data = self._read(0, np.array(leaf, dtype=np.int64), self._block_size(0, leaf))
                hires.append(self._decimate(data, 1, self._snapshot(self._last[(0, leaf)])))
        data = self._paste4(*hires)
        result = None
        if self._nlods > 2:
            result = self._decimate(data, 2, self._snapshot(self._last[(1, pos)]))
        return data, result

    def _combine(self, pos, lod, subtrees):
        """
        Calculate and write one frame at lod 1 or above. The lod1 frames come
        from subtrees, in the same depth first order as the recursion.
        """
        if lod == 1:
            data, result = next(subtrees)
            self._write(1, np.array(pos, dtype=np.int64), data)
            return result
        hires = []
        for child in self._children(pos, lod):
            hires.append(None if child is None else self._combine(child, lod-1, subtrees))
        data = self._paste4(*hires)
        self._write(lod, np.array(pos, dtype=np.int64), data)
        if lod == self._nlods - 1:
            return None
        return self._decimate(data, lod+1, self._snapshot(self._last[(lod, pos)]))

    def _decimate(self, data, lod, histogram):
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
                                histogram = histogram,
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
                 progress = None, verbose = None, num_threads = 1):
        # Could add defaultvalue. And add a unit test for correct operation.
        super().__init__(size = accessor._metadata._ih._size,
                         bricksize = accessor._metadata._ih._bricksize,
                         dtype = impl_enum._map_DataTypeToNumpyType(
                             accessor._metadata._ih._datatype),
                         range_hint = (accessor._sample_min,
                                       accessor._sample_max),
                         nlods = accessor._metadata._ih._nlods,
                         decimation = decimation,
                         progress = progress, verbose = verbose,
                         num_threads = num_threads)
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
