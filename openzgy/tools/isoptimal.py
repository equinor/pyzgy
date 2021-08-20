#!/usr/bin/env python3

"""
Analyze one of more ZGY files to determine how bricks are ordered on disk.
Ideally the ordering is strictly one of the 6 possible permutations.
For each possibility the program will calculate how close the file is
to the goal. It will also suggest the access ordering the applications
should use for best results.

Unfortunately the check can be rather expensive for large files.
So it might not be practical to offer this as a method applications
can call at runtime. Alternatively the code might be changed to just
analyze part of the file and hope the same ordering is used for the
entire file.
"""

import numpy as np
import os, sys, math
from collections import namedtuple
from ..api import ZgyReader
from ..test.utils import SDCredentials
from ..impl import enum as impl_enum

_statisticsType = namedtuple("Statistics", "perfect partial noncont")

def iterate_InlineCrosslineSlice(reader, lod):
    lodsize = reader.brickcount[lod]
    for ii in range(lodsize[0]):
        for jj in range(lodsize[1]):
            for kk in range(lodsize[2]):
                yield (lod, ii, jj, kk)

def iterate_SliceInlineCrossline(reader, lod):
    lodsize = reader.brickcount[lod]
    for kk in range(lodsize[2]):
        for ii in range(lodsize[0]):
            for jj in range(lodsize[1]):
                yield (lod, ii, jj, kk)

def iterate_CrosslineSliceInline(reader, lod):
    lodsize = reader.brickcount[lod]
    for jj in range(lodsize[1]):
        for kk in range(lodsize[2]):
            for ii in range(lodsize[0]):
                yield (lod, ii, jj, kk)

def iterate_CrosslineInlineSlice(reader, lod):
    lodsize = reader.brickcount[lod]
    for jj in range(lodsize[1]):
        for ii in range(lodsize[0]):
            for kk in range(lodsize[2]):
                yield (lod, ii, jj, kk)

def iterate_InlineSliceCrossline(reader, lod):
    lodsize = reader.brickcount[lod]
    for ii in range(lodsize[0]):
        for kk in range(lodsize[2]):
            for jj in range(lodsize[1]):
                yield (lod, ii, jj, kk)

def iterate_SliceCrosslineInline(reader, lod):
    lodsize = reader.brickcount[lod]
    for kk in range(lodsize[2]):
        for jj in range(lodsize[1]):
            for ii in range(lodsize[0]):
                yield (lod, ii, jj, kk)

orderings = [
    ("InlineCrosslineSlice", iterate_InlineCrosslineSlice),
    ("SliceInlineCrossline", iterate_SliceInlineCrossline),
    ("CrosslineSliceInline", iterate_CrosslineSliceInline),
    ("CrosslineInlineSlice", iterate_CrosslineInlineSlice),
    ("InlineSliceCrossline", iterate_InlineSliceCrossline),
    ("SliceCrosslineInline", iterate_SliceCrosslineInline),
]

def isContiguous(reader, work, maxhole):
    """
    A single brick-column is considered to be fully optimized if bricks
    are contiguous and partly optimized if the gap between any two bricks
    is small. It is not optimized if any gap is negative, because the
    logic that consolidates bricks might refuse to handle that case.

    Note that sum(perfect, partial, noncont) will be one less than the actual
    number of bricks since the method only deals with gaps between bricks.
    """
    perfect, partial, noncont, totalsize, waste = (0, 0, 0, 0, 0)
    prev = None
    for lod, ii, jj, kk in work:
        (brickstatus, fileoffset, constvalue, bricksize) = reader._accessor._getBrickFilePosition(ii, jj, kk, lod)
        if brickstatus in (impl_enum.BrickStatus.Compressed,
                           impl_enum.BrickStatus.Normal):
            totalsize += bricksize
            if prev is not None:
                delta = fileoffset - prev
                if delta == 0:
                    perfect += 1
                elif delta > 0 and delta <= maxhole:
                    partial += 1
                    waste += delta
                else:
                    noncont += 1
            prev = fileoffset + bricksize
    return perfect, partial, noncont, totalsize, waste

def run_one_ordering(reader, lods, maxhole, ordering):
    """
    Run isContiguous() for a range of lods and sum the results.
    Also convert the result to a named tuple.
    """
    count_perfect = 0
    count_partial = 0
    count_noncont = 0
    for lod in lods:
        work = ordering(reader, lod)
        perfect, partial, noncont, _, _ = isContiguous(reader, work, maxhole)
        count_perfect += perfect
        count_partial += partial
        count_noncont += noncont
    return _statisticsType(count_perfect, count_partial, count_noncont)

def show_one_ordering(filename, reader, maxhole, ordername, ordering):
    """
    For a single ordering, e.g. InlineCrosslineSlice, check both the full
    resolution and the low resolution bricks to see how many are contiguous.
    Bricks that are less than maxhole distant count as 50% contiguous.
    Return the result as a human readable string. Also return the score
    as a number, to make it possible to pick the access order that gives
    the best result.
    """
    lod0 = run_one_ordering(reader, [0], maxhole, ordering)
    lodN = run_one_ordering(reader, range(1, reader.nlods), maxhole, ordering)
    total = (lod0.perfect + lod0.partial + lod0.noncont +
             lodN.perfect + lodN.partial + lodN.noncont)
    score = (total -
             (lod0.partial + lodN.partial) / 2 -
             (lod0.noncont + lodN.noncont)) * (100.0 / total)
    if score < 95.0:
        overall = "Suboptimal:"
    elif lod0.partial or lod0.noncont or lodN.partial or lodN.noncont:
        overall = "Acceptable:"
    else:
        overall = "Optimized: "
    if True or lod0.partial or lod0.noncont or lodN.partial or lodN.noncont:
        fmt = ("{overall} " +
               "contig {lod0.perfect:4d} " +
               "partial {lod0.partial:4d} " +
               "noncont {lod0.noncont:4d} " +
               "lowres: " +
               "contig {lodN.perfect:4d} " +
               "partial {lodN.partial:4d} " +
               "noncont {lodN.noncont:4d} " +
               "score {score:6.2f} "
               "{name} {ordername}")
    else:
        fmt = ("{overall} " +
               " size {lod0.perfect}+{lodN.perfect} bricks {name}")
    message = fmt.format(name=filename, overall=overall, lod0=lod0, lodN=lodN, score=score, ordername=ordername)
    return score, ordername, ordering, message

def show_6_orderings(filename, reader, maxhole):
    """
    Try all 6 access orders and see which of them gives the highest number
    of contiguous bricks.
    """
    results = []
    for ordername, ordering in orderings:
        result = show_one_ordering(filename, reader, maxhole, ordername, ordering)
        results.append(result)
        print(result[3])
    best = sorted(results)[-1]
    print("Recommended access order {0} score {1:6.2f} for {2}".format(best[1], best[0], filename))
    return best[1]

def run(filename, maxhole = 2*1024*1024):
    with ZgyReader(filename, iocontext = SDCredentials()) as reader:
        show_6_orderings(filename, reader, maxhole)

if __name__ == "__main__":
    np.seterr(all='raise')
    for filename in sys.argv[1:]:
        run(filename)
    #import cProfile
    #cProfile.run('run(sys.argv[1])', sort="cumulative")

# To optimize a file using the old tools:
#   env SALMON_LOD_AUTOFLUSH=0 zgycopy -v -b 64,1024,0 IN OUT
# To un-optimize it completely use --random.
# There is currently no simple way to optimize for a different ordering.

# Copyright 2017-2021, Schlumberger
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
