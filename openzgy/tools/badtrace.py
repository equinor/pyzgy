#!/usr/bin/env python3

"""
Detect corrupt traces by looking for any trace having a sample magnitudes.
that is more than 10 times the 99-percentile of all magnitudes. If any are
found, set the entire trace to zero. Might also have considered clipping
to the 99-percentile instead.

To avoid running out of memory the 99-percentile is computed from max
magnitude of each trace, not of each sample. I guess I could save even
more by removing the bottom 99% of the collected & sorted values, ever
so often. Note: It would be nice to include his functionality in test_crop.
"""

import numpy as np
import os, sys, math
from ..api import ZgyReader, ZgyWriter, ProgressWithDots
from ..test.utils import SDCredentials
from ..iterator import readall

def round_sig(x, sig=2):
    return 0 if not x else round(x, sig-1-int(math.floor(math.log10(abs(x)))))

def scan_open_file(r, progress):
    limits = []
    for datastart, datasize, data in readall(r, progress=progress):
        lim = np.amax(np.abs(data), axis=2)
        limits.extend(map(float, lim.flat))
    limits.sort()
    cutoff = limits[(len(limits)*99)//100]
    print("Trace count: {0:,d}, 99-percentile: {1:.6g}".format(
        len(limits), cutoff))
    return round_sig(cutoff * 10)

def copy_open_file(r, w, cutoff, progress):
    nuked = 0
    total = 0
    for datastart, datasize, data in readall(r, progress=progress):
        lim = np.amax(np.abs(data), axis=2)
        for ii in range(lim.shape[0]):
            for jj in range(lim.shape[1]):
                total += 1
                if lim[ii,jj] > cutoff:
                    nuked += 1
                    data[ii,jj,:] = 0
                    if False:
                        print("trace {0},{1} max {2:.6g}".format(
                            datastart[0]+ii, datastart[1]+jj, lim[ii,jj]))
        w.write(datastart, data)
    print("{0:,d} of {1:,d} traces spiked > {2:.6g} and were zeroed.".format(
        nuked, total, cutoff))

def copy(srcfilename, dstfilename):
    ProgressWithDots()(100, 100)
    with ZgyReader(srcfilename, iocontext = SDCredentials()) as r:
        with ZgyWriter(dstfilename, templatename=srcfilename,
                       iocontext = SDCredentials()) as w:
            cutoff = scan_open_file(r, progress=ProgressWithDots())
            copy_open_file(r, w, cutoff, progress=ProgressWithDots())
            w.finalize(progress=ProgressWithDots())

if __name__ == "__main__":
    np.seterr(all='raise')
    copy(sys.argv[1], sys.argv[2])

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
