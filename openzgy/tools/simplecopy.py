#!/usr/bin/env python3

"""
Implement a simple copy command that exercises the new code.
"""

import numpy as np
import os, sys
from ..api import ZgyReader, ZgyWriter, ProgressWithDots, SampleDataType
from ..test.utils import SDCredentials
from ..iterator import readall

def suggest_range(value, dt):
    """Special case handling for inconsistent all-constant files."""
    dt_lo, dt_hi = {SampleDataType.int8: (-128, +127),
                    SampleDataType.int16: (-32768, +32767)}[dt]
    if value == 0:
        return (dt_lo / dt_hi, 1)
    elif value > 0:
        return (0, value * (1 - dt_hi / dt_lo))
    else:
        return (value * (1 - dt_lo / dt_hi), 0)

def copy_open_file(r, w, progress):
    """Simple example of manually iterating over files."""
    def _roundup(x, step): return ((x + step - 1) // step) * step
    blocksize = (r.bricksize[0], r.bricksize[1],
                 _roundup(r.size[2], r.bricksize[2]))
    total = (((r.size[0] + blocksize[0] - 1) // blocksize[0]) *
             ((r.size[1] + blocksize[1] - 1) // blocksize[1]))
    done = 0
    data = np.zeros(blocksize, dtype=np.float32)
    for ii in range(0, r.size[0], blocksize[0]):
        for jj in range(0, r.size[1], blocksize[1]):
            r.read((ii, jj, 0), data)
            w.write((ii, jj, 0), data)
            done += 1
            if progress: progress(done, total)

def copy_open_v2(r, w, progress):
    """Fewer lines of code but more going on behind the scenes."""
    for datastart, datasize, data in readall(r, progress=progress):
        w.write(datastart, data)

def copy(srcfilename, dstfilename):
    with ZgyReader(srcfilename, iocontext = SDCredentials()) as r:
        if r.datatype in (SampleDataType.int8, SampleDataType.int16):
            if r.raw_datarange[0] == r.raw_datarange[1]:
                datarange = suggest_range(r.raw_datarange[0], r.datatype)
                with ZgyWriter(dstfilename, templatename=srcfilename,
                               datarange = datarange,
                               iocontext = SDCredentials()) as w:
                    w.writeconst((0,0,0), r.raw_datarange[0], w.size, False)
                    w.finalize(progress=ProgressWithDots())
                    return
        with ZgyWriter(dstfilename, templatename=srcfilename,
                       iocontext = SDCredentials()) as w:
            copy_open_file(r, w, progress=ProgressWithDots())
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
