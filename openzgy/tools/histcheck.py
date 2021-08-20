#!/usr/bin/env python3

print('Running' if __name__ == '__main__' else 'Importing', __file__)

import numpy as np
import sys
from time import time

# The new, pure Python API
from ..api import ZgyReader
# The old Python wrapper on top of C++ ZGY-Public
#from zgy import ZgyReader
from ..iterator import readall

def scanForRange(reader, verbose = False):
    realmin = []
    realmax = []
    begtime = time()
    for datastart, datasize, brick in readall(reader, dtype=np.float32):
        realmin.append(np.nanmin(brick))
        realmax.append(np.nanmax(brick))
    valuerange = (np.nanmin(realmin), np.nanmax(realmax))
    print("VALUE RANGE", valuerange)
    elapsed = time() - begtime
    voxels = np.product(reader.size) / (1024*1024)
    print("  {0:.1f} MVoxel read in {1:.1f} sec, {2:.1f} Mvoxel/s".format(
        voxels, elapsed, voxels/elapsed))
    return valuerange

def scanForHistogram(reader, valuerange, verbose = False):
    """
    Generate a histogram for all samples on the file.
    Note that if you want to verify the histogram on the file,
    the saved histogram range gives you the center of the first
    and last range but numpy wants the outer edges.
    """
    hh = None
    begtime = time()
    for datastart, datasize, brick in readall(reader, dtype=np.float32):
        h = np.histogram(brick, bins=256, range=valuerange)
        if hh is None:
            hh = h[0]
        else:
            hh += h[0]
    elapsed = time() - begtime
    voxels = np.product(reader.size) / (1024*1024)
    print("  {0:.1f} MVoxel read in {1:.1f} sec, {2:.1f} Mvoxel/s".format(
        voxels, elapsed, voxels/elapsed))
    return hh

def verifyHistogram(reader, verbose = False):
    h = reader.histogram
    binwidth = (h.max - h.min) / 255.0
    nprange = (h.min - binwidth/2, h.max + binwidth/2)
    hh = scanForHistogram(reader, nprange, verbose=verbose)
    stored = np.array(h.bin)
    delta = hh - stored
    #print("STORED", stored, sep="\n")
    #print("COMPUTED", hh, sep="\n")
    #print("COMPARED", delta, sep="\n")
    mismatch = np.array([(x, delta[x]) for x in range(len(delta)) if delta[x]])
    if len(mismatch):
        print("MISMATCH (bin, count) =", mismatch)
    else:
        print("HISTOGRAM CORRECT")

if __name__ == "__main__":

    np.seterr(all='raise')

    if len(sys.argv) <= 1:
        args = [ "/home/paal/git/Salmon/UnitTestData/Salmon/UnitTest/Salt2-v3.zgy" ]
    else:
        args = sys.argv[1:]

    for filename in args:
        with ZgyReader(filename) as reader:
            #reader.dump()
            print("size", reader.size, "datatype", reader.datatype)
            valuerange = scanForRange(reader, verbose=True)
            verifyHistogram(reader, verbose=True)

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
