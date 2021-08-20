#!/usr/bin/env python3

#print('Running' if __name__ == '__main__' else 'Importing', __file__)

import numpy as np
import sys
import os
import time
from PIL import Image

from .. import api as newzgy
try:
    from .. import zgypublic as oldzgy
except Exception:
    oldzgy = None
from ..exception import ZgyMissingFeature
from ..test.utils import SDCredentials, SDTestData
from .viewzgy import savePNG

def read_data_all_at_once(reader, lod, start, size):
    section = np.zeros(size, dtype=np.float32)
    reader.read(start, section, lod = lod)
    return section

def read_data_b_at_a_time(reader, lod, start, size):
    bs = np.array(reader.bricksize, dtype=np.int64)
    padsize = ((np.array(size, np.int64) + bs - 1) // bs) * bs
    brick = np.zeros(bs, dtype=np.float32)
    section = np.zeros(padsize, dtype=np.float32)
    for ii in range(0, size[0], bs[0]):
        for jj in range(0, size[1], bs[1]):
            for kk in range(0, size[2], bs[2]):
                reader.read((start[0]+ii, start[1]+jj, start[2]+kk),brick, lod = lod)
                section[ii:ii+bs[0], jj:jj+bs[1], kk:kk+bs[2]] = brick
    return section[:size[0],:size[1],:size[2]]

def timing_report(reader, lod, start, size, elapsed):
    bs = np.array(reader.bricksize if isinstance(reader, newzgy.ZgyReader) else (64, 64, 64), dtype=np.int64)
    padsize = ((np.array(size, np.int64) + bs - 1) // bs) * bs
    bandwidth = np.product(padsize) / elapsed # should I use size or padsize?
    bandwidth /= (1024*1024)
    print("Elapsed {0:6.2f} seconds, bandwidth {1:6.2f} MVoxel/s reading {2} lod {3} size {4} start {5}".format(elapsed, bandwidth, reader.datatype, lod, tuple(size), tuple(start)))

_zgycloud_inited = False

def init_zgycloud():
    global _zgycloud_inited
    if _zgycloud_inited or oldzgy is None: return
    import zgycloud
    zgy = oldzgy.zgy
    #print("Glue together error handling", flush=True)
    zgy.setErrorHooks(zgycloud.resetError, zgycloud.lastError)
    #print("Handle logging inside Python", flush=True)
    zgycloud.setLogger(lambda pri,x: print("log:", x, end='', flush=True), 0)
    # Modify this to run the tests in a different environment.
    if False:
        zgycloud.configure('sd://', '',
                           '@/etc/delfi/demo-sdapi-key',
                           '@/etc/delfi/demo-sdapi-url')
    #print("Get test token", flush=True)
    token = zgycloud.getTestToken("sd://")
    zgycloud.setToken("sd://", token, "stoken", True)
    zgycloud.enableRealCache(True, 1024)
    zgycloud.setSegmentSize(1024)
    _zgycloud_inited = True

def run(filename, *, lods = [0], direction = 0, slurp=True,
        readerfactory = newzgy.ZgyReader, outname = None, iocontext=None):
    """
    Read 64 traces or slices in the specified direction.
    Optionally save the first of these as PNG.
    """
    if iocontext is None and filename[:5] == "sd://":
        iocontext = SDCredentials()
    if False:
        print("Read", filename,
              ("64 inlines", "64 crosslines", "64 slices")[direction],
              "slurp" if slurp else "block at a time")
    allslices = []
    with readerfactory(filename, iocontext=iocontext) as reader:
        slicenumber = reader.size[direction] // 2
        #reader.dump()
        for lod in lods:
            step = 1<<lod
            start = [0, 0, 0]
            size = np.array(reader.size, dtype=np.int64) // (1 << lod)
            start[direction] = slicenumber >> lod
            size[direction] = 1
            if direction == 2:
                size[0] = min(size[0], 1024)
                size[1] = min(size[1], 1024)
            start = tuple(start)
            size = tuple(size)
            starttime = time.time()
            if slurp:
                section = read_data_all_at_once(reader, lod, start, size)
            else:
                section = read_data_b_at_a_time(reader, lod, start, size)
            #timing_report(reader, lod, start, size, time.time() - starttime)
            # Display only the first section or slice
            if outname:
                if direction == 0:
                    myslice = section[0,...]
                elif direction == 1:
                    myslice = section[:,0,:]
                else:
                    myslice = section[...,0]
                #savePNG(myslice, outname + "_lod" + str(lod) + ".png")
                allslices.append(myslice)

    if outname:
        s = allslices[0].shape
        w = np.sum([allslices[i].shape[0] for i in range(3)])
        combined = np.zeros((w, s[1]), dtype=allslices[0].dtype)
        combined[0:s[0],:] = allslices[0]
        ss = allslices[1].shape
        combined[s[0]:s[0]+ss[0], 0:ss[1]] = allslices[1]
        sss = allslices[2].shape
        combined[s[0]+ss[0]:s[0]+ss[0]+sss[0], 0:sss[1]] = allslices[2]
        combined = combined[::-1,::]
        savePNG(combined, outname + ".png")

def Rema1000(filelist):
    """
    Specific test for the Rema 1000 solution: What is the best case
    and worst case effect of using 4 MB block size regardless of how
    the data is layed out?
    The file list should be one completely sorted, one completely random

    In all cases, read one brick at a time, strictly ordered by
    inline slowest, crossline, vertical fastest
    Currently the cache only holds the last block read, but
    in these lab conditions that should be enough.

    Control: config aligned = 0. Access direction and sorted or not
    shouldn't have any effect. If it does then there is probably
    some caching that I am not aware of, and it will be more
    difficult to interpret the results. Test all combinations of
    unsorted/sorted and access directions.

    Test: config aligned = 4 MB. What I expect / hope is that for a
    sorted file the inline- and crossline access should be
    significantly better. With inline access possibly a bit better
    than crossline. Slice access and all access to scrambled files
    should all be similar (assuming the controls were all similar)
    and hopefully not very much worse than the control.

    The entire test suite should be run both for 8-bit and float.
    Primarily test on cloud, but perhaps also on-prem with limited
    bandwidth.
    """
    for alignment in [0, 4]:
        print("Force alignment:", alignment)
        for filename in filelist:
            for direction in (0, 1, 2):
                run(filename, direction=direction, slurp=False,
                    readerfactory=newzgy.ZgyReader,
                    iocontext=SDCredentials(aligned=alignment))
    # For comparison, best case for the fancy reader.
    run(filelist[0], direction=0, slurp=True, readerfactory=newzgy.ZgyReader)


def Main(args):

    if not args:
        args = ["../build/testdata/Empty-v3.zgy",
                "../build/testdata/Empty-v1.zgy",
                SDTestData("SyntFixed.zgy")]

    if any([name[:5] == "sd://" for name in args]):
        init_zgycloud()

    suffix = 1
    for filename in args:
        out = os.path.join(os.getenv("TESTRUNDIR", '.'), "new" + str(suffix))
        try:
            run(filename, lods=range(3), direction=0, slurp=True,
                readerfactory=newzgy.ZgyReader, outname = out)
        except ZgyMissingFeature as ex:
            print("{0}: {1}".format(filename, str(ex)))

        out = os.path.join(os.getenv("TESTRUNDIR", '.'), "old" + str(suffix))
        if oldzgy is not None:
            run(filename, lods=range(3), direction=0, slurp=True,
                readerfactory=oldzgy.ZgyReader, outname = out)

        suffix += 1

if __name__ == "__main__":
    np.seterr(all='raise')
    Main(sys.argv[1:])
    sys.exit(0)
    Rema1000([SDTestData("bigtestdata/synt-50gb-8bit-sorted.zgy"),
              SDTestData("bigtestdata/synt-50gb-8bit-randomized.zgy")])

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
