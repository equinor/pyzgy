#!/usr/bin/env python3

"""
Test the fancy iterator.
"""

import sys
import numpy as np

from ..api import ProgressWithDots, ZgyReader, ZgyWriter, SampleDataType
from ..test.utils import SDCredentials
from ..iterator import readall

def _test_consume(iterator):
    ncalls = 0
    nvoxels = 0
    maxconsumed = [0, 0, 0]
    for pos, count, data in iterator:
        if tuple(count) != data.shape: return
        ncalls += 1
        nvoxels += int(data.size)
        for i in range(3):
            maxconsumed[i] = max(maxconsumed[i], data.shape[i])
    return ncalls, nvoxels, tuple(maxconsumed)

def test_iterator():

    class MockReader:
        def __init__(self):
            self.readcount = 0
            self.voxelcount = 0
            self.maxread = [0, 0, 0]
        def read(self, pos, data, *, lod=0):
            if False:
                print("read", tuple(pos), tuple(data.shape), lod)
            self.readcount += 1
            self.voxelcount += data.size
            for i in range(3):
                self.maxread[i] = max(self.maxread[i], data.shape[i])
        @property
        def size(self):
            return (201, 334, 1001)
        @property
        def bricksize(self):
            return (64, 64, 64)
        @property
        def datatype(self):
            return SampleDataType.int16
        @property
        def result(self):
            return self.readcount, self.voxelcount, tuple(self.maxread)

    class CountWork:
        def __init__(self, abort = None):
            self.count = 0
            self.done = 0
            self.total = 0
            self._abort = abort
        def __call__(self, done, total):
            self.count += 1
            self.done = done
            self.total = total
            return not self._abort or self.count < self._abort
        def __str__(self):
            return "done {0}/{1} in {2} calls".format(
                self.done, self.total, self.count)

    ssize = 201 * 334 * 1001
    ssize_lod1 = (202//2) * (334//2) * (1002//2)

    # For the first set of tests, leave chunksize unset so the
    # consumer gets called once for each data read.

    # If blocksize is set it will be honored, and zero means all.
    mock = MockReader()
    tmp = readall(mock, blocksize = (0, 0, 0))
    assert mock.readcount == 0
    assert _test_consume(tmp) == (1, ssize, tuple(mock.size))
    assert mock.result        == (1, ssize, tuple(mock.size))

    # Try a reasonable blocksize
    mock = MockReader()
    tmp = readall(mock, blocksize = (128, 128, 0))
    assert _test_consume(tmp) == (6, ssize, (128, 128, 1001))
    assert mock.result        == (6, ssize, (128, 128, 1001))

    # Try a silly blocksize
    mock = MockReader()
    tmp = readall(mock, blocksize = (0, 42, 501))
    assert _test_consume(tmp) == (16, ssize, (201, 42, 501))
    assert mock.result        == (16, ssize, (201, 42, 501))

    # If blocksize is not set, should end up reading the entire file
    # because it is less than 1 GB
    mock = MockReader()
    tmp = readall(mock, maxbytes=1024*1024*1024)
    assert _test_consume(tmp) == (1, ssize, (201, 334, 1001))
    assert mock.result        == (1, ssize, (201, 334, 1001))

    # If blocksize is not set but max size only has room for 3 brick-columns
    # we will be reading more in the crossline direction.
    mock = MockReader()
    tmp = readall(mock, maxbytes = 3*2048*64*64)
    assert _test_consume(tmp) == (12, ssize, (64, 128, 1001))
    assert mock.result        == (12, ssize, (64, 128, 1001))

    # Setting the blocksize to zero means as little as possible,
    # which is one brick colums unless you want to be really inefficient.
    mock = MockReader()
    tmp = readall(mock, maxbytes = 0)
    assert _test_consume(tmp) == (24, ssize, (64, 64, 1001))
    assert mock.result        == (24, ssize, (64, 64, 1001))

    # Now test setting the chunksize, leaving blocksize unset
    # causing it to read our small data in just one read.
    mock = MockReader()
    tmp = readall(mock, chunksize = (64, 64, 64), maxbytes=1024*1024*1024)
    assert _test_consume(tmp) == (4*6*16, ssize, (64, 64, 64))
    assert mock.result        == (1, ssize, tuple(mock.size))

    # As above but set a silly chunksize.
    mock = MockReader()
    tmp = readall(mock, chunksize = (100, 42, 0), maxbytes=1024*1024*1024)
    assert _test_consume(tmp) == (24, ssize, (100, 42, 1001))
    assert mock.result        == (1, ssize, tuple(mock.size))

    # Setting both.
    mock = MockReader()
    tmp = readall(mock, blocksize = (128, 128, 0), chunksize = (64, 64, 0))
    assert _test_consume(tmp) == (24, ssize, (64, 64, 1001))
    assert mock.result        == (6, ssize, (128, 128, 1001))

    # Setting chunksize to more than blocksize has no effect.
    mock = MockReader()
    tmp = readall(mock, blocksize = (64, 64, 0), chunksize = (128, 199, 0))
    assert _test_consume(tmp) == (24, ssize, (64, 64, 1001))
    assert mock.result        == (24, ssize, (64, 64, 1001))

    # A dry run may be used to get the "total" argument to the progress
    # report without actually doing any work. Pass a dummy progress
    # that saves the total and then returns False to abort.
    # Make user that (1) the iterator is consumed, so total can be saved,
    # and (2) when the iterator is consumed it doesn't actually read.
    mock = MockReader()
    progress = CountWork(abort=1)
    tmp = readall(mock, chunksize = (100, 42, 0), progress=progress)
    assert mock.readcount == 0
    assert progress.total == 0 # Because iterator not consumed yet.
    assert _test_consume(tmp) == (0, 0, (0, 0, 0))
    assert mock.result        == (0, 0, (0, 0, 0))
    assert mock.readcount == 0
    assert progress.total == ssize

    # The same run, not dry run this time, to test the progress report.
    mock = MockReader()
    progress = CountWork()
    tmp = readall(mock, chunksize = (100, 42, 0), maxbytes=1024*1024*1024, progress=progress)
    assert _test_consume(tmp) == (24, ssize, (100, 42, 1001))
    assert mock.result        == (1, ssize, tuple(mock.size))
    #print(progress)
    assert progress.count == 26 # one initial, one final, 24 regular.
    assert progress.done == ssize
    assert progress.total == ssize

    # Decimated data with silly chunksize.
    mock = MockReader()
    tmp = readall(mock, chunksize = (100, 42, 0), lod=1)
    assert _test_consume(tmp) == (8, ssize_lod1, (100, 42, 501))
    assert mock.result        == (1, ssize_lod1, (101, 167, 501))

def copyall(srcfilename, dstfilename, *, maxbytes=128*1024*1024):
    """
    Simple test that exercises the fancy iterator.
    Sets chunksize to a silly value to stress the code more.
    This is really inefficient and triggers read/modify/write
    in the writer.
    """
    p1, p2 = (ProgressWithDots(), ProgressWithDots())
    with ZgyReader(srcfilename, iocontext = SDCredentials()) as r:
        with ZgyWriter(dstfilename, templatename=srcfilename,
                       iocontext = SDCredentials()) as w:
            alldata = readall(r, maxbytes=maxbytes, progress=p1,
                              chunksize=(100, 100, 0))
            for datastart, datasize, data in alldata:
                w.write(datastart, data)
            w.finalize(progress=p2)

if __name__ == "__main__":

    test_iterator()

    if len(sys.argv) > 1:
        copyall(sys.argv[1],
                sys.argv[2],
                maxbytes = int(sys.argv[3]) if len(sys.argv) > 3 else None)

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
