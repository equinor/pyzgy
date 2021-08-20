#!/usr/bin/env python3

"""
White-box unit tests for OpenZGY.
"""

import numpy as np
import os
import io
import sys

from ..impl import file as impl_file
from ..exception import *
from ..test.utils import SDCredentials, LocalFileAutoDelete, CloudFileAutoDelete, HasSeismicStore, SDTestData, SDTestSink

def checkThrows(e, msg, fn):
    try:
        fn()
        assert False and "Should have gotten an exception here."
    except e or ZgyUserError as ex:
        if msg:
            if str(ex).find(msg) < 0:
                print('Expect "{0}" got "{1}"'.format(msg, str(ex)))
            assert str(ex).find(msg) >= 0 and "Wrong exception message"
    except AssertionError:
        raise
    except Exception as ex:
        print('Expect "{0}" got "{1}"'.format(e or ZgyUserError, str(type(ex))))
        assert False and "Got the wrong type of exception."

class SimpleReceiver:
    def __call__(self, data, *args, **kwargs):
        self.data = bytes(data)

def _test_FileADT_read(f):
    """
    Check contents of a file created by test_SDFile.
    The test can be run both on a file descriptor still
    open for write and one recently opened read only.
    Also it works both for SD and local access.
    """
    # Simple interface
    data = f.xx_read(7, 9)
    assert data == b"continent"
    # Scatter/gather interface, a bit more roundabout to use.
    # Note: When reading back a SD file while it is still open
    # for write, the last (i.e. open) segment will be ' seg\n'.
    # This means that the part3 request spans the closed / open
    # segment boundary. Testing that corner case is important
    # because gthe ZGY library will encounter it very rarely and
    # only in conjunction with misaligned (compressed) files.
    part1 = SimpleReceiver()
    part2 = SimpleReceiver()
    part3 = SimpleReceiver()
    f.xx_readv([(1, 4, part1), (18, 7, part2), (60, 8, part3)])
    assert part1.data == b'ello'
    assert part2.data == b'this is'
    assert part3.data == b'last seg'
    assert f.xx_eof == 69
    checkThrows(None, "Invalid offset", lambda: f.xx_read(None, 1))
    checkThrows(None, "Invalid size", lambda: f.xx_read(0, None))
    checkThrows(None, "Invalid size", lambda: f.xx_read(0, 0))

def test_FileADT(filename):
    try:
        old_pread = os.pread if hasattr(os, "pread") else None
        with impl_file.FileFactory(filename, "w+b", "faux-context") as f:
            f.xx_write(b"hello, continent, THIS is a test.\n", 0)
            f.xx_write(b"hello again.\n", 34)
            f.xx_write(b"third block.\n", 47)
            f.xx_write(b"hello, continent, this is a test.\n", 0) # update
            f.xx_write(b"last sEG\n", 60)
            f.xx_write(b"eg", 66)
            checkThrows(None, "Invalid offset", lambda: f.xx_write("X", None))
            checkThrows(None, "Invalid offset", lambda: f.xx_write("X", -1))
            checkThrows(None, "Invalid size", lambda: f.xx_write(None, 0))
            checkThrows(None, "Invalid size", lambda: f.xx_write("", 0))
        with impl_file.FileFactory(filename, "rb", "faux-context") as f:
            assert f.threadsafe
            checkThrows(None, "open for writing", lambda: f.xx_write("X", 0))
            _test_FileADT_read(f)
        if old_pread: del os.pread
        with impl_file.FileFactory(filename, "rb", "faux-context") as f:
            assert not f.threadsafe
            _test_FileADT_read(f)
        checkThrows(None, "Opening ZGY as a is not supported.", lambda: impl_file.FileFactory(filename, "a", "faux-context"))
    finally:
        if old_pread: os.pread = old_pread

def test_SDFile(filename, buffered):
    with impl_file.FileFactory(filename, "w+b", SDCredentials(segsize=10/(1024*1024) if buffered else 0)) as f:
        assert not f.threadsafe
        assert f.xx_eof == 0
        f.xx_write(b"hello, continent, THIS is a test.\n", 0)
        f.xx_write(b"hello again.\n", 34)
        f.xx_write(b"third block.\n", 47)
        f.xx_write(b"hello, continent, this is a test.\n", 0) # update
        if not buffered:
            f.xx_write(b"last seg\n", 60)
        else:
            f.xx_write(b"last sEG\n", 60)
            f.xx_write(b"eg", 66) # writing to open segment
            # With segment size 10 and segment 0 unrestricted,
            # this is how the blocks should end up in storage.
            #   "hello, continent, THIS is a test.\n"
            #   "hello agai"
            #   "n.\nthird b"
            #   "lock.\nlast"
            #   " seg\n"
        assert f.xx_eof == 69
        checkThrows(ZgySegmentIsClosed, "write resized segment", lambda: f.xx_write(b"X", 34))
        checkThrows(ZgySegmentIsClosed, "write part of segment", lambda: f.xx_write(b"X", 35))
        checkThrows(ZgyUserError, "segments out of order", lambda: f.xx_write(b"X", 99))
        if not buffered:
            # When buffered this will be in the open segment, hence ok.
            checkThrows(ZgySegmentIsClosed, "write part of segment", lambda: f.xx_write(b"eg", 66))
            # The very ZGY specific limitation that all segments except
            # first and last must have the same size. Note that this will
            # not fail in the buffered test, because the buffering also
            # takes care of making all the segments the same size.
            checkThrows(ZgyUserError, "arbitrarily sized seg", lambda: f.xx_write(b"X", 69))
        # It should be possible to read back the file, even if some of it
        # is still sitting in the open segment. The test below also checks
        # reading across the closed / open segment boundary when buffered.
        _test_FileADT_read(f)
    with impl_file.FileFactory(filename, "rb", SDCredentials()) as f:
        assert f.threadsafe
        assert f.xx_eof == 69
        if not buffered:
            assert tuple(f._sizes) == (34, 13, 13, 9)
        else:
            assert tuple(f._sizes) == (34, 10, 10, 10, 5)
        _test_FileADT_read(f)
        # Test a read that crosses segment boundaries.
        slurp = f.xx_read(18, 45)
        assert slurp == b"this is a test.\nhello again.\nthird block.\nlas"
        # Test a read of a single, complete segment. There is a short cut.
        slurp = f.xx_read(34, 13)
        assert slurp == b"hello again.\n"
        # Test the readv interface in the same way.
        delivery = SimpleReceiver()
        f.xx_readv([(18, 45, delivery)])
        assert delivery.data == b"this is a test.\nhello again.\nthird block.\nlas"
        delivery = SimpleReceiver()
        f.xx_readv([(34, 13, delivery)])
        assert slurp == b"hello again.\n"

def test_SDFileReadPastEOF(filename):
    """
    Reading a block that crosses EOF should throw.
    """
    with impl_file.FileFactory(filename, "rb", SDCredentials()) as f:
        parts = list([SimpleReceiver() for i in range(8)])
        BS = 16 * 256 * 1024
        checkThrows(ZgyEndOfFile, None, lambda: (
            f.xx_readv([(0*BS, BS, parts[0]),
                        (1*BS, BS, parts[1]),
                        (2*BS, BS, parts[2]),
                        (3*BS, BS, parts[3])])
        ))
        checkThrows(ZgyEndOfFile, None, lambda: (
            f.xx_readv([(3*BS, BS, parts[7])])))

def test_Consolidate():
    """
    The logic for consolidating multiple requests into one is tricky
    enough to warrant a separate unit test.
    """
    class Want:
        def __init__(self, offset, size):
            if offset < 100000:
                # EOF is at 100000, we never get more than that.
                self._offset = offset
                self._size = min(offset + size, 100000) - offset
            else:
                # At or past EOF, we don't get any data at all.
                self._offset = None
                self._size = 0
            self.called = 0
        def __call__(self, data):
            self.called += 1
            data = np.frombuffer(data, dtype=np.uint8)
            if False:
                print("Delivery", (data[0], len(data)),
                      "expected", (self._offset, self._size),
                      "->", (self._offset % 251, self._size))
            assert self.called == 1
            assert len(data) == self._size
            if len(data):
                assert data[0] == self._offset % 251

    def test(requests, *args, **kwargs):
        # Add a functor that remembers what offset and size we ordered,
        # so that this can be checked when data gets delivered.
        requests = list([(offset, size, Want(offset, size)) for offset, size, _ in requests])
        # Invoke the function we are testing.
        result =  impl_file.FileADT._consolidate_requests(requests, *args, **kwargs)
        # Make a delivery of data where each element contains the absolute
        # offset of that element. Unless we pass our EOF marker.
        end_of_file = 100000
        for r in result:
            data = np.arange(r[0], min(r[0] + r[1], end_of_file), dtype=np.uint32)
            # Make it fit in a byte. Use a prime number modulus.
            data = (data % 251).astype(np.uint8)
            r[2](data.tobytes())
        # Check that each of our callbacks actually got called.
        for offset, size, fn in requests:
            assert fn.called == 1
        # Strip off the forward_result functor for easier checking of results.
        # Also convert to a tuple for the same reason.
        result = tuple([(e[0], e[1], None) for e in result])
        # Also strip functors off the input request, only for the below print()
        requests = tuple([(e[0], e[1], None) for e in requests])
        #print("in ", requests)
        #print("out", result)
        return result

    # Hole too large, no consolidation.
    requests = ((2, 98, None), (1000, 42, None))
    out = test(requests, max_hole = 300, max_size = None, force_align = None)
    assert out == requests

    # Same input, hole now deemed small enough.
    out = test(requests, max_hole = 1000, max_size = None, force_align = None)
    assert len(out) == 1
    assert out[0] == (2, 1040, None)

    # Same input, Hole too large, alignment was added.
    out = test(requests, max_hole = 300, max_size = None, force_align = 100)
    assert out == ((0, 100, None), (1000, 100, None))

    # Same input, hole now deemed small enough, alignment was added.
    out = test(requests, max_hole = 1000, max_size = None, force_align = 100)
    assert out[0] == (0, 1100, None)

    # max_size was exceeded.
    # Splitting is not very smart; see comments in source code.
    requests = list([(i, 70, None) for i in range(0, 66000, 100)])
    out = test(requests, max_hole = 300, max_size = 64000, force_align = None)
    assert out == ((0, 64000-30, None), (64000, 2000-30, None))

    # Requests larger than max_size
    requests = ((0, 1000, None), (1000, 2000, None), (3000, 500, None))
    out = test(requests, max_hole = 300, max_size = 10, force_align = None)
    assert out == requests

    # Request passes EOF
    requests = ((90000, 8000, None), # Inside
                (98000, 5000, None),  # Crosses EOF
                (103000, 5000, None), # Past EOF
    )
    out = test(requests, max_hole = 1000000, max_size = None, force_align = None)
    assert out == ((90000, 18000, None),)

    # Ditto but no consolidation, due to a small max_size.
    # No clipping needed in this case, as the lowest level
    # read will handle that.
    out = test(requests, max_hole = 300, max_size = 10, force_align = None)
    assert out == requests

if __name__ == "__main__":
    np.seterr(all='raise')
    with LocalFileAutoDelete("testfile.dat") as fn:
        test_FileADT(fn.name)
    if HasSeismicStore():
        with CloudFileAutoDelete("openzgy-1.dat", SDCredentials()) as fn:
            test_SDFile(fn.name, False)
        with CloudFileAutoDelete("openzgy-2.dat", SDCredentials()) as fn:
            test_SDFile(fn.name, True)
        test_SDFileReadPastEOF(SDTestData("Synt2.zgy"))
        test_Consolidate()
    sys.exit(0)

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
