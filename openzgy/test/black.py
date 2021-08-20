#!/usr/bin/env python3

"""
Black-box end to end tests for OpenZGY.

If the old ZGY-Public Python module is available, several of the
tests will be run both on the old and the new implementation to
verify that they mach. This is a bit messy. Because there are some
known bugs in the old code and some deliberate changes in the new.

If the Seismic Store plug-in and/or the ZFP compression plug-in
are available then this functionality is tested as well.

 * Tested in checkReadingDeadArea(), checkContents, etc.
   On read, bricks written as explicit all zero and bricks that were
   never written should be treated identically. This is not quite the
   case with the legacy reader: A read request covering both existing
   and non-existing bricks works as described. A read request where
   all the corresponding bricks are missing will return an error. This
   is unfortunate as it makes the caller more aware of the file's
   layout. The test should try reading a small rectangle fully inside
   a partially written brick but outside the written area, and one
   fully inside a never-written brick, and one that overlaps both
   types of brick.

 * Tested in checkContents() and checkRawContents().
   On read of integral data, requesting data as float should give the
   same result as requesting the data as storage values and doing the
   conversion afterwards. Make sure this holds both for regular data,
   data from all-constant bricks, and values in missing bricks. In
   particular, make sure that "absent" data doesn't get returned as
   storage-zero when reading as integer and converted-zero when
   reading as float. To test for this, make sure the coding range is
   not symmetrical. I.e. storage zero must not map to converted zero.

 * Tested in checkContents() and checkRawContents().
   When doing a partial write of a brick that did not exist, the
   missing values should be "zero after conversion to float", or as
   close to zero as possible. Make sure they are not garbage and not
   "zero before conversion to float" instead. See
   Accessor::WriteDataExT

 * Tested in checkStatistics() and checkHistogram().
   Statistics and histogram information stored in the ZGY file should
   have the same values as if the entire survey was read and statistics
   and histogram was computed from those values. In other words,
   statistics should count all samples inside the survey boundary
   regardless of whether they come from regular, all-constant, or
   never written bricks. Samples from the padding area outside the
   survey boundary must not be counted.
   This is trivial if statistics and histogram is computed in a
   separate pass. Less so if the information is collected during write.
   NOTE: The old accessor will not count never-written bricks.

 * Tested in checkStatistics() and checkHistogram().
   The above rule holds true even when the coding range is not zero centric
   (cannot precisely represent zero after conversion) or does not contain
   zero so zero cannot be represented at all. In these cases, even
   never-written bricks will affect "sum of all samples" et cetera.
   To test this, two additional copies of the test data is needed
   with "non zero centric" coding and "only positive" coding.
   NOTE: The statistics will be pretty useless in this case, so we might
   not really care either way.
   NOTE: The old accessor will not count never-written bricks.

 * Tested in checkStatistics() and checkHistogram().
   Statistics and histogram should handle overwritten data correctly.
   This is trivial if statistics and histogram is computed in a
   separate pass. Less so if the information is collected during write.
   In the test data, the inersection of "A" and "B" is overwritten.

 * The histogram range should be wide enough for all samples to fit.
   It is allowed to be wider. Specifically, for an 8-bit file the only
   thing that makes sense is to have a 1:1 correspondence between
   storage values and histogram bins. So, histogram range equals
   coding range. For a 16-bit file which makes use of most of the
   available storage values (which is a reasonable assumption) one
   could also set histogram range equals coding range, assigning 256
   storage values to each histogram bin. Not explicitly tested yet.

 * If alpha information is written, and this is done before writing
   the bricks, then histogram and statistics should only include actually
   live traces. This test is N/A if we completely deprecate alpha support.

 * If alpha information is written to change a trace to dead after
   bulk has been written for that trace, the effect on the statistics
   is unspecified. Technically it would make sense to immediately
   correct the statistics once the alpha changes. This is not
   implemented even in the old accessor. And probaby never will be.
   This is N/A for testing in any case since the result is unspecified.

 * Just in case we are forced to keep the old behavior that treats all-zero
   bricks slightly different from never-written bricks, it is recommended
   that applications that don't need this odd behavior explicitly fills
   all newly created file with zeros before writing real data.
   This is N/A for testing.

 * Tested in testFancyReadConstant().
   Applications that do want to distinguish between never-written,
   all-constant, and regular bricks should only do so for performance
   reasons. A separate api function will be provided to query the
   brick status. This new api function obviously needs to be tested.

 * Not tested. Mostly of historical interest.
   The curorig and cursize members are deprecated but we might add a
   unit test just to document the current behavior. They were supposed
   to give the bounding box of data actually written to the file. Or
   possibly the bounding box including padding to the nearest brick
   boundary. Or maybe somebody just gave up and set them equal to the
   survey. I suspect the origin will always be included, see
   OnBrickWritten. I also suspect that DataInfo::SetExtent is always
   called with setcur=true which means the range will always match the
   full size.

"""

#print('Running' if __name__ == '__main__' else 'Importing', __file__)

import numpy as np
import os
import sys
import io
import math
import json
import base64
import time
from contextlib import suppress, ExitStack, contextmanager
from enum import Enum
from collections import namedtuple

try:
    from .. import zgypublic as oldzgy
    print("Also testing the old ZGY-Public API.")
except Exception as ex:
    print("Old ZGY-Public is not available:", ex)
    class FakeAPI:
        zgy = None
        ZgyReader = object()
        ZgyWriter = object()
    oldzgy = FakeAPI()

from .. import api as newzgy
from ..api import SampleDataType, UnitDimension, ProgressWithDots, ZgyCompressFactory, ZgyKnownCompressors, ZgyKnownDecompressors
from ..impl.lodalgo import DecimationType # TODO-Low encapsulation?
from ..test.utils import SDCredentials, TempFileAutoDelete, LocalFileAutoDelete, CloudFileAutoDelete, HasSeismicStore, HasZFPCompression, SDTestData, SDTestSink
from ..impl.enum import UpdateMode
from ..exception import *

def HasOldZgy():
    return oldzgy.zgy is not None

def showZgy(*args):
    msg = ""
    for a in args:
        if a is None: pass
        elif a is newzgy.ZgyReader: msg += " and new reader"
        elif a is newzgy.ZgyWriter: msg += " and new writer"
        elif a is oldzgy.ZgyReader: msg += " and old reader"
        elif a is oldzgy.ZgyWriter: msg += " and old writer"
        else: msg += " and " + a.__module__ + "." + a.__name__
    return msg[5:] if msg else ""

# ----- Called by test code; not runnable by themselves. ----- #

@contextmanager
def TimeMe(name):
    #start = time.perf_counter()
    yield None
    #elapsed = time.perf_counter() - start
    #print("TIMED: %-20.20s %7.3f" % (name+":", elapsed), flush=True)

class TraceCallsToSD:
    """
    Suitable for use as a _debug_trace callback.
    """
    _entry = namedtuple("io", "what nbytes padded parts")
    def __init__(self, *, verbose = False):
        self.calls = []
        self._verbose = verbose
    def __call__(self, what, nbytes, padded, parts):
        self.calls.append(self._entry(what, nbytes, padded, parts))
        if self._verbose:
            print("  {0:9s} size {1:10s} padded {2:10s} parts {3:1d}".format(
                what, self._pretty(nbytes), self._pretty(padded), parts))
    @staticmethod
    def _pretty(n):
        if (n < 1024) or (n % (1024) != 0):
            return "{0:4d} bytes".format(n)
        elif (n < 1024*1024) or (n % (1024*1024) != 0):
            return "{0:7d} KB".format(n//1024)
        else:
            return "{0:7d} MB".format(n//(1024*1024))
    def reset(self):
        self.calls = []

class MustThrow:
    """
    Check that we get the expected exception.
    """
    def __init__(self, message = None, extypes = None):
        self._extypes = extypes
        self._message = message
        if isinstance(extypes, type) and issubclass(extypes, Exception):
            self._extypes = (extypes,)
        self._exnames = tuple([e.__name__ for e in self._extypes]) if self._extypes else "Exception"

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is None:
            problem = 'Expected {0}, got no exception'.format(self._exnames)
        elif self._extypes and type not in self._extypes:
            problem = 'Expected {0} got {1} "{2}"'.format(self._exnames, type.__name__, str(value))
        elif self._message and str(value).find(self._message) < 0:
            problem = 'Expected "{0}" got "{1}"'.format(self._message, str(value))
        else:
            problem = None
            #print('Ok: Expected {0} "{1}" got {2} "{3}"'.format(self._exnames, self._message or "", type.__name__, str(value)))
        if problem:
            raise AssertionError(problem) from None
        return True # suppress the exception.

def pretty(n):
    """
    Format a number, assumed to be a size in bytes, as a human readable string.
    """
    if type(n) != type(42):
        return str(n)
    if n >= (1024*1024*1024) and (n % (1024*1024*1024)) == 0:
        return str(n//(1024*1024*1024)) + " GB"
    if n >= (1024*1024) and (n % (1024*1024)) == 0:
        return str(n//(1024*1024)) + " MB"
    if n >= (512*1024) and (n % (256*1024)) == 0:
        return str(n//(256*1024)) + "*256 KB"
    if n >= (1024) and (n % (1024)) == 0:
        return str(n//(1024)) + " KB"
    return str(n) + " bytes"

def savePNG(data, outfile):
    from PIL import Image
    def normalize(a):
        a = a.astype(np.float32)
        dead = np.isnan(a)
        amin, amax = (np.nanmin(a), np.nanmax(a))
        a[dead] = amin
        if amin == amax:
            a *= 0
        else:
            a = (a - amin) / (amax - amin)
        a = (a * 255).astype(np.uint8)
        return a, dead

    data = np.squeeze(data)
    data = np.transpose(data)
    data = np.flip(data, 1)
    data, dead = normalize(data)
    tmp = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    r = tmp[...,0]
    g = tmp[...,1]
    b = tmp[...,2]
    r += data
    g += data
    b += data
    r[dead] = 255
    g[dead] = 255
    b[dead] = 0
    im = Image.fromarray(tmp, mode="RGB")
    im.save(outfile, format="PNG")

def isMutable(obj, *, verbose = False, seen = set()):
    """
    Recursive check for whether an object is mutable.
    The idea was to check that all members of e.g. ZgyReader are
    immutable so the user cannot (a) shoot himself in the foot by
    directly modifying a data members, or (b) even worse, change
    some cached value by modifying a mutable member of a container.
    Unfortunately this was a lot harder then I thought.
      - A callable might or might not be const. Need to check the code.
      - A property and a data member look rather similar.
      - A readonly property may have a syub __set__ that will throw.
      - A __setattr__, if present,  can make any attribute mutable.
      - Python has no frozendict (yet) unless I want to add a
        rather pointless dependency, So I copy dicts before
        returning them. This is safe, but the code here cannot know.
        I might make my own dict-like wrapper but this is getting
        way too complicated.
    Looks like I just jave to rely on dump() followed by eyeballing
    the source code.
    """
    # Known types
    if isinstance(obj, (type(None), type, str, int, bool, float, tuple, bytes, Enum, np.dtype)):
        if verbose: print("Immutable type", type(obj).__name__)
        return False
    elif isinstance(obj, (list, set, dict, bytearray, np.ndarray)):
        if verbose: print("MUTABLE   type", type(obj).__name__)
        return True
    elif callable(obj):
        if verbose: print("CALLABLE  type", type(obj).__name__)
        return False

    # Recursive checks
    if id(obj) in seen:
        if verbose: print("skipping cycle of", type(obj).__name__)
        return False
    print("Adding", id(obj), "to seen")
    seen |= set((id(obj),))
    if isinstance(obj, dict):
        obj = obj.items()
    if isinstance(obj, tuple):
        if verbose: print("recursively checking", type(obj).__name__)
        return any([isMutable(e, verbose=verbose, seen=seen) for e in obj])
    if verbose: print("unknown type, assuming mutable", type(obj).__name__)
    return True

def hasMutableMembers(obj, *, safe = set(), verbose = False):
    """
    Try to detect whether obj (which is some kind of instance variable)
    has any plain data members or any properties that contain data that
    in turn looks like it is mutable. Note that this turned out to be
    a lot harder then I first thought. The tests are by no means complete.
    """
    if obj is not None:
        for x in sorted(dir(obj)):
            if x[0] != '_' and not x in safe:
                is_prop = isinstance(getattr(type(obj), x, None), property)
                is_call = callable(getattr(obj, x))
                if not is_prop and not is_call:
                    if verbose: print(type(obj).__name__ + "." + x,
                                      "looks like a DATA member")
                    return True
                if isMutable(getattr(obj, x), verbose=False, seen=set()):
                    if verbose: print(type(obj).__name__ + "." + x,
                                      "is of a MUTABLE type")
                    return True
    return False

def dump(message, obj, verbose = False):
    if message: print(message)
    class Dummy:
        """(no doc)"""
    for x in sorted(dir(obj)):
        if x[0] != '_':
            value = getattr(obj, x)
            if isinstance(getattr(type(obj), x, None), property):
                vt = "prop "
            elif callable(value):
                vt = "call "
            else:
                vt = "DATA "
            if isMutable(value, seen=set()):
                vt = "MUTABLE " + vt
            if verbose:
                doc = '\n' + str(getattr(obj.__class__, x, Dummy).__doc__)
                doc = doc.replace('\n', '\n\t\t')
                print('\t' + vt + x, "=", value, doc)
            else:
                if not callable(value):
                    print('\t' + vt + x, "=", value)
                else:
                    print('\t' + vt + x + "()")

def createFancyBuffer(defaultvalue, unwrittenvalue):
    """
    Create test data as described elsewhere. This version saves the
    data in an in-memory numpy array making the code quite trivial.
    There is no point in writing the data in multiple operations
    because we aren't testing numpy.

    The caller needs to specify the default value that will be
    assigned to samples that were never written. Separate defaults
    may be given for unwritten samples inside a brick vs. bricks
    never written to at all. If these two differ this is arguably
    a bug in the implementation.
    """
    data = np.full((112, 64, 176), defaultvalue, dtype=np.float32)
    data[16:16+40, 16:16+41, 16:16+42] = 31
    data[48:48+72, 20:20+10, 24:24+16] = 97
    data[:,:,128:176] = unwrittenvalue
    return data

def createFancyFile(filename, datatype, datarange, zgyWriterFactory, *, single_write = False, kwargs = dict()):
    """
    The layout of this test data is described in detail in doc/testdata.png
    The figure also explains how to compute the expected statistics by hand.
    As for computing the expected sample values, this is done by
    createFancyBuffer().

     * Create a ZGY file with size (112, 64, 176) which gives it a bricksize
       of 2x1x3. Other parameters vary.

     * Write an oddly sized rectangle "A" inside the first brick.

     * Write an oddly sized rectangle "B" covering two cubes and partly
       intersecting the first write, and also runs slightly into the
       padding area.

     * Write an all-zero region "C" that completely covers one brick and
       also covers a second brick completely apart from padding area
       outside the survey.

    Additional arguments such as "snr" can be passed as kwargs={"snr": 99},
    note that I have not declared the parameter as **kwargs so the dict
    must be created by hand. To make it more explicit what the extras are.

    Accounting for existing bugs:

    Several of the tests have arguments (defaultvalue,unwrittenvalue,countdead).
     - defaultvalue should be the value closest to 0 that can be represented.
     - unwrittenvalue ought to have been the same as defaultvalue, but with the
       old reader it might be 0 for float access and 0 converted to float for
       raw.

     - countdead should be True meaning unwritten samples are included in the
       statistics and the histogram, but if the file was created by the old
       writer then it needs to be set False.

    Future: If implementing aplha support (currently not the case) we will
    also need a file with alpha tiles set to the horizontal extent of the
    actual stored data. In this data set there will still be unwritten
    data at the tail end of each trace. Production code rarely does
    this though; the assumption is that all traces have the same length
    and that traces are written fully or not at all.
    Note that currently, neither the old ZGY-Public nor the new OpenZGY
    API can write alpha tiles. Only ZGY-Internal can do that. That API
    does not have any Python wrapper.
    """
    with zgyWriterFactory(filename,
                       iocontext = SDCredentials(),
                       size = (112, 64, 176),
                       datatype = datatype,
                       datarange = datarange,
                       zunitdim = UnitDimension.time,
                       zunitname = "ms",
                       zunitfactor = 0.001,
                       hunitdim = UnitDimension.length,
                       hunitname = "ft",
                       hunitfactor = 0.3048,
                       zstart = 2500,
                       zinc = 4.125,
                       annotstart = (1234, 5678),
                       annotinc = (5, 2),
                       corners = ((1000, 1000),
                                  (3775, 1000),
                                  (1000, 2890),
                                  (3775, 2890)),
                       **kwargs
    ) as writer:
        expect_datarange_1 = datarange
        if datatype == SampleDataType.float and zgyWriterFactory != oldzgy.ZgyWriter:
            # The value is unspecified. It could be NaN if the file was never
            # flushed, or (0,0) if it was flushed before writing anything.
            # Or it could be the (likely not calculated yet) statistical
            # range if the code in api.ZgyMeta.datarange chooses to return
            # the statistical range instead.
            expect_datarange_1 = (0, 0)
        #dump(filename, writer)
        checkmeta(writer, datatype, expect_datarange_1)
        if single_write:
            # Read/modify/write is not allowed whan writing compressed data,
            # or at least not recommended since noise will accumulate.
            writer.write((0, 0, 0), createFancyBuffer(0, 0))
        else:
            writer.write((16,16,16), np.full((40,41,42), 31, dtype=np.float32))
            writer.write((48,20,24), np.full((72,10,16), 97, dtype=np.float32))
            writer.write((0,0,64),   np.full((112,64,64), 0, dtype=np.float32))
        # Statistics haven't been computed yet, so datarange for float cubes
        # should still be returned as empty.
        checkmeta(writer, datatype, expect_datarange_1)
    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        expect_datarange_2 = datarange
        if datatype == SampleDataType.float:
            if True or zgyWriterFactory != oldzgy.ZgyWriter:
                # The value has been explicitly set to the statistical range
                # if written by the new writer. If api.ZgyMeta.datarange chooses
                # to return the statistical range instead this this happens
                # also for files written by the old accessor. The second
                # conditinal should be disabled in that case.
                expect_datarange_2 = (reader.statistics.min, reader.statistics.max)
        checkmeta(reader, datatype, expect_datarange_2)

def checkmeta(meta, datatype = None, datarange = None):
    """
    Verify round trip of metadata. This can be used both by a writer
    (ensure the data we set is still available as properties) and a
    reader (ensure the roundtrip to a stored file and back worked).
    """
    assert(meta.size == (112, 64, 176))
    assert(datatype is None or meta.datatype == datatype)
    assert(datarange is None or meta.datarange == datarange)
    assert(meta.raw_datarange == meta.datarange)
    assert(meta.zunitdim == UnitDimension.time)
    assert(meta.zunitname == "ms")
    assert(abs(meta.zunitfactor - 0.001) < 1.0e-5)
    assert(meta.hunitdim == UnitDimension.length)
    assert(meta.hunitname == "ft")
    assert(abs(meta.hunitfactor - 0.3048) < 0.0001)
    assert(meta.zstart == 2500)
    assert(abs(meta.zinc - 4.125) < 0.0001)
    assert(meta.annotstart == (1234, 5678))
    assert(meta.annotinc == (5, 2))
    assert np.sum(np.abs(np.array(meta.corners) -
                         np.array(((1000, 1000),
                                   (3775, 1000),
                                   (1000, 2890),
                                   (3775, 2890))))) < 0.0001

def explaincontents(expect, actual, delta):
    """
    Detailed checking of a small part of the standard test cube.
    A single trace that covers many special cases. Show an explanation
    of what is being tested as well as expected vs. actual results.
    See doc/testdata.png. This method is meant to be used to understand
    why a particular test has failed.
    """
    table = [(  0,  16, "default(r/m/w)"),
             ( 16,  24, "written once  "),
             ( 24,  40, "written twice "),
             ( 40,  58, "written once  "),
             ( 58,  63, "default(r/m/w)"),
             ( 64, 128, "constant-zero "),
             (128, 176, "default(empty)")]
    print("Displaying the trace at [50,22,:]")
    for beg, end, text in table:
        ex = expect[50,22,beg:end]
        ac = actual[50,22,beg:end]
        if np.amin(ex) == np.amax(ex) and np.amin(ac) == np.amax(ac):
            print("  ", text, "expect", ex[0], "actual", ac[1])
        else:
            print("  ", text, "expect", ex, "actual", ac)
    print("   largest error in entire cube:", delta)

def checkContents(filename, zgyReaderFactory, defaultvalue, unwrittenvalue, *, maxdelta = 0.001):
    """
    Read back the entire survey from one of the files created by
    createFancyFile() and compare with the expected results.
    Also check the metadata.
    """
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    expect = createFancyBuffer(defaultvalue, unwrittenvalue)
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader, io.StringIO() as bitbucket:
        # Improve coverage by exercising the debug log statements
        verbose = lambda *args, **kwargs: print(*args, file=bitbucket, **kwargs)
        checkmeta(reader)
        actual = np.zeros((112, 64, 176), dtype=np.float32)
        reader.read((0,0,0), actual, verbose = verbose)
    delta = np.amax(np.abs(expect - actual))
    if not delta <= maxdelta:
        explaincontents(expect, actual, delta)
    assert delta <= maxdelta

def compareArrays(expect, actual, value_epsilon = 0.02, count_epsilon = 0.01, *, verbose = False):
    value_range = np.amax(expect) - np.amin(expect)
    count_total = len(expect.flat)
    # Error in each sample, relative to the total expected value range.
    # Can technically be greater than 1 if "actual" has wild values.
    # A value of e.g. <= 0.01 might be considered close enough.
    value_delta = np.abs(expect - actual) / (value_range if value_range else 1)
    count_bad = np.count_nonzero(value_delta > value_epsilon)
    # In addition to the test for not exactly equal, allow a certain
    # fraction of samples to differ by any amount. Typically this
    # might be needed due to edge effects in lowres data.
    relative_bad = count_bad / count_total
    ok = relative_bad <= count_epsilon
    if verbose:
        print("{5}: {0:6d} of {1:7d} samples ({2:.2f}%) differ > {3:.2f}%. Allowed {4:.2f}%.".format(
            count_bad, count_total, 100.0 * count_bad / count_total,
            100.0 * value_epsilon, 100.0 * count_epsilon,
            "pass" if ok else "FAIL"))
    return ok

def showdecimation(lod0, lod1):
    """
    Input 4 hires traces (2,2,n) and a corresponding decimated
    trace (n//2) and display those to manually inspect the result.
    """
    print(" decimated   from these input samples")
    for ii in range(0, lod0.shape[2], 2):
        print("{0:10.5g}   {1}".format(lod1[ii//2], list(lod0[:,:,ii:ii+2].flat)))

def checkLodContents(filename, zgyReaderFactory, defaultvalue, unwrittenvalue):
    """
    As checkContents, but caller specifies which LOD to read and we
    allow some slop in the result since the "expect" array uses trivial
    decimation while the zgy writer uses something fancier.
    NOTE: Due to bugs in the old writer, no checks are done for samples
    where the fullres data has never been written. I have given up on
    figuring out the current behavior; I just know that it is wrong.
    """
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        nlods = 1
        size = np.array(reader.size, dtype=np.int64)
        while np.any(size > reader.bricksize):
            nlods += 1
            size = (size + 1) // 2
        assert nlods == reader.nlods
        for lod in range(0, nlods):
            step = 1<<lod
            expect = createFancyBuffer(defaultvalue, unwrittenvalue)
            expect = expect[:,:,:128] # Hard coded edge of written data.
            expect = expect[::step,::step,::step]
            size = (np.array(reader.size, dtype=np.int64) + (step-1)) // step
            size[2] = 128//step
            actual = np.zeros(size, dtype=np.float32)
            reader.read((0,0,0), actual, lod = lod)
            ok = compareArrays(expect, actual,
                                value_epsilon = 0.02 if lod < 2 else 0.04,
                                count_epsilon = 0.01 if lod < 2 else 0.03)
            if not ok:
                deltas = np.abs(expect - actual).astype(np.float64)
                # A single 2d section in the "interesting" part of the survey.
                actual_2d = actual[:,22//step,:]
                expect_2d = expect[:,22//step,:]
                deltas_2d = deltas[:,22//step,:]
                # A single trace in the "interesting" part of the survey.
                expect_1d = expect_2d[50//step,:]
                actual_1d = actual_2d[50//step,:]
                deltas_1d = deltas_2d[50//step,:]
                # Now visualize these for debugging
                savePNG(actual[:,22//step,:], "actual-" + str(lod) + ".png")
                savePNG(expect[:,22//step,:], "expect-" + str(lod) + ".png")
                savePNG(deltas[:,22//step,:], "deltas-" + str(lod) + ".png")
                print("\n{0} LOD {1} check: {2}".format(
                    filename, lod, ("pass" if ok else "FAIL")))
                print("Default", defaultvalue, "unwritten", unwrittenvalue)
                print("first sample expect {0} actual {1}".format(
                    expect[0,0,0], actual[0,0,0]))
                print("last sample expect {0} actual {1}".format(
                    expect[-1,-1,-1], actual[-1,-1,-1]))
                print("interesting trace expect", expect_1d,
                      "interesting trace actual", actual_1d,
                      "delta", deltas_1d,
                      sep="\n")
            assert ok

def checkRawContents(filename, zgyReaderFactory, defaultvalue, unwrittenvalue, *, maxdelta = 0.001):
    """
    As checkContents, but do the value conversion ourselves.
    There may be issues with never written bricks.
    """
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    expect = createFancyBuffer(defaultvalue, unwrittenvalue)
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        dtype = {SampleDataType.int8: np.int8,
                 SampleDataType.int16: np.int16,
                 SampleDataType.float: np.float32 }[reader.datatype]
        checkmeta(reader)
        actual = np.zeros((112, 64, 176), dtype=dtype)
        reader.read((0,0,0), actual)
        #print("raw...", actual[50,22,:])
        if np.issubdtype(dtype, np.integer):
            iinfo = np.iinfo(dtype)
            actual = actual.astype(np.float32)
            a = (reader.datarange[1]-reader.datarange[0])/(iinfo.max-iinfo.min)
            b = reader.datarange[0] - a * iinfo.min
            actual *= a
            actual += b
    delta = np.amax(np.abs(expect - actual))
    if not delta <= maxdelta:
        # A single trace in the "interesting" part of the survey.
        print("expect", expect[50,22,:])
        print("actual", actual[50,22,:])
        print("delta", delta)
    assert delta <= maxdelta

def computeStatisticsByRead(filename, zgyReaderFactory):
    """
    Read back the entire survey from one of the files created by
    createFancyFile() and compute statistics from the bulk data.
    Concentrate on sum of samples and count of samples.
    Also check the metadata.
    """
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        checkmeta(reader)
        data = np.zeros((112, 64, 176), dtype=np.float32)
        reader.read((0,0,0), data)
        theSum = np.sum(data.flat, dtype=np.float64)
        theCount = len(data.flat)
    #print("Read sum {0}, sample count {1}".format(theSum, theCount))
    #cnt = 0
    #for x in (0, 1, 31, 97):
    #    c = np.count_nonzero(data == x)
    #    print(x, c)
    #    cnt += c
    #print("?", theCount - cnt) # unaccounted for
    return theSum, theCount

def readStatisticsStoredInFile(filename, zgyReaderFactory):
    """
    Open the ZGY file and retrieve only the stored statistics information.
    This is only supported in the new API.
    """
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        stats = reader.statistics
        #print(stats)
        return (stats.sum, stats.cnt)

def computeStatisticsByHand(defaultvalue, unwrittenvalue):
    S = 112 * 64 * 176     # Total samples in survey, excluding padding.
    P = 128 * 64 * 192 - S # Padding samples to align with 64^3 bricks.
    A = 40 * 41 * 42       # Rect A beg (16,16,16) end (56,57,58) value 31.
    B = 72 * 10 * 16       # rect B beg (48,20,24) end (120,30,40) value 97.
    C = 112 * 64 * 64      # rect C beg (0,0,64) end (112,64,128) value 0.
    D = 8 * 10 * 16        # overlap A/B, begin at (48,20,24).
    E = 8 * 10 * 16        # B outside survey: begin at(128,30,40).
    Z = 112 * 64 * 48      # Samples inside survey in never-written bricks.
    nSample_31 = A - D
    nSample_97 = B - E
    nSample_unwritten = Z
    nSample_default = S - nSample_31 - nSample_97 - nSample_unwritten
    theSum = (31 * nSample_31 +
              97 * nSample_97 +
              defaultvalue * nSample_default +
              (unwrittenvalue or 0) * nSample_unwritten)
    theCount = S if unwrittenvalue is not None else S - Z
    #print("Expected sum {0} * {1} + {2} * {3} + {4} * {5} + {6} * {7} = {8}, sample count {9}".format(31, nSample_31, 97, nSample_97, defaultvalue, nSample_default, unwrittenvalue, nSample_unwritten, theSum, theCount))
    if unwrittenvalue is None:
        theHist = { 31: nSample_31, 97: nSample_97,
                    defaultvalue: nSample_default }
    elif defaultvalue == unwrittenvalue:
        theHist = { 31: nSample_31, 97: nSample_97,
                    defaultvalue: nSample_default + nSample_unwritten }
    else:
        theHist = { 31: nSample_31, 97: nSample_97,
                    defaultvalue: nSample_default,
                    unwrittenvalue: nSample_unwritten }
    return theSum, theCount, theHist

def checkStatistics(filename, zgyReaderFactory, defaultvalue, unwrittenvalue, countdead, *, maxdelta = 0.001):
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    byhand = computeStatisticsByHand(defaultvalue, unwrittenvalue)
    byread = computeStatisticsByRead(filename, zgyReaderFactory)
    if not (abs(byhand[0]-byread[0]) < maxdelta and byhand[1] == byread[1]):
        print("stat sum: byhand: {0}, byread {1}, maxdelta {2}, count byhand: {3} byread {4}".format(byhand[0], byread[0], maxdelta, byhand[1], byread[1]))
    assert(abs(byhand[0]-byread[0]) < maxdelta and byhand[1] == byread[1])
    if zgyReaderFactory is not oldzgy.ZgyReader:
        byhand = computeStatisticsByHand(defaultvalue, unwrittenvalue if countdead else None)
        byload = readStatisticsStoredInFile(filename, zgyReaderFactory)
        assert(abs(byhand[0]-byload[0]) < maxdelta and byhand[1] == byload[1])

def findHistogramSlot(value, histrange):
    """
    Which slot this value belongs to in a 256-bin histogram.
    The result is guaranteed to be in the range [0..255].
    Values outside range are clipped to 0 or 255. This is not
    how the actual histogram computation is done, but for the
    tests it should not make any difference.
    """
    value = 255 * (value - histrange[0]) / (histrange[1] - histrange[0])
    return int(np.rint(np.clip(value, 0, 255)))

def checkHistogram(filename, zgyReaderFactory, defaultvalue, unwrittenvalue, countdead):
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    if zgyReaderFactory is not oldzgy.ZgyReader:
        with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
            stat = (reader.statistics.min, reader.statistics.max)
            hist = (reader.histogram.min, reader.histogram.max)
            data = (reader.datarange[0], reader.datarange[1])
            if False:
                print("checkHistogram:",
                      "stat", stat, "hist", hist, "data", data,
                      "type", reader.datatype.name)
            if reader.datatype == SampleDataType.float:
                # Float data written by the old writer currently writes
                # the histogram on the fly and may end up with a too wide
                # range. The new reader doesn't do this now but it might do
                # so in the future. Note that data == stat for float zgy.
                assert hist[0] <= data[0] and hist[1] >= data[1]
            else:
                assert math.isclose(hist[0],data[0]) and math.isclose(hist[1],data[1])

            assert reader.histogram.cnt == reader.statistics.cnt
            hist = reader.histogram
            #print(hist)
            _, _, byhand = computeStatisticsByHand(defaultvalue, unwrittenvalue if countdead else None)
            #print(byhand)

            expect_hist = np.zeros(256, dtype=np.int64)
            for value, expect in byhand.items():
                slot = findHistogramSlot(value, (hist.min, hist.max))
                expect_hist[slot] += expect
            for slot in range(256):
                actual = hist.bin[slot]
                expect = expect_hist[slot]
                if actual != expect:
                    print("histogram value", value, "slot", slot,
                          "expect", expect, "actual", actual)
                    #print("actual", hist)
                    #print("expect", expect_hist)
                assert actual == expect

def isReaderOpen(reader):
    """
    Return True if the zgy file is open for read.
    There isn't a property for that in the API because
    typically this is only needed when testing.
    """
    tmp = np.zeros((1, 1, 1), dtype=np.float32)
    try:
        reader.read((0,0,0), tmp)
    except (RuntimeError, newzgy.ZgyUserError) as ex:
        assert "ot open for" in str(ex)
        return False
    return True

def checkReadingDeadArea(filename, pos, zgyReaderFactory, expected):
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        tmp = np.full((2, 2, 2), 42, dtype=np.float32)
        reader.read(pos, tmp)
        #print(list(tmp.flat), "expected", expected)
        assert np.all(np.abs(tmp - expected) < 0.001)

def checkReadingOutsideRange(filename, zgyReaderFactory):
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        tmp = np.full((2, 2, 2), 42, dtype=np.float32)
        with MustThrow("outside the valid range"):
            reader.read((0, 0, 10000), tmp)
        with MustThrow("outside the valid range"):
            reader.read((0, 0, -9999), tmp)
        with MustThrow("outside the valid range"):
            reader.readconst((0, 0, 10000), (2, 2, 2))
        with MustThrow("outside the valid range"):
            reader.readconst((0, 0, -9999), (2, 2, 2))
        #with MustThrow("outside the valid range"):
        #    reader.readconst((0, 0, 0), (1000000, 1000000, 1000000))

def checkReadingOutsideLod(filename, zgyReaderFactory):
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        tmp = np.full((2, 2, 2), 42, dtype=np.float32)
        with MustThrow("outside the valid range"):
            reader.read((0, 0, 0), tmp, lod=-1)
        with MustThrow("outside the valid range"):
            reader.read((0, 0, 0), tmp, lod=9)
        with MustThrow("outside the valid range"):
            reader.readconst((0, 0, 0), (2, 2, 2), lod=-1)
        with MustThrow("outside the valid range"):
            reader.readconst((0, 0, 0), (2, 2, 2), lod=9)

def checkReadingToWrongValueType(filename, zgyReaderFactory):
    """
    This was supposed to cover a test in readToExistingBuffer()
    but now the error is caught already in the API layer.
    Which is already tested in testBadArgumentsOnReadWrite.
    Keeping the test here in case this changes back later.
    """
    if zgyReaderFactory == oldzgy.ZgyReader and not HasOldZgy(): return
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        tmp = np.full((2, 2, 2), 42, dtype=np.int16)
        #with MustThrow("conversion only supported"):
        with MustThrow("array of np.float32 or np.int8"):
            reader.read((0, 0, 0), tmp)

def hasSAuthToken():
    try:
        jwt = json.loads(base64.urlsafe_b64decode(SDCredentials().sdtoken.split(".")[1] + "====").decode("ascii"))
        print(json.dumps(jwt, indent=2, sort_keys=True))
        timeleft = jwt["exp"] - int(time.time())
        print("SAuth token has", timeleft // 60, "minutes to expiry")
        return timeleft > 0
    except IOError:
        # Missing or malformed token, including "FILE:" tokens.
        # Unfortunately, impersonation tokens that are still
        # good to refresh will also fail here.
        return True # optimist.

# ----- Separate tests, but needs testFancy() to create the test files. ----- #

def runCloseOnException(filename, zgyReaderFactory):
    """
    Test that the "with" guard is working properly.
    On leaving the scope the reader should be closed.
    Even if we left via an exception.
    """
    class DummyException(Exception):
        pass
    try:
        # If the reader raises an exception in __init__ then "reader"
        # remains unassigned. While if we raise an exception ourselves
        # it gets caught at the same level but now with "reader" known.
        # No big deal as long as we *only* catch the dummy exception,
        with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
            assert isReaderOpen(reader)
            raise DummyException("testing...")
    except DummyException:
        pass
    assert not isReaderOpen(reader)

def runErrorOnClose(filename, ZgyReaderFactory):
    """
    Only relevant for openzgy. Verify correct behavior when we exit
    the context manager due to an exception. For the old zgy wrapper
    there is no easy way of forcing an error to be thrown on close,
    so while I would like to have tested that one as well, I won't.
    """
    # Exception was thrown from inside the block only.
    # Make sure the reader was closed. This peeks at internal data.
    try:
        message = ""
        with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
            raise RuntimeError("xyzzy")
    except Exception as ex:
        message = str(ex)
    assert message == "xyzzy"
    assert reader._fd is None

    # Exception was thrown from the reader's close() method only.
    try:
        message = ""
        with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
            reader._fd.xx_close()
            reader._fd = "oops"
    except Exception as ex:
        message = str(ex)
    assert message.find("object has no attribute") >= 0

    # Exception was thrown from inside the block, then when handling
    # that exception another exception was thrown inside close().
    try:
        message1 = ""
        message2 = ""
        with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
            reader._fd.xx_close()
            reader._fd = "oops"
            raise RuntimeError("xyzzy")
    except Exception as ex:
        message1 = str(ex)
        message2 = str(ex.__cause__ or ex.__context__)
    assert message1.find("object has no attribute") >= 0
    assert message2 == "xyzzy"

def runConversions(filename, zgyReaderFactory):
    """
    Verify that coordinate conversion between index, annot, and world works.
    """
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as demo:
        #dump("", demo, True)
        a = demo.indexToAnnot((3, 7))
        i = demo.annotToIndex(a)
        #print(a, i)
        assert(a == (1249, 5692) and i == (3, 7))
        w = demo.indexToWorld((0, 0))
        i = demo.worldToIndex(w)
        #print(w, i)
        assert(w == (1000, 1000) and i == (0, 0))
        w = demo.indexToWorld((1, 0))
        i = demo.worldToIndex(w)
        #print(w, i)
        assert(w == (1025, 1000) and i == (1, 0))
        w = demo.indexToWorld((0, 1))
        i = demo.worldToIndex(w)
        #print(w, i)
        assert(w == (1000, 1030) and i == (0, 1))
        w = demo.indexToWorld((3, 7))
        i = demo.worldToIndex(w)
        #print(w, i)
        assert(w == (1000 + 3*25, 1000 + 7*30) and i == (3, 7))
        w = demo.annotToWorld(a)
        a = demo.worldToAnnot(w)
        #print(w, a)
        assert(w == (1000 + 3*25, 1000 + 7*30) and a == (1249, 5692))


def runErrorIfNotOpenForRead(filename, zgyReaderFactory):
    size = (1, 1, 1)
    tmp = np.zeros(size, dtype=np.float32)
    pos = (0, 0, 0)
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        reader.close()
        with MustThrow("ot open for read"):
            reader.read(pos, tmp)
        if zgyReaderFactory is not oldzgy.ZgyReader:
            with MustThrow("ot open for read"):
                reader.readconst(pos, size)

def runDumpToDevNull(filename, zgyReaderFactory):
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader, io.StringIO() as stream:
        reader._meta.dumpRaw(file=stream)
        # No test on the result, only see that it doesn't crash.
        assert len(stream.getvalue()) > 0

def runClone(filename, templatename):
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(), templatename=templatename) as writer:
        checkmeta(writer, SampleDataType.int8, (-28,+227))
    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        checkmeta(reader, SampleDataType.int8, (-28,+227))

def runUpdate(filename):
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(), templatename=filename) as writer:
        checkmeta(writer, SampleDataType.int8, (-28,+227))
    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        checkmeta(reader, SampleDataType.int8, (-28,+227))

def runDumpMembers(filename, templatename):
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(), templatename=templatename) as writer:
        #dump("\nZgyWriter contents:", writer, verbose=False)
        assert not hasMutableMembers(writer, safe=set(("meta",)), verbose=True)
    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        #dump("\nZgyReader contents:", reader, verbose=True)
        assert not hasMutableMembers(reader, safe=set(("meta",)), verbose=True)

# ----- Separately runnable tests, might need caller to clean up files. ----- #

def testRegisteredCompressors():
    #print("Known compressors", ",".join(ZgyKnownCompressors()),
    #      "decompressors", ",".join(ZgyKnownDecompressors()))
    assert "ZFP" in ZgyKnownCompressors()
    assert "ZFP" in ZgyKnownDecompressors()
    with MustThrow('"XYZZY" not recognized. Must be one of', ZgyMissingFeature):
        lossy = ZgyCompressFactory("XYZZY", snr=30)

def testProgressWithDots():
    with io.StringIO() as line:
        p = ProgressWithDots(length=51, outfile=line)
        assert line.getvalue() == ""
        p(0, 1000)
        assert line.getvalue() == "."
        p(1, 1000)
        assert line.getvalue() == "."
        p(500, 1000)
        assert line.getvalue() == "." * 26
        p(999, 1000)
        assert line.getvalue() == "." * 50
        p(1000, 1000)
        assert line.getvalue() == "." * 51 + "\n"

def testBadArgumentsOnCreate():
    fname = "should-not-exist.zgy"
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass
    with MustThrow("size must be specified", newzgy.ZgyUserError):
        with newzgy.ZgyWriter(fname):
            pass
    with MustThrow("size must be at least 1", newzgy.ZgyUserError):
        with newzgy.ZgyWriter(fname, size=(10,0,20)):
            pass
    with MustThrow("bricksize must be specified in 3 dimensions", newzgy.ZgyUserError):
        with newzgy.ZgyWriter(fname, size=(10,15,20), bricksize=(64,64)):
            pass
    with MustThrow("bricksize must be >= 4 and a power of 2", newzgy.ZgyUserError):
        with newzgy.ZgyWriter(fname, size=(10,15,20), bricksize=(64,64,48)):
            pass
    with MustThrow("datarange must be specified for integral types", newzgy.ZgyUserError):
        with newzgy.ZgyWriter(fname, size=(10,15,20), datatype=SampleDataType.int8):
            pass
    with MustThrow("datarange must have min < max", newzgy.ZgyUserError):
        with newzgy.ZgyWriter(fname, size=(10,15,20), datatype=SampleDataType.int8, datarange=(3,2)):
            pass
    with MustThrow("datarange must have min < max", newzgy.ZgyUserError):
        with newzgy.ZgyWriter(fname, size=(10,15,20), datatype=SampleDataType.int8, datarange=(3,3)):
            pass
    with MustThrow("datarange must be finite", newzgy.ZgyUserError):
        with newzgy.ZgyWriter(fname, size=(10,15,20), datatype=SampleDataType.int8, datarange=(np.nan,np.nan)):
            pass
    # The consistency checks should be done before actually creating the file.
    # Which means that the next call should fail.
    with MustThrow(None, FileNotFoundError):
        os.remove(fname)

def testBadArgumentsOnReadWrite(filename):
    origin = (0, 0, 0)
    expect = "Expected a 3d numpy array of np.float32 or np.float32"
    with newzgy.ZgyWriter(filename, size=(10,15,20)) as w:
        with MustThrow(expect): # no data
            w.write(origin, None)
        with MustThrow(expect): # not numpy data
            w.write(origin, [[[1,1,1]]])
        with MustThrow(expect): # wrong data type
            w.write(origin, np.array([[[1,1,1]]], dtype=np.int8))
        with MustThrow(expect): # wrong number of dimensions
            w.write(origin, np.array([1,1,1], dtype=np.float32))

    expect = "Expected a writeable 3d numpy array of np.float32 or np.float32"
    with newzgy.ZgyReader(filename) as r:
        with MustThrow(expect): # no data
            r.read(origin, None)
        with MustThrow(expect): # not numpy data
            r.read(origin, [[[1,1,1]]])
        with MustThrow(expect): # wrong data type
            r.read(origin, np.array([[[1,1,1]]], dtype=np.int8))
        with MustThrow(expect): # wrong number of dimensions
            r.read(origin, np.array([1,1,1], dtype=np.float32))
        with MustThrow(expect): # buffer not writeable
            a = np.array([[[1,1,1]]], dtype=np.float32)
            a.setflags(write=False)
            r.read(origin, a)

def testAutoDelete():
    # It is an error if the expected file is missing.
    with MustThrow("", FileNotFoundError):
        with LocalFileAutoDelete("xyzzy", silent=True) as fn:
            pass

    # As above, but if some other error occurred that will have precedence.
    with MustThrow("", IndexError):
        with LocalFileAutoDelete("xyzzy", silent=True) as fn:
            foo = [][1]

    # No attempt is made to remove, if we explicitly disarmed.
    with LocalFileAutoDelete("xyzzy") as fn:
        assert "/tmp-" in fn.name or "\\tmp-" in fn.name or fn.name[:4] == "tmp-"
        fn.disarm()

    # Actually try creating the file. Auto cleanup happens.
    with LocalFileAutoDelete("xyzzy") as fn:
        assert "/tmp-" in fn.name or "\\tmp-" in fn.name or fn.name[:4] == "tmp-"
        myname = fn.name
        with open(fn.name, "w"):
            pass
        assert os.path.exists(myname)
    assert not os.path.exists(myname)

    myname = [None, None]
    with ExitStack() as cleanup:
        fn1 = LocalFileAutoDelete("one")
        myname[0] = fn1.name
        cleanup.enter_context(fn1)
        with open(fn1.name, "w"):
            pass
        fn2 = LocalFileAutoDelete("two")
        myname[1] = fn2.name
        cleanup.enter_context(fn2)
        with open(fn2.name, "w"):
            pass
        assert os.path.exists(myname[0])
        assert os.path.exists(myname[1])
    assert not os.path.exists(myname[0])
    assert not os.path.exists(myname[1])

    myname = [None, None]
    with MustThrow("", FileNotFoundError):
        with ExitStack() as cleanup:
            fn1 = LocalFileAutoDelete("one")
            myname[0] = fn1.name
            cleanup.enter_context(fn1)
            with open(fn1.name, "w"):
                pass
            fn2 = LocalFileAutoDelete("two", silent=True)
            myname[1] = fn2.name
            cleanup.enter_context(fn2)
    # I did not get around to creating the second file.
    # This means the fn2 context will raise an exception.
    # fn1 should still have been deleted though.
    assert not os.path.exists(myname[0])

def testHistogramRangeIsCenterNotEdge(filename):
    """
    When the histogram gets generated by the ZGY writer, the range gives
    the center value of bin 0 and the center value of bin 255. NOT the
    lowest value that maps to bin 0 and the highest value that maps to
    bin 255. Which would arguably also make sense. Verify that behavior.
    """
    with oldzgy.ZgyWriter(filename, iocontext = SDCredentials(),
                       size = (64, 64, 64),
                       datatype = SampleDataType.float,
                       datarange =(0, 255),
                       zstart = 0, zinc = 4,
                       annotstart = (1, 1), annotinc = (1, 1),
                       corners = ((1000, 1000), (1630, 1000),
                                  (1000, 1630), (1630, 1630))
                   ) as writer:
        # With the 0..255 histogram range interpreted as the center of the
        # first and last bin, we have the following:
        # slot 0 is -0.5..+0.5, slot 2 is 1.5..2.5, slot 5 is 4.5..5.5
        # If we instead had a 0..256 histogram range interpreted as the
        # extreme eddes of the first and last bin, we have this:
        # slot 0 is 0..1, slot 2 is 2..3, slot 5 is 5..6, slot 255: 255..256
        # That would still be approximately correct at least for the first
        # few bins when setting the histogram range to 0..255 instead of
        # 0..256. So if the histogram algorithm choose to use the range
        # as the extreme limits (which it is NOT supposed to do),
        # 1.8 and 2.2 would end up in different slots. And 4.3 and 4.7
        # would end up in the same slot. It should be the other way around.
        #
        writer.write((0, 0, 0), np.full((1, 10, 10), 1.8, dtype=np.float32))
        writer.write((1, 0, 0), np.full((1,  1,  1), 2.2, dtype=np.float32))
        writer.write((2, 0, 0), np.full((1, 10, 5),  4.3, dtype=np.float32))
        writer.write((3, 0, 0), np.full((1,  1, 2),  4.7, dtype=np.float32))

    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        #print(reader.histogram)
        assert math.isclose(reader.histogram.min, 0.0)
        assert math.isclose(reader.histogram.max, 255.0)
        assert reader.histogram.bin[2] == 101
        assert reader.histogram.bin[4] == 50
        assert reader.histogram.bin[5] == 2

def testEmptyFile(filename, zgyWriterFactory = newzgy.ZgyWriter, zgyReaderFactory = newzgy.ZgyReader):
    """
    Create a file without writing bulk data to it; make sure it is
    well behaved both on write and on read back. Ideally test both
    on-prem and cloud, and test all 9 combinations of ZGY, OpenZGY/C++,
    and OpenZGY/Python readers and writers. With the current test
    framework it gets a bit tricky to test the two OpenZGY/C++ vs.
    OpenZGY/Python cases. Can I make a test that imports all three?
    """
    #print('testEmptyFile("{0}")'.format(filename))
    #print(' -> Using ' + showZgy(zgyWriterFactory, zgyReaderFactory))
    with zgyWriterFactory(filename,
                       iocontext = SDCredentials(),
                       size = (100, 200, 300),
                       datatype = SampleDataType.float,
                       datarange = (-1, 1),
                       zunitdim = UnitDimension.time,
                       zunitname = "ms",
                       zunitfactor = 0.001,
                       hunitdim = UnitDimension.length,
                       hunitname = "ft",
                       hunitfactor = 0.3048,
                       zstart = 2500,
                       zinc = 4.125,
                       annotstart = (1234, 5678),
                       annotinc = (5, 2),
                       corners = ((1000, 1000),
                                  (1005, 1000),
                                  (1000, 1002),
                                  (1005, 1002))
    ) as writer:
        pass

    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        slurp = np.ones(reader.size, dtype=np.float32)
        reader.read((0,0,0), slurp)
        assert np.count_nonzero(slurp) == 0
        if zgyReaderFactory == newzgy.ZgyReader:
            assert reader.readconst((0,0,0), reader.size) == 0

def testEmptyExistingFile(filename, zgyReaderFactory = newzgy.ZgyReader):
    """
    Access a file that has already been created by the old ZGY accessor
    with no written bricks and an invalid coding range.
    To create, use the old ZGY-Public Python wrapper:
        with zgy.ZgyWriter("OldEmpty2.zgy", size=(512, 640, 1000),
                           datarange=(101,101), datatype="int16") as w: pass
    Can leave the file locally, or upload with ZGY, or with sdutil.
    Currently the latter is the most interesting case to test.
    """
    #print('testEmptyExistingFile("{0}")'.format(filename))
    #print(' -> Using ' + showZgy(zgyReaderFactory))
    with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
        if zgyReaderFactory == oldzgy.ZgyReader:
            slurp = np.ones(reader.size, dtype=np.float32)
            reader.read((0,0,0), slurp)
            value = slurp[0,0,0] if np.all(slurp.flat == slurp[0,0,0]) else None
        else:
            value = reader.readconst((0,0,0), reader.size, as_float=True)
        #print(" -> VALUE", value, "RANGE", reader.datarange)
        # In spite of the 101..101 coding range, the file will contain
        # all zeros. In the new accessor the coding range is rejected
        # as bad, no conversion is done, so empty bricks read as zero.
        # In the old accessor there is a "feature" that cause empty
        # bricks to read as zero regardless of whether caller wants conversion.
        assert value == 0

def testRmwFile(filename, zgyWriterFactory = newzgy.ZgyWriter):
    """
    The layout of this test data is described in detail in doc/testdata-rmw.png.
    """
    rmwsize = (((0,0,0),    (304,64,384)), # Survey size.
               ((0,0,192),  (304,64,384)), # Half the survey set to constant "1".
               ((28,0,84),  (144,64,304)), # Touches 12 bricks.
               ((40,0,100), (160,64,288)), # Touches 12 bricks.
               ((204,0,0),  (216,64,384)), # Tall, thin, to fill up this segment.
               ((52,0,120), (176,64,272)), # Touches 12 bricks.
               ((256,0,0),  (304,64,352)), # Constant-value at survey edge.
               ((0,0,256),  (64,64,320))) # Normal brick changed to constant.

    surveysize = rmwsize[0][1]
    expect = np.zeros(surveysize, dtype=np.float32)
    partnum = 0
    for part in rmwsize[1:]:
        partnum += 1
        beg, end = part
        #print("part", part, "beg", beg, "end", end)
        expect[beg[0]:end[0],beg[1]:end[1],beg[2]:end[2]] = partnum

    with zgyWriterFactory(filename,
                       iocontext = SDCredentials(segsize=11/4),
                       size = surveysize,
                       datatype = SampleDataType.int8,
                       datarange = (-28,+227),
                       zunitdim = UnitDimension.time,
                       zunitname = "ms",
                       zunitfactor = 0.001,
                       hunitdim = UnitDimension.length,
                       hunitname = "ft",
                       hunitfactor = 0.3048,
                       zstart = 2500,
                       zinc = 4.125,
                       annotstart = (1234, 5678),
                       annotinc = (5, 2),
                       corners = ((1000, 1000),
                                  (1005, 1000),
                                  (1000, 1002),
                                  (1005, 1002))
    ) as writer:
        partnum = 0
        sizes = [(0,)]
        for part in rmwsize[1:]:
            partnum += 1
            beg, end = part
            size = (end[0]-beg[0], end[1]-beg[1], end[2]-beg[2])
            #print("part", part, "beg", beg, "end", end, "size", size)
            if partnum == 1:
                # Just doing this to exercise both the write functions.
                data = np.full(size, partnum, dtype=np.float32)
                writer.write(beg, data)
            else:
                data = np.float32(partnum)
                writer.writeconst(beg, data, size=size, is_storage=False)
            if filename[:5] == "sd://":
                closed_sizes = tuple(writer._fd._relay._sizes)
                opened_sizes = tuple([len(writer._fd._open_segment)])
                sizes.append(closed_sizes + opened_sizes)
            else:
                sizes.append((writer._fd.xx_eof,))

        #print(sizes)
        sizes_in_bricks = []
        for e in sizes:
            for bytecount in e:
                assert all([(bytecount % 64) == 0 for bytecount in e])
            sizes_in_bricks.append(tuple(np.array(e, dtype=np.int64) // (256*1024)))

        # The expected results have been computed by hand.
        # See testdata-rmw.svg for a detailedexplanation with figures.
        #print(sizes_in_bricks)
        local = filename[:5] != "sd://"
        assert sizes_in_bricks[1] == (( 1,) if local else (1, 0))
        assert sizes_in_bricks[2] == ((11,) if local else (1, 10))
        assert sizes_in_bricks[3] == ((11,) if local else (1, 10))
        assert sizes_in_bricks[4] == ((17,) if local else (1, 11, 5))
        assert sizes_in_bricks[5] == ((17,) if local else (1, 11, 11, 4))
        assert sizes_in_bricks[6] == ((18,) if local else (1, 11, 11, 5))
        assert sizes_in_bricks[7] == ((18,) if local else (1, 11, 11, 6))

    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        # Read the entire survey, excluding padding bytes, in a single
        # operation. Compare with the survey built in memory.
        slurp = np.zeros(reader.size, dtype=np.float32)
        reader.read((0,0,0), slurp)
        assert np.all(slurp == expect)

        # Check each brick for whether it takes up space in the file or
        # is flagged as constant value. The expected result is explained
        # in the textual- and image descriptionof the test data.
        is_const = np.zeros((5, 6), dtype=np.float32)
        for ii in range(0, 320, 64):
            for kk in range(0, 384, 64):
                c = reader.readconst((ii, 0, kk), (64, 64, 64))
                is_const[ii//64, kk//64] = -1 if c is None else c

        expect_const = np.array([[0, -1, -1, -1, -1, 1],
                                 [0, -1, 5, 5, -1, 1],
                                 [0, -1, -1, -1, -1, 1],
                                 [-1, -1, -1, -1, -1, -1],
                                 [6, 6, 6, 6, 6, -1]], dtype=np.float32)
        assert np.all(is_const == expect_const)

def testNoRmwInCompressedFile(filename):
    lossy = ZgyCompressFactory("ZFP", snr=30)
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(), size=(100, 64, 64), compressor=lossy) as w:
        # Writing a constant value should not  prevent overwriting later.
        w.writeconst((0,0,0), value=42, size=w.size, is_storage=False)
        # Write part of a brick for the first time.
        data = np.arange(50*64*64, dtype=np.float32).reshape((50, 64, 64))
        w.write((0,0,0), data)
        # Write needing to update the first brick.
        with MustThrow("Updating a local BrickStatus.Compressed brick with Compressed data is illegal"):
            w.write((50,0,0), data)
        # The above error might have set the global _is_bad flag, in spite of
        # this being a recoverable user error. But it probably doesn't
        # matter much either way.
        w.errorflag = False
        # Write entire survey. This is an update, but no read/modify/write.
        # The old brick will be leaked if new one compresses larger.
        data = np.arange(100*64*64, dtype=np.float32).reshape((100, 64, 64))
        with MustThrow("Updating a local BrickStatus.Compressed brick with Compressed data is illegal"):
            w.write((0,0,0), data)
        w.errorflag = False
        # This should actually have been set when we opened the file,
        # that feature isn't implemented yet. Besides, for the purpose
        # of this test I need to change it while the file is in use.
        w._accessor._update_mode = UpdateMode.Pedantic
        w.write((0,0,0), data)

def testFatalErrorFlag(filename):
    class BogusFile:
        def close(self): pass

    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(), size=(100, 64, 64)) as w:
        data = np.arange(64*64*64, dtype=np.float32).reshape(64, 64, 64)
        w.write((0,0,0), data)
        w.write((0,0,0), data)
        hack = w._accessor._file._file
        w._accessor._file._file = BogusFile()
        with MustThrow("BogusFile", AttributeError):
            w.write((0,0,0), data)
        w._accessor._file._file = hack
        # File is now usable again, but the global error flag is set.
        with MustThrow("previous errors"):
            w.write((0,0,0), data)
        # Explicitly reset it and we should be good.
        w.errorflag = False
        w.write((0,0,0), data)
        # Another bad write
        w._accessor._file._file = BogusFile()
        with MustThrow("BogusFile", AttributeError):
            w.write((0,0,0), data)
        # Verify that lod generation and meta flush is either
        # turned off or is ignoring errors. The final close()
        # of the python file descriptor will not throw because
        # BogusFile wraps close().
        w.close()
        hack.close()

def testLargeSparseFile(filename, zgyWriterFactory, zgyReaderFactory):
    size = (5000, 6000, 1000)
    wbeg = (1000, 9000)
    wend = (wbeg[0] + 10 * (size[0]-1), wbeg[1] + 10 * (size[1]-1))
    if zgyWriterFactory:
        with zgyWriterFactory(filename,
                              iocontext = SDCredentials(),
                              size = size,
                              datatype = SampleDataType.int8,
                              datarange = (-28,+227),
                              zunitdim = UnitDimension.time,
                              zunitname = "ms",
                              zunitfactor = 0.001,
                              hunitdim = UnitDimension.length,
                              hunitname = "ft",
                              hunitfactor = 0.3048,
                              zstart = 2500,
                              zinc = 4.125,
                              annotstart = (1234, 5678),
                              annotinc = (5, 2),
                              corners = ((wbeg[0], wbeg[1]),
                                         (wend[0], wbeg[1]),
                                         (wbeg[0], wend[1]),
                                         (wend[0], wend[1]))) as writer:
            writer.write((size[0]-1, size[1]-1, 0), np.array([[[42, 10, 10]]], dtype=np.int8))
            writer.finalize(progress=ProgressWithDots(), decimation=[DecimationType.Maximum])
    if zgyReaderFactory:
        with zgyReaderFactory(filename, iocontext = SDCredentials()) as reader:
            assert reader.size == size
            data = np.zeros((1,1,4), dtype=np.int8)
            pos = np.array((size[0]-1, size[1]-1, 0), dtype=np.int64)
            reader.read(pos, data, lod=0)
            assert tuple(data.flat) == (42, 10, 10, -100)
            reader.read(pos//2, data, lod=1)
            assert tuple(data.flat) == (42, 10, -100, -100)
            for lod in range(2,8):
                reader.read(pos//(1<<lod), data, lod=lod)
                assert tuple(data.flat) == (42, -100, -100, -100)

def testNaan(filename, snr = -1):
    compressor = ZgyCompressFactory("ZFP", snr = snr) if snr > 0 else None
    with newzgy.ZgyWriter(filename,
                          compressor = compressor,
                          iocontext = SDCredentials(),
                          size = (256, 128, 128),
                          datatype = SampleDataType.float) as writer:
        data = np.zeros((64, 64, 64), dtype=np.float32)
        count_nan = 0
        count_inf = 0
        counts = np.zeros(256, dtype=np.int32)

        # Some NaN, a few other different values, mostly zero.
        data.fill(0)
        data[0,0,:3] = np.nan
        data[0,0,3] = 2
        data[0,0,4] = 3
        writer.write((0,0,0), data)
        count_nan += 3
        counts[2] += 1
        counts[3] += 1

        # Some NaN, only one other value (42)
        data.fill(42)
        data[0,0,:5] = np.nan
        writer.write((64,0,0), data)
        count_nan += 5
        counts[42] += (64*64*64) - 5

        # NaN only
        data.fill(np.nan)
        writer.write((128,0,0), data)
        count_nan += (64*64*64)

        # NaN explicitly written as constant value
        writer.writeconst((192, 0, 0), np.nan, (64, 64, 64), is_storage=False)
        count_nan += (64*64*64)

        # Now repeat for +/- inf
        # Some Inf, a few other different values. Mostly zero.
        data.fill(0)
        data[0,0,0] = np.inf
        data[0,0,1] = -np.inf
        data[0,0,2] = np.inf
        data[0,0,3] = 3
        data[0,0,4] = 4
        writer.write((0,64,0), data)
        count_inf += 3
        counts[3] += 1
        counts[4] += 1

        # Some Inf, only one other value (255).
        data.fill(255)
        data[0,0,:13] = np.inf
        data[0,1,:10] = -np.inf
        writer.write((64,64,0), data)
        count_inf += 23
        counts[255] = (64*64*64) - 23

        # +Inf only
        data.fill(np.inf) # 64^3 Inf
        writer.write((128,64,0), data)
        count_inf += (64*64*64)

        # -Inf explicitly written as constant value
        writer.writeconst((192, 64, 0), -np.inf, (64, 64, 64), is_storage=False)
        count_inf += (64*64*64)

        counts[0] = 256*128*128 - np.sum(counts[1:]) - count_nan - count_inf
        writer.finalize(decimation = [DecimationType.Average])

        # Exercise logging & debug code in the compression module.
        # Discard the output. Yes, this is a shameless trick to
        # increase coverage. But in Python a test that only checks
        # that a function is callable is in fact somewhat useful.
        if compressor is not None:
            with io.StringIO() as devnull:
                compressor.dump(msg=None, outfile=devnull,
                                text=True, csv=True, reset=True)

    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        # --- statistics and histogram ---

        #print(reader.statistics)
        #print(reader.histogram)
        #print(list(counts))
        #print("Expect total size", 256*128*128,
        #      "nan", count_nan,
        #      "inf", count_inf,
        #      "valid", 256*128*128 - count_nan - count_inf)
        #print("Got valid",
        #      "stats", reader.statistics.cnt,
        #      "histo", reader.histogram.cnt,
        #      "sampl", np.sum(reader.histogram.bin))
        # Limits are set automatically to the value range. I carefully
        # chose 0..255 since the histogram then has one bin per sample value.
        assert reader.histogram.min == 0 and reader.histogram.max == 255
        h = reader.histogram.bin
        for i in range(256):
            if counts[i] != h[i]:
                print("Histogram bin", i, "expected", counts[i], "actual", h[i])
        assert reader.statistics.cnt == 256*128*128 - count_nan - count_inf
        assert reader.histogram.cnt == 256*128*128 - count_nan - count_inf
        assert np.all(np.array(reader.histogram.bin) == counts)
        #assert reader.statistics.inf == count_nan + count_inf # not in api

        # --- bricks stored as all-constant or not ---

        BRICK = (64, 64, 64)

        assert reader.readconst((0,0,0), BRICK) is None
        assert reader.readconst((64,0,0), BRICK) is None
        assert np.isnan(reader.readconst((128,0,0), BRICK))
        assert np.isnan(reader.readconst((192,0,0), BRICK))

        assert reader.readconst((0,64,0), BRICK) is None
        assert reader.readconst((64,64,0), BRICK) is None
        assert reader.readconst((128,64,0), BRICK) == np.inf
        assert reader.readconst((192,64,0), BRICK) == -np.inf

        # -- read back samples ---

        reader.read((0,0,0), data)
        assert np.all(np.isnan(data[0,0,:3]))
        assert data[0,0,3] == 2
        assert data[0,0,4] == 3
        assert np.count_nonzero(data) == 5

        reader.read((64,0,0), data)
        assert np.all(np.isnan(data[0,0,:5]))
        assert np.count_nonzero(data == 42) == 64*64*64 - 5

        reader.read((0,64,0), data)
        assert data[0,0,0] == np.inf
        assert data[0,0,1] == -np.inf
        assert data[0,0,2] == np.inf
        assert data[0,0,3] == 3
        assert data[0,0,4] == 4
        assert np.count_nonzero(data) == 5

        reader.read((64,64,0), data)
        assert np.all(data[0,0,:13] == np.inf)
        assert np.all(data[0,1,:10] == -np.inf)
        assert np.count_nonzero(data == 255) == 64*64*64 - 13 - 10

        # --- read back low resolution ---

        # LOD1 should be sufficient to test.
        # Note that this only tests a single decimation algorithm
        # and the functions that call it. There needs to be separate
        # unit tests to verify that all decimation algorithms have a
        # reasonable behavior for nan and inf.

        fullres = np.zeros((128, 128, 128), dtype=np.float32)
        reader.read((0,0,0), fullres, lod=0)
        reader.read((0,0,0), data, lod=1)
        # Input first trace: nan, nan, nan, 2, 3
        # An extra slop factor is needed because calculation done in float32.
        assert math.isclose(data[0,0,0], 0, rel_tol=1.0e-5)   # 2 NaN (skipped), the rest zero.
        assert math.isclose(data[0,0,1], 2/7, rel_tol=1.0e-5) # 1 NaN (skipped), 1 "2", rest "0"
        assert math.isclose(data[0,0,2], 3/8, rel_tol=1.0e-5) # one "3", rest default to zero

        # Input trace: 5*nan, rest is 42. With "Average" decimation
        # each output sample found at least one finite value.
        assert np.all(data[32:64, 0:32, 0:32] == 42)

        # Input trace: +inf, -inf, +inf, 3, 4. All others 0.
        # Note: The C++ code skips +/- inf. Numpy includes them unless
        # told otherwise, and the average of +inf and -inf is NaN.
        # These rules are pretty obscure and it is probably easier to
        # TODO-Low adopt the C++ strategy both places.
        #showdecimation(fullres[0:2,64:66,0:20], data[0,32,0:10])
        assert np.isnan(data[0,32,0])
        assert data[0,32,1] == np.inf
        assert math.isclose(data[0,32,2], 4/8, rel_tol=1.0e-5) # one "4", rest default to zero

        # Input trace: 13 * +inf in one trace, 10 * -inf in another.
        # So the first 5 samples have average(-inf,+inf) => nan
        # the next 2 samples have average(255,+inf) => +inf
        # Everything else should be 255.
        # UPDATE: In the C++ version (and soon also Python)
        # +/- inf is ignored so all decimated samples are 255.
        #showdecimation(fullres[64:66,64:66,0:20], data[32,32,0:10])
        assert np.all(np.isnan(data[32,32,:5]))
        assert data[32,32,5] == np.inf
        assert data[32,32,6] == np.inf
        assert data[32,32,7] == 255

        # Now read the brick built from all-constant input.
        reader.read((64,0,0), data, lod=1)
        d1 = data[:32,:32,:32] # from data written at (128,0,0)
        d2 = data[32:,:32,:32] # from data written at (192,0,0)
        d3 = data[:32,32:,:32] # from data written at (128,64,0)
        d4 = data[32:,32:,:32] # from data written at (192,64,0)
        assert np.all(np.isnan(d1))
        assert np.all(np.isnan(d2))
        assert np.all(d3 == np.inf)
        assert np.all(d4 == -np.inf)

def testWriteNaanToIntegerStorage(filename):
    with newzgy.ZgyWriter(filename,
                          size = (256, 128, 128),
                          iocontext = SDCredentials(),
                          datatype = SampleDataType.int8,
                          datarange = (-128,+127)
    ) as writer:
        data = np.zeros((64, 64, 64), dtype=np.float32)
        data[0,0,42] = np.nan
        writer.write((0,0,0), data)

def testZeroCentric(filename):
    """
    Specific test for the zero-centric property. When the hard coded
    (in this test) datarange is zero-centric then the rounding makes
    an equal number of small positive and small negative numbers
    end up being returned as zero after a roundtrip.
    """
    data = np.array([[[
        -1.4, -1.2, -1.0, -0.8, -0.6,
        -0.4, -0.2, +0.0, +0.2, +0.4,
        +0.6, +0.8, +1.0, +1.2, +1.4,
        100.0, 200.0]]], dtype=np.float32)
    expect = np.array([[[
        -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        100, 200]]], dtype=np.float32)
    with newzgy.ZgyWriter(filename,
                          iocontext = SDCredentials(),
                          size = (64, 64, 64),
                          datatype = SampleDataType.int8,
                          datarange = (-28,+227),
    ) as writer:
        writer.write((0,0,0), data)
    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        actual = np.zeros((1, 1, expect.size), dtype=np.float32)
        reader.read((0,0,0), actual)
    assert np.all(np.isclose(expect, actual))

def testFinalizeProgress(filename, abort = False):
    """
    Check the progress callback that can be installed while generating
    low resolution bricks. Optionally check that the callback can be
    used to abort the generation.
    """
    class Progress:
        def __init__(self, abort = False):
            self._abort = abort
            self._complete = False
            self._waszero = False
        def __call__(self, done, total):
            self._complete = bool(done == total)
            self._waszero = self._waszero or done == 0
            #print("done {0}/{1}".format(done, total))
            return not abort or done < total//4

    with newzgy.ZgyWriter(filename,
                          iocontext = SDCredentials(),
                          size = (112+640, 64+320, 176),
    ) as writer:
        writer.write((16,16,16), np.full((40,41,42), 31, dtype=np.float32))
        writer.write((48,20,24), np.full((72,10,16), 97, dtype=np.float32))
        writer.write((0,0,64),   np.full((112,64,64), 0, dtype=np.float32))
        writer.write((512,0,0),  np.full((128,128,64), 42, dtype=np.float32))
        progress = Progress(abort)
        if abort:
            # The progress callback will return False on 25% done.
            with MustThrow(extypes = newzgy.ZgyAborted):
                writer.finalize(progress=progress)
            assert progress._waszero
            assert not progress._complete
        else:
            writer.finalize(progress=progress)
            assert progress._waszero
            assert progress._complete

def testHugeFile(filename):
    """
    Create a very sparse file where the declared size is large enough
    to make the header area > 1 MB. This can trigger some issues.
    Number of bricks:
        Lod 0: 64*64*32 bricks = 131072
        Lod 1: 32*32*16 bricks = 16384
        Lod 2: 16*16*8 bricks  = 2048
        Lod 3: 8*8*4 bricks    = 256
        Lod 4: 4*4*2 bricks    = 32
        Lod 5: 2*2*1 bricks    = 4
        Lod 6: 1*1*1 brick     = 1
        SUM: 149797 bricks, 1.14 Mb of brick lookup tables
    Rounded up to brick size there will be 1.25 MB of headers.
    Non-constant bricks: Only one per layer. 1.75 MB total
    Total file size: 3 MB.
    """
    with newzgy.ZgyWriter(filename,
                          iocontext = SDCredentials(),
                          datatype = SampleDataType.int8,
                          datarange = (-128,+127),
                          size = (64*64, 64*64, 32*64),
    ) as writer:
        writer.write((640,512,0), np.full((64,64,65), 42, dtype=np.float32))
        #writer.finalize(progress=ProgressWithDots())
    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        assert reader.nlods == 7
        c1 = reader.readconst((640,512,0), (64,64,64))
        c2 = reader.readconst((640,512,64), (64,64,64))
        c3 = reader.readconst((640,512,129), (64,64,64))
    assert c1 == 42   # writer detected it was constant
    assert c2 is None # partly written
    assert c3 == 0    # never written
    assert os.stat(filename).st_size == 3 * 1024 * 1024

def testDecimateOddSize(filename):
    """
    At the survey edge, the decimation that normally has 8 samples input
    might only have 4, 2, or 1. Make sure the code doesn't include
    the padding in its computation.
    """
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(),
                          size = (7, 13, 64+17)
    ) as writer:
        data = np.full(writer.size, 200, dtype=np.float32)
        data[0::2,:,:] = 100
        data[:,0::2,:] = 50
        assert np.all(data[:,:,:] == data[:,:,0:1])
        writer.write((0,0,0), data)
        writer.finalize(decimation = [DecimationType.Average])

    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        assert reader.nlods == 2
        data = np.zeros((4, 7, 32+9), dtype=np.float32)
        reader.read((0,0,0), data, lod=1)

        # Within each trace all samples should be the same, also
        # the last one, since this is true also for the input.
        assert np.all(data[:,:,:] == data[:,:,0:1])

        # Most output values will be avg(200, 100, 50, 50) = 100.
        # At the edges in i/j it should be average(50, 100) or (50,50).
        # At the corner expect average(50) i.e. 50.
        # If the implemenation erroneously tried to read the
        # padding (which ought to be zero) the numbers will be lower.
        # Currently in OpenZGY/C++ the samples not based on 8 neighbors
        # might be set to 0.
        assert np.all(data[:3, :6, :] == 100)
        assert np.all(data[:3, 6, :] == 50)
        assert np.all(data[3, :6, :] == 75)
        assert np.all(data[3, 6, :] == 50)


def testDecimateWeightedAverage(filename):
    """
    As test.lodalgo.testSpecial but very simplified, just to make sure
    the default lod2 algorithm is in fact WeightedAverage. The lod1
    default is LowPass; to avoid this getting in the way I will
    make all traces constant-value. This makes LowPass behave as
    Decimate (or Average, or Median, etc.)
    """
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(),
                          size = (64, 256, 512)
    ) as writer:
        data = np.zeros((64, 64, 512), dtype=np.float32)
        # 1/4 brick of 300, 3/4 brick of 100, 3 bricks of unwritten 0.
        data[:16,:,:] = 300
        data[16:,:,:] = 100
        tiny = np.array([[300, 300, 0, 0],
                         [300, 300, 0, 0],
                         [0, 0, 100, 100],
                         [0, 0, 100, 100]], dtype=np.float32)
        # In lod 1 this will be just 300, 0, 0, 1000
        tiny = tiny.reshape((4,4,1))
        data[:4,:4,:] = tiny
        assert np.all(data[:,:,:] == data[:,:,0:1])
        writer.write((0,0,0), data)
        #writer.finalize(decimation = [DecimationType.Average])

    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        assert reader.nlods >= 3

        # Checking the lowpass output, including the fact that it is
        # supposed to have zero DC bias.
        data = np.zeros((2, 2, 256), dtype=np.float32)
        reader.read((0,0,0), data, lod=1)
        #print(data[:,:,0])
        assert np.all(np.isclose(data[0,0,:], 300))
        assert np.all(np.isclose(data[0,1,:], 0))
        assert np.all(np.isclose(data[1,0,:], 0))
        assert np.all(np.isclose(data[1,1,:], 100))

        data = np.zeros((1, 1, 1), dtype=np.float32)
        reader.read((0,0,0), data, lod=2)
        # average(300, 0, 0, 100) is 100 but we expect something closer to
        # 300 since this value is relatively more scarce.
        #print(data)
        assert data.flat[0] > 200

def testMixingUserAndStorage(filename):
    """
    When the file has an integer type both reading and writing can be done
    either in float user sample values or in integral storage values.
    Try all 4 combinations.
    """
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(),
                          datatype = SampleDataType.int8, datarange = (-2,+763),
                          size = (64, 64, 512)
    ) as writer:
        # user = 3*storage  + 382
        # storage = (user - 382) / 3
        # user 3 -> storage -126.33 -> -126 -> user 4
        # user 12 -> storage -123.33 -> -123 -> user 13
        # user 40 -> storage -114
        # user 71 -> storage -103.66 -> -104 -> user 70

        w1 = np.zeros((64, 64, 64), dtype=np.float32)
        w2 = np.zeros((64, 64, 64), dtype=np.float32)
        w3 = np.zeros((64, 64, 64), dtype=np.int8)
        w4 = np.zeros((64, 64, 64), dtype=np.int8)

        w1[0,0,0] = 3.0   # user  4 <-> storage -126
        w2[0,0,0] = 12.0  # user 13 <-> storage -123
        w3[0,0,0] = -114  # user 40 <-> storage -114
        w4[0,0,0] = -104  # user 70 <-> storage -104

        writer.write((0,0,0), w1)
        writer.write((0,0,64), w2)
        writer.write((0,0,128), w3)
        writer.write((0,0,192), w4)

    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:

        r1 = np.zeros((1, 1, 1), dtype=np.float32)
        r2 = np.zeros((1, 1, 1), dtype=np.int8)
        r3 = np.zeros((1, 1, 1), dtype=np.float32)
        r4 = np.zeros((1, 1, 1), dtype=np.int8)

        reader.read((0,0,0), r1)
        reader.read((0,0,64), r2)
        reader.read((0,0,128), r3)
        reader.read((0,0,192), r4)

        #print("expect", 4.0, -123, 40.0, -114)
        #print("actual", r1.flat[0], r2.flat[0], r3.flat[0], r4.flat[0])
        assert np.isclose(r1.flat[0], 4.0)
        assert r2.flat[0] == -123
        assert np.isclose(r3.flat[0], 40.0)
        assert r4.flat[0] == -104

def testSmallConstArea(filename):
    """
    Check what happens when writeconst() is called with a region
    smaller than one brick. Application code might well specify
    a region that doesn't align with the bricks. Actually writing
    less than a brick in total would be odd, but the corner cases
    that need to be handled are similar.
    """
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(),
                          datatype = SampleDataType.int8, datarange = (-128,+127),
                          size = (64, 64, 256)
    ) as writer:
        writer.writeconst((0,0,128), 42, size=(64,64,128), is_storage=True)
        # unwritten brick, value matches defaultvalue   -> mark as const
        # unwritten brick, value does not match default -> inflate
        # const brick, value matches previous brick     -> no-op
        # const brick, value differs                    -> inflate
        writer.writeconst((1,2,3+0),   0, size=(11,12,13), is_storage=True)
        writer.writeconst((1,2,3+64),  15, size=(11,12,13), is_storage=True)
        writer.writeconst((1,2,3+128), 42, size=(11,12,13), is_storage=True)
        writer.writeconst((1,2,3+192), 67, size=(11,12,13), is_storage=True)

    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        BRICK = (64,64,64)
        r1 = reader.readconst((0,0,0),   BRICK, as_float = False)
        r2 = reader.readconst((0,0,64),  BRICK, as_float = False)
        r3 = reader.readconst((0,0,128), BRICK, as_float = False)
        r4 = reader.readconst((0,0,192), BRICK, as_float = False)

    #print("testSmallConstArea:", r1, r2, r3, r4)
    assert r1 == 0    # Was converted from "unwritten" to "const zero"
    assert r2 is None # Brick now contains a mix of 0 and 15.
    assert r3 == 42   # No-op; the brick already contained const 42.
    assert r4 is None # Brick now contains a mix of 42 and 67.

onevalue_t = namedtuple("result", "range stats histo stats_count histo_count bins")
def testHistoOneValue(filename, dtype, value, fill, *, datarange = None, verbose = False):
    if verbose:
        print("Test dtype", dtype, "value", value,
              ("only" if fill else "and unwritten bricks"))
    center = value if np.isfinite(value) else -0.25
    with newzgy.ZgyWriter(filename, iocontext = SDCredentials(),
                          size = (64, 64, 3*64),
                          datatype = dtype,
                          datarange = datarange or (center-1, center+1)
    ) as writer:
        if np.isfinite(value):
            writer.writeconst((0, 0, 0), value,
                              size=(64, 64, 64), is_storage=False)
            if fill:
                writer.writeconst((0, 0, 64), value,
                                  size=(64, 64, 128), is_storage=False)
        writer.finalize(force=True)

    with newzgy.ZgyReader(filename, iocontext = SDCredentials()) as reader:
        if verbose:
            print("Data range", reader.datarange)
            print("Statistics", reader.statistics)
            print("Histogram ", (reader.histogram.min, reader.histogram.max))
        return onevalue_t((reader.datarange[0], reader.datarange[1]),
                          (reader.statistics.min, reader.statistics.max),
                          (reader.histogram.min, reader.histogram.max),
                          reader.statistics.cnt,
                          np.sum(reader.histogram.bin),
                          reader.histogram.bin)

def testHistoCornercaseFloat(filename):

    # Float: datarange with zero size is valid on input,
    # in fact the data range isn't specified by the user.
    # Reading back data gives the statistical range
    # which for float may include defaultvalue.
    # The histogram will use the fuzzy algorithm.

    # The numbers in brackets correspond to the ones in
    # GenLodImpl::suggestHistogramRange().

    # [3] nothing written.
    # Note that the writer might need to pass force=True to finalize()
    # to get the histogram- and statistics information written out even
    # when no actual data has been written. I am unsure about how the
    # principle of least surprise applies here. As of Oct 2020 the force
    # is required. See the ZgyWriter constructor setting _dirty(False).

    BRICK = 64*64*64
    r = testHistoOneValue(filename, SampleDataType.float, np.nan, False)

    assert r.range == r.stats
    assert r.histo_count == r.stats_count
    assert r.stats == (0, 0)
    assert r.histo == (-128, +127)
    assert r.stats_count == 3*BRICK # Assuming finalize with force=True
    assert r.bins[128] == r.histo_count

    # [4] one all zero brick, two never written.
    # Expected result same as for nothing written.
    r = testHistoOneValue(filename, SampleDataType.float, 0, False)

    assert r.range == r.stats
    assert r.histo_count == r.stats_count
    assert r.stats == (0, 0)
    assert r.histo == (-128, +127)
    assert r.stats_count == 3*BRICK
    assert r.bins[128] == r.histo_count

    # [4] three all zero bricks.
    # Expected result same as for nothing written.
    r = testHistoOneValue(filename, SampleDataType.float, 0, True)

    assert r.range == r.stats
    assert r.histo_count == r.stats_count
    assert r.stats == (0, 0)
    assert r.histo == (-128, +127)
    assert r.stats_count == 3*BRICK
    assert r.bins[128] == r.histo_count

    # [6] single negative value, plus two never written bricks.
    # The statistics and histogram include the never-written
    # samples as if they were zero.
    # Note: I won't be testing the "some never written" scenario
    # for every remaining case; it is hopefully enough to
    # confirm once that never-written is treated as written-zero.
    r = testHistoOneValue(filename, SampleDataType.float, -42, False)

    assert r.range == r.stats
    assert r.histo_count == r.stats_count
    assert r.stats == (-42, 0)
    assert r.histo == (-42, 0)
    assert r.stats_count == 3*BRICK
    assert r.bins[0] == BRICK
    assert r.bins[255] == 2*BRICK

    # [6] single negative value in all three bricks.
    # The value range and the statistics should have the True
    # range i.e. low==high and the histogram range should be wider.
    r = testHistoOneValue(filename, SampleDataType.float, -42, True)

    assert r.range == r.stats
    assert r.histo_count == r.stats_count
    assert r.stats == (-42, -42)
    assert r.histo == (-42, 0)
    assert r.stats_count == 3*BRICK
    assert r.bins[0] == 3*BRICK
    assert r.bins[255] == 0

    # [6] single positive value in all three bricks.
    # Result similar to the above but the ranges are swapped.
    r = testHistoOneValue(filename, SampleDataType.float, +42, True)

    assert r.range == r.stats
    assert r.histo_count == r.stats_count
    assert r.stats == (42, 42)
    assert r.histo == (0, 42)
    assert r.stats_count == 3*BRICK
    assert r.bins[0] == 0
    assert r.bins[255] == 3*BRICK

def testHistoCornercaseInt(filename):

    # Integral data.
    # Histogram range should always match the user provided range,
    # which for never-written is -1.25 to +0.75 and for the
    # remaining cases value +/- 1. This means that value won't be
    # exactly representable as an integer (it maps to -0.5) and
    # this will be noticeable in the statistics. The 0.5 factor
    # may also lead to numerical instability. The samples end up
    # either in bin 127 or bin 128.
    # Also, range might be wider then statistics (unlike the float
    # case) if not all possible storage values have been used.

    BRICK = 64*64*64
    r = testHistoOneValue(filename, SampleDataType.int8, np.nan, False)

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats[0] == r.stats[1]
    assert abs(r.stats[0] - 0) < 0.25
    assert abs(r.stats[0] - 0) > 0.001 # 0.0 not representable.
    assert r.histo[0] == -1.25 and r.histo[1] == 0.75 # user choice exactly.
    assert r.stats_count == 3*BRICK # Assuming finalize with force=True
    # I don't really care where the "0" samples end up. It won't be the center.
    assert r.bins[127] + r.bins[128] == 0

    r = testHistoOneValue(filename, SampleDataType.int8, 0, True)

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats[0] == r.stats[1]
    assert abs(r.stats[0] - 0) < 0.25
    assert abs(r.stats[0] - 0) > 0.001 # 0.0 not representable.
    assert r.histo[0] == 0-1 and r.histo[1] == 0+1 # user choice exactly.
    assert r.stats_count == 3*BRICK
    assert r.bins[127] + r.bins[128] == 3*BRICK

    r = testHistoOneValue(filename, SampleDataType.int8, -42, True)

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats[0] == r.stats[1]
    assert abs(r.stats[0] + 42) < 0.25
    assert abs(r.stats[0] + 42) > 0.001 # 42.0 not representable.
    assert r.histo[0] == -42-1 and r.histo[1] == -42+1 # user choice exactly.
    assert r.stats_count == 3*BRICK
    assert r.bins[127] + r.bins[128] == 3*BRICK

    r = testHistoOneValue(filename, SampleDataType.int8, +42, True)

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats[0] == r.stats[1]
    assert abs(r.stats[0] - 42) < 0.25
    assert abs(r.stats[0] - 42) > 0.001 # 42.0 not representable.
    assert r.histo[0] == 42-1 and r.histo[1] == 42+1 # user choice exactly.
    assert r.stats_count == 3*BRICK
    assert r.bins[127] + r.bins[128] == 3*BRICK

    # 16 bit not much different from 8 bit, but the statistics will be
    # closer to the supplied value because the quantization error is smaller.
    r = testHistoOneValue(filename, SampleDataType.int16, np.nan, False)

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats[0] == r.stats[1]
    assert abs(r.stats[0] - 0) < 0.25/256
    assert abs(r.stats[0] - 0) > 0.001/256 # 0.0 not representable.
    assert r.histo[0] == -1.25 and r.histo[1] == 0.75 # user choice exactly.
    assert r.stats_count == 3*BRICK
    # I don't really care where the "0" samples end up. It won't be the center.
    assert r.bins[127] + r.bins[128] == 0

    r = testHistoOneValue(filename, SampleDataType.int16, 0, True)

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats[0] == r.stats[1]
    assert abs(r.stats[0] - 0) < 0.25/256
    assert abs(r.stats[0] - 0) > 0.001/256 # 0.0 not representable.
    assert r.histo[0] == 0-1 and r.histo[1] == 0+1 # user choice exactly.
    assert r.stats_count == 3*BRICK
    assert r.bins[127] + r.bins[128] == 3*BRICK

    r = testHistoOneValue(filename, SampleDataType.int16, -42, True)

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats[0] == r.stats[1]
    assert abs(r.stats[0] + 42) < 0.25/256
    assert abs(r.stats[0] + 42) > 0.001/256 # 42.0 not representable.
    assert r.histo[0] == -42-1 and r.histo[1] == -42+1 # user choice exactly.
    assert r.stats_count == 3*BRICK
    assert r.bins[127] + r.bins[128] == 3*BRICK

    r = testHistoOneValue(filename, SampleDataType.int16, +42, True)

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats[0] == r.stats[1]
    assert abs(r.stats[0] - 42) < 0.25/256
    assert abs(r.stats[0] - 42) > 0.001/256 # 42.0 not representable.
    assert r.histo[0] == 42-1 and r.histo[1] == 42+1 # user choice exactly.
    assert r.stats_count == 3*BRICK
    assert r.bins[127] + r.bins[128] == 3*BRICK

    # Behavior when all explicitly written values get clipped.
    # Expect both the histogram and the statistics to only reflect
    # the clipped value (-5) as if that value and not -42 had been
    # written.
    r = testHistoOneValue(filename, SampleDataType.int8, -42, True,
                          datarange = (-5, +760))

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats == (-5, -5)
    assert r.histo == (-5, +760)
    assert r.stats_count == 3*BRICK
    assert r.bins[0] == 3*BRICK

    # As above, all explicitly written values get clipped but now
    # there are a few unwritten bricks. Expect both the histogram
    # and the statistics to only reflect the clipped value (-5) as
    # if that value and not -42 had been written.
    # Defaultvalue is +1 because the range does not give a zero
    # centric histogram. The statistics should also reflect that.
    # I.e. expect +1 to be part of the range.
    r = testHistoOneValue(filename, SampleDataType.int8, -42, False,
                          datarange = (-5, +760))

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats == (-5, +1)
    assert r.histo == (-5, +760)
    assert r.stats_count == 3*BRICK
    assert r.bins[0] == BRICK
    assert r.bins[2] == 2*BRICK

    # Similar to the above but no values written at all.
    # Defaultvalue is still 1 due to missing zero-centric propery
    # so this is what should be reflected in the statistics.

    r = testHistoOneValue(filename, SampleDataType.int8, np.nan, False,
                          datarange = (-5, +760))

    # Invariants for the integer case
    assert r.range[0] <= r.stats[0] and r.range[1] >= r.stats[1]
    assert r.histo == r.range
    assert r.histo_count == r.stats_count
    # Data dependent
    assert r.stats == (+1, +1)
    assert r.histo == (-5, +760)
    assert r.stats_count == 3*BRICK
    assert r.bins[2] == 3*BRICK

def testFancyDefaultValue():
    """
    Part of the test suite using the same test data stored in different ways.
    Check what happens when reading samples that were never written.

    The rectangles used are:
      a) Dead area of partly written brick
      b) Part dead area, part all-constant brick
      c) all-constant brick
      d) part all-constant brick, part unwritten brick
      e) unwritten brick.

    In the new reader all should return the default value.
    In the old reader the last one might throw a missing brick exception,
    it does in the C++ ZGY-Public API but the Python wrapper catches it.
    And the penultimate one might read zero from the unwritten area
    while still seeing the default (1 in this case) elsewhere.

    Also check reading completely outside range. The new accessor should
    raise exceptions; the old one does whatever it feels like doing.
    """
    with LocalFileAutoDelete("fancy-2.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.int8, (-2,+763),
                        newzgy.ZgyWriter)
        checkReadingDeadArea(fn.name, (5, 22, 1), oldzgy.ZgyReader, 1)
        checkReadingDeadArea(fn.name, (5, 22, 63), oldzgy.ZgyReader, 1)
        checkReadingDeadArea(fn.name, (5, 22, 65), oldzgy.ZgyReader, 1)
        checkReadingDeadArea(fn.name, (5, 22, 127), oldzgy.ZgyReader,
                             np.array([[[1, 0],[1, 0]],[[1, 0],[1, 0]]]))
        checkReadingDeadArea(fn.name, (5, 22, 129), oldzgy.ZgyReader, 0)
        #checkReadingOutsideRange(fn.name, oldzgy.ZgyReader)
        #checkReadingOutsideLod(fn.name, oldzgy.ZgyReader)
        #checkReadingToWrongValueType(fn.name, oldzgy.ZgyReader)

        checkReadingDeadArea(fn.name, (5, 22, 1), newzgy.ZgyReader, 1)
        checkReadingDeadArea(fn.name, (5, 22, 63), newzgy.ZgyReader, 1)
        checkReadingDeadArea(fn.name, (5, 22, 65), newzgy.ZgyReader, 1)
        checkReadingDeadArea(fn.name, (5, 22, 127), newzgy.ZgyReader, 1)
        checkReadingDeadArea(fn.name, (5, 22, 129), newzgy.ZgyReader, 1)
        checkReadingOutsideRange(fn.name, newzgy.ZgyReader)
        checkReadingOutsideLod(fn.name, newzgy.ZgyReader)
        checkReadingToWrongValueType(fn.name, newzgy.ZgyReader)

def testFancyReadConstant():
    """
    Test the new API in openzgy to return brick status.
    """
    with LocalFileAutoDelete("fancy-2.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.int8, (-2,+763),
                        newzgy.ZgyWriter)
        with newzgy.ZgyReader(fn.name, iocontext = SDCredentials()) as reader, io.StringIO() as bitbucket:
            verbose = lambda *args, **kwargs: print(*args, file=bitbucket, **kwargs)
            # While the data inside this small rectangle is indeed constant,
            # the whole brick is not. So, it won't be flagged as const val.
            a = reader.readconst((17,17,17), (2,2,2), as_float = True, verbose=verbose)
            b = reader.readconst((17,17,17), (2,2,2), as_float = False)
            assert(a is None)
            assert(b is None)

            # In this case the enclosing brick was explicitly written with
            # constant value 0, which will be read back as 1 because
            # the range is not zero centric.
            a = reader.readconst((1,2,67), (4,5,6), as_float = True)
            b = reader.readconst((1,2,67), (4,5,6), as_float = False)
            assert math.isclose(a, 1.0)
            assert math.isclose(b, -127)

            # Brick written as constant value 0 but only the region inside
            # the survey. Whether this registers as "constant" may be
            # considered an implementation detail. But ideally it ought to.
            a = reader.readconst((65,2,67), (4,5,6), as_float = True)
            b = reader.readconst((65,2,67), (4,5,6), as_float = False)
            assert math.isclose(a, 1.0)
            assert math.isclose(b, -127)

            # Two bricks never written, two with constant value 0.
            a = reader.readconst((0,0,64), (128,64,128), as_float = True)
            b = reader.readconst((0,0,64), (128,64,128), as_float = False)
            assert math.isclose(a, 1.0)
            assert math.isclose(b, -127)

def testFancyMisc():
    """
    Part of the test suite using the same test data stored in different ways.
    """
    with LocalFileAutoDelete("fancy-1.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.int8, (-28,+227),
                        newzgy.ZgyWriter)
        # Doesn't really belong here but doesn't bother to create a test file.
        runCloseOnException(fn.name, newzgy.ZgyReader)
        runErrorOnClose(fn.name, newzgy.ZgyReader)
        runConversions(fn.name, newzgy.ZgyReader)
        runErrorIfNotOpenForRead(fn.name, newzgy.ZgyReader)
        runDumpToDevNull(fn.name, newzgy.ZgyReader)

        if HasOldZgy():
            runCloseOnException(fn.name, oldzgy.ZgyReader)
            runConversions(fn.name, oldzgy.ZgyReader)
            runErrorIfNotOpenForRead(fn.name, oldzgy.ZgyReader)

        with LocalFileAutoDelete("fancy-1-clone.zgy") as cloned:
            runClone(cloned.name, fn.name)
            runUpdate(cloned.name)
            runDumpMembers(cloned.name, fn.name)

def testFancy1():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY writer, both OpenZGY and ZGY-Public reader, local file, int8.
    The coding range is asymmetric but zero centric.
    """
    with LocalFileAutoDelete("fancy-1.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.int8, (-28,+227),
                        newzgy.ZgyWriter)
        checkContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        # The next line reveals a bug in ZGY-Public.
        checkRawContents(fn.name, oldzgy.ZgyReader, 0, 100)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkStatistics(fn.name, oldzgy.ZgyReader, 0, 0, True)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, oldzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testFancy2():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY writer, both OpenZGY and ZGY-Public reader, local file, int8.
    Unlike #1 the coding range is not zero centric. So 0 cannot be represented.
    When can be stored is -2, +1, +4, ..., +763 i.e. only values 3*n+1.
    So my sample data values 31 and 301 are representable, but zero is not.
    """
    with LocalFileAutoDelete("fancy-2.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.int8, (-2,+763),
                        newzgy.ZgyWriter)
        checkContents(fn.name, oldzgy.ZgyReader, 1, 0)
        checkContents(fn.name, newzgy.ZgyReader, 1, 1)
        checkLodContents(fn.name, oldzgy.ZgyReader, 1, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 1, 1)
        # The next line reveals a bug in ZGY-Public.
        checkRawContents(fn.name, oldzgy.ZgyReader, 1, 382)
        checkRawContents(fn.name, newzgy.ZgyReader, 1, 1)
        checkStatistics(fn.name, oldzgy.ZgyReader, 1, 0, True)
        checkStatistics(fn.name, newzgy.ZgyReader, 1, 1, True)
        checkHistogram(fn.name, oldzgy.ZgyReader, 1, 0, True)
        checkHistogram(fn.name, newzgy.ZgyReader, 1, 0, True)

def testFancy3():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY writer, both OpenZGY and ZGY-Public reader, local file, int16.
    Unlike #1 and #2 zero is not included in the coding range.
    The closest representable value to zero is +20
    The valuetype is now int16 instead of int8 for variation.
    """
    with LocalFileAutoDelete("fancy-3.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.int16, (+20,+16403.75),
                        newzgy.ZgyWriter)
        checkContents(fn.name, oldzgy.ZgyReader, 20, 0)
        checkContents(fn.name, newzgy.ZgyReader, 20, 20)
        checkLodContents(fn.name, oldzgy.ZgyReader, 20, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 20, 20)
        checkRawContents(fn.name, oldzgy.ZgyReader, 20, 8212)
        checkRawContents(fn.name, newzgy.ZgyReader, 20, 20)
        checkStatistics(fn.name, oldzgy.ZgyReader, 20, 0, True)
        checkStatistics(fn.name, newzgy.ZgyReader, 20, 20, True)
        checkHistogram(fn.name, oldzgy.ZgyReader, 20, 0, True)
        checkHistogram(fn.name, newzgy.ZgyReader, 20, 20, True)

def testFancy4():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY writer, both OpenZGY and ZGY-Public reader, local file, float32.
    Bad coding range hint.
    The coding range for float cubes is just a hint that might be used as a
    hint for the histogram range. Or it might be completely ignored
    if the histogram is written during a separate pass where the exact
    range is already known.
    """
    with LocalFileAutoDelete("fancy-4.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.float, (-1,+1),
                        newzgy.ZgyWriter)
        checkContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkStatistics(fn.name, oldzgy.ZgyReader, 0, 0, True)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, oldzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testFancy5():
    """
    Part of the test suite using the same test data stored in different ways.
    Unline 1..4, this uses the old ZGY-Public writer, to help verify that
    the old and new code produces the same result. The test uses  both OpenZGY
    and ZGY-Public reader, local file, int8.
    """
    with LocalFileAutoDelete("fancy-5.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.int8, (-28,+227),
                        oldzgy.ZgyWriter)
        checkContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        # The next line reveals a bug in ZGY-Public.
        checkRawContents(fn.name, oldzgy.ZgyReader, 0, 100)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkStatistics(fn.name, oldzgy.ZgyReader, 0, 0, False)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, False)
        checkHistogram(fn.name, oldzgy.ZgyReader, 0, 0, False)
        checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, False)

def testFancy6():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY Python writer, both OpenZGY and ZGY-Public reader, local file, float.
    Compared to the old writer the user specified codingrange
    will now be ignored and the statistical range used instead.
    Note that if api.ZgyMeta.datarange chooses to enforce this
    then only the old reader will be able to verify what was written.
    """
    with LocalFileAutoDelete("fancy-6.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.float, (-1,+42),
                        newzgy.ZgyWriter)
        checkContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, oldzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkStatistics(fn.name, oldzgy.ZgyReader, 0, 0, True)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, oldzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testFancy7():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY Python writer, int8 with lossless compression.
    Currently this is explicitly forbidden by a test in the api.
    See comments in the doc and in the ZgyWriter source code for why. Also,
    fewer checks because the old reader cannot handle the new compression.
    """
    lossless = ZgyCompressFactory("ZFP", snr = 99)
    with LocalFileAutoDelete("fancy-7.zgy") as fn:
        with MustThrow("need to be stored as float", newzgy.ZgyUserError):
            createFancyFile(fn.name, SampleDataType.int8, (-28,+227),
                            newzgy.ZgyWriter, single_write=True,
                            kwargs={"compressor": lossless})
        #checkContents(fn.name, newzgy.ZgyReader, 0, 0)
        #checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        #checkRawContents(fn.name, newzgy.ZgyReader, 0, 0)
        #checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True)
        #checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)
        fn.disarm()

def testFancy8():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY Python writer, float32 with lossy compression.
    """
    lossless = ZgyCompressFactory("ZFP", snr = 99)
    with LocalFileAutoDelete("fancy-8.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.float, (-1,+42),
                        newzgy.ZgyWriter, single_write=True,
                        kwargs={"compressor": lossless})
        checkContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testFancy9():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY Python writer, int8 with lossy compression.
    Currently this is explicitly forbidden by a test in the api.
    See comments in the doc and in the ZgyWriter source code for why. Also,
    fewer checks because the old reader cannot handle the new compression.
    """
    lossy = ZgyCompressFactory("ZFP", snr = 30)
    with LocalFileAutoDelete("fancy-9.zgy") as fn:
        with MustThrow("need to be stored as float", newzgy.ZgyUserError):
            createFancyFile(fn.name, SampleDataType.int8, (-28,+227),
                            newzgy.ZgyWriter, single_write=True,
                            kwargs={"compressor": lossy})
        #checkContents(fn.name, newzgy.ZgyReader, 0, 0, maxdelta=1.5)
        #checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        #checkRawContents(fn.name, newzgy.ZgyReader, 0, 0, maxdelta=2.5)
        #checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True, maxdelta=8000)
        #checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)
        fn.disarm()

def testFancy10():
    """
    Part of the test suite using the same test data stored in different ways.
    OpenZGY Python writer, float32 with lossy compression.
    """
    lossy = ZgyCompressFactory("ZFP", snr = 30)
    with LocalFileAutoDelete("fancy-10.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.float, (-1,+42),
                        newzgy.ZgyWriter, single_write=True,
                        kwargs={"compressor": lossy})
        checkContents(fn.name, newzgy.ZgyReader, 0, 0, maxdelta=2.0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0, maxdelta=2.0)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True, maxdelta=5000)
        #checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testFancy11():
    """
    Part of the test suite using the same test data stored in different ways.
    New code only, small bricksize, no compression.
    """
    with LocalFileAutoDelete("fancy-11.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.float, (-28,+227),
                        newzgy.ZgyWriter,
                        kwargs={"bricksize": (32,32,32)})
        checkContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testFancy12():
    """
    Part of the test suite using the same test data stored in different ways.
    New code only, large bricksize, no compression.
    """
    with LocalFileAutoDelete("fancy-12.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.float, (-28,+227),
                        newzgy.ZgyWriter,
                        kwargs={"bricksize": (128,128,128)})
        checkContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True)
        checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testFancy13():
    """
    Part of the test suite using the same test data stored in different ways.
    New code only, non-rectangular bricks, no compression.
    Need single_write=True because with the very small
    bricksize my test code ends up writing nore than
    one brick past the end of the survey.
    """
    with LocalFileAutoDelete("fancy-13.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.float, (-28,+227),
                        newzgy.ZgyWriter, single_write=True,
                        kwargs={"bricksize": (16,32,128)})
        checkContents(fn.name, newzgy.ZgyReader, 0, 0, maxdelta=2.0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0, maxdelta=2.0)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True, maxdelta=5000)
        checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testFancy14():
    """
    Part of the test suite using the same test data stored in different ways.
    New code only, non-rectangular bricks, with compression.
    """
    lossy = ZgyCompressFactory("ZFP", snr = 30)
    with LocalFileAutoDelete("fancy-14.zgy") as fn:
        createFancyFile(fn.name, SampleDataType.float, (-28,+227),
                        newzgy.ZgyWriter, single_write=True,
                        kwargs={"bricksize": (16,32,128), "compressor": lossy})
        checkContents(fn.name, newzgy.ZgyReader, 0, 0, maxdelta=2.0)
        checkLodContents(fn.name, newzgy.ZgyReader, 0, 0)
        checkRawContents(fn.name, newzgy.ZgyReader, 0, 0, maxdelta=2.0)
        checkStatistics(fn.name, newzgy.ZgyReader, 0, 0, True, maxdelta=5000)
        #FAILS checkHistogram(fn.name, newzgy.ZgyReader, 0, 0, True)

def testCloudAutoDelete():
    with CloudFileAutoDelete("xyzzy", None) as fn:
        assert fn.name[:5] == "sd://"
        fn.disarm()

    # Seismic drive, missing credentials.
    with MustThrow("service URL has not been defined", RuntimeError):
        with CloudFileAutoDelete("xyzzy", None, silent=True) as fn:
            assert fn.name[:5] == "sd://"

    # Seismic drive, file not found.
    # As of 2021-02-12 it is no longer an error to delete a non-existing file.
    #with MustThrow("does not exist", RuntimeError):
    with CloudFileAutoDelete("xyzzy", SDCredentials(), silent=True) as fn:
        assert fn.name[:5] == "sd://"

def testReadFromCloud(filename):
    with newzgy.ZgyReader(filename, iocontext=SDCredentials()) as reader, io.StringIO() as bitbucket:
        verbose = lambda *args, **kwargs: print(*args, file=bitbucket, **kwargs)
        assert reader.size == (181, 241, 169)
        tmp = np.zeros((100, 50, 30), dtype=np.int8)
        reader.read((42, 70, 50), tmp, verbose=verbose)
        #print(tuple(tmp[0,0,:5]), tuple(tmp[0,0,-5:]))
        assert tuple(tmp[0,0,:5]) == (57, 48, 38, 28, 17)
        assert tuple(tmp[0,0,-5:]) == (-101, -91, -79, -65, -51)

def testCloudWriter(filename):
    """
    File written by the new code to seismic store
    I haven't hooked up the old API to seismic store, so do the read
    checks only with newzgy.
    """
    with TimeMe("  createFancyFile"):
        createFancyFile(filename, SampleDataType.int8, (-28,+227), newzgy.ZgyWriter)
    with TimeMe("  checkContents"):
        checkContents(filename, newzgy.ZgyReader, 0, 0)
    with TimeMe("  checkLodContents"):
        checkLodContents(filename, newzgy.ZgyReader, 0, 0)
    with TimeMe("  checkRawContents"):
        checkRawContents(filename, newzgy.ZgyReader, 0, 0)
    with TimeMe("  checkStatistics"):
        checkStatistics(filename, newzgy.ZgyReader, 0, 0, True)
    with TimeMe("  checkHistogram"):
        checkHistogram(filename, newzgy.ZgyReader, 0, 0, True)
    with TimeMe("  delete #1"):
        newzgy.ZgyUtils(SDCredentials()).delete(filename)
    with TimeMe("  delete #2"):
        newzgy.ZgyUtils(SDCredentials()).delete(filename)

def testLegalTag(filename):
    meta = {"foo": "bar", "foocount": 42}
    meta = {"kind": "slb:openzgy:test:1.0.0", "data": meta}
    iocontext = SDCredentials(legaltag="slb-synthetic-seismic",
                              writeid="test-my-write", seismicmeta=meta)
    with newzgy.ZgyWriter(filename,
                          iocontext = iocontext,
                          size = (64, 64, 64),
                          datatype = SampleDataType.float) as writer:
        data = np.zeros((64, 64, 64), dtype=np.float32)
        writer.write((0, 0, 0), data)
        writer.finalize()
        #os.system("sdutil stat " + SDTestSink("legaltag.zgy") + " --detailed")
    # TODO-Test, read back metadata and confirm it was stored correctly.
    # Not possible yet.
    # TODO-Question, there is both a {get,set}MetaData and a {get,set}SeismicMeta().
    # I suspect the former only sets the "data" portion of SeismicMeta
    # but the two might also be completely unrelated.
    # TODO-Question, when (and only when) I specify seismicmeta I see that
    # sdutil stat --detailed will show me the seismicmeta and this
    # includes the legaltag. Is the legaltag in the seismicmeta
    # different from the "old" legaltag? Can it be changed, since we
    # do have a setSeismicMeta?

def testCloudConsolidateBricks(filename, *, verbose = False):
    """
    When reading from seismic store, bricks that are contiguous in memory
    should be read in a single operation because larger brick size is
    faster (up to a point). When not contiguous the reads should still
    make just a single call to seismic store with a scatter/gather array
    so the lower level code miggt do multi-threading.

    This test also enables the single-block caching which will cause
    all the headers to be read in a single operation. It can also speed
    up regular brick access. Note that this cache is extremely simplistic,
    it only remembers the previous result and it only returns a match
    if the next request is exactly identical.

    TODO-Low consider splitting this into multiple tests.
    """
    vprint = ((lambda *args, **kwargs: print(*args, **kwargs)) if verbose
              else (lambda *args, **kwargs: None))
    trace = TraceCallsToSD(verbose = verbose)
    iocontext = SDCredentials(aligned=1, maxsize=64, maxhole=1, threads=1,
                              _debug_trace = trace
    )
    bricksize = np.array((64, 64, 64), dtype=np.int64)
    brick = np.product(bricksize) * np.dtype(np.float32).itemsize
    size = np.array((181, 241, 169), dtype=np.int64)
    numbricks = (size + bricksize - 1) // bricksize
    vprint("Creating. Expect header written twice, then bulk data once.")
    with newzgy.ZgyWriter(filename, iocontext=iocontext,
                          bricksize = tuple(bricksize),
                          size = tuple(size)) as writer:
        data = np.arange(np.product(size), dtype=np.float32).reshape(size)
        writer.write((0,0,0), data)

    # lod 0 bricks: 3 * 4 * 3 = 36
    # lod 1 bricks: 2 * 2 * 2 = 8
    # lod 2 bricks: 1
    # sum bricks on file: 45

    # Writing the final header is the penultimate and not the last write.
    # This is due to how SeismicStoreFileDelayedWrite works. See also
    # comments in ZgyWriter.close().

    assert len(trace.calls) == 3
    assert trace.calls[0] == ("append", brick, brick, 1)
    assert trace.calls[1] == ("write", brick, brick, 1)
    assert trace.calls[2] == ("append", 45 * brick, 45*brick, 1)
    trace.reset()

    vprint("Opening. Expect all headers read in just one real access.")
    with newzgy.ZgyReader(filename, iocontext = iocontext) as reader:

        assert len(trace.calls) >= 1
        assert trace.calls[0].what in ("read", "readv", "cachemiss")
        assert all([t.what == "cachehit" for t in trace.calls[1:]])
        trace.reset()

        # The size in bricks, il/xl/slice, is (3, 4, 3).
        # Reading a single inline should require just a single access.
        # Reading a single crossline should read one brick-column (3 bricks)
        # at a time, so it will need 3 reads. Each brick is 256 KB.
        ildata = np.zeros((1, size[1], size[2]), dtype=np.float32)
        xldata = np.zeros((size[0], 1, size[2]), dtype=np.float32)

        vprint("read one il,", numbricks[1] * numbricks[2], "bricks")
        reader.read((0,0,0), ildata)
        assert len(trace.calls) == 1
        assert trace.calls[0] == ("readv",
                                  brick*numbricks[1]*numbricks[2],
                                  brick*numbricks[1]*numbricks[2], 1)
        trace.reset()

        vprint("read one xl,", numbricks[0], "*",  numbricks[2], "bricks")
        reader.read((0,0,0), xldata)
        # Not contiguous, but a single scatter/gather read.
        assert len(trace.calls) == 1
        assert trace.calls[0] == ("readv",
                                  brick*numbricks[0]*numbricks[2],
                                  brick*numbricks[0]*numbricks[2], 3)
        trace.reset()

        sample = np.zeros((1,1,1), dtype=np.float32)
        vprint("read one sample. Should require just one brick.")
        reader.read((100,100,100), sample)
        assert len(trace.calls) == 1
        assert trace.calls[0].nbytes == brick
        trace.reset()

        vprint("read another sample in the same brick. Should be cached.")
        reader.read((101,102,103), sample)
        assert len(trace.calls) == 1
        assert trace.calls[0] == ("cachehit", brick, brick, 1)
        trace.reset()

    vprint("Opening with 64 MB buffers. Everything ought to be cached.")
    # Note that the entire file is smaller than the requested blocking,
    # it is important to veryfy that this doesn't cause problems when
    # hitting EOF. The "simple cache" and the "scatter/gather" cases
    # need to be tested separately.
    iocontext = SDCredentials(aligned=64, maxsize=64, maxhole=1, threads=1,
                              _debug_trace = trace
    )
    with newzgy.ZgyReader(filename, iocontext = iocontext) as reader:

        # As with the previous case there should just be a single read.
        assert len(trace.calls) >= 1
        assert trace.calls[0].what in ("read", "readv", "cachemiss")
        assert all([t.what == "cachehit" for t in trace.calls[1:]])
        trace.reset()

        # This will currently not be very performant. The requested
        # padding will be applied but the simplistic cache won't use it.
        # Not that big a deal since the padding in real cases should
        # probably be just 4 MB or so, Small enough for the wasted
        # bytes not actually costing anything.
        # The test is important though. The padding to align reads
        # is still applied, but in a different place in the code.
        vprint("read one il,", numbricks[1] * numbricks[2], "bricks")
        ildata = np.zeros((1, size[1], size[2]), dtype=np.float32)
        reader.read((0,0,0), ildata)
        assert len(trace.calls) == 1
        # See FileAdt._consolidate_requests._groupsize()
        # The header segment is not aligned to out oversized "align"
        # parameter. This causes some needless data access because
        # the padding will cross a segment boundary. Segment 0 (headers)
        # will be read again even though we don't need it.
        # The asserts below reflect the current implementation.
        #assert trace.calls[0] == ("readv", 12*brick, 45*brick, 2)
        assert trace.calls[0] == ("readv", 12*brick, 46*brick, 2)
        trace.reset()

        vprint("read one xl,", numbricks[0], "*",  numbricks[2], "bricks")
        xldata = np.zeros((size[0], 1, size[2]), dtype=np.float32)
        reader.read((0,0,0), xldata)
        # Consolidate and split causes this to end up as 3 separate
        # non contiguous reads. Applying "align" is done too late
        # which causes each of these 3 reads to cover the exact same
        # area. And those areas in turn consist of two reads since
        # we are reading the header also. The naive cache doesn't
        # help us here. Fortunately this is a very contrived case.
        assert len(trace.calls) == 1
        #assert trace.calls[0] == ("readv", 9*brick, 45*brick, 1)
        assert trace.calls[0] == ("readv", 9*brick, 3*46*brick, 6)
        trace.reset()

        # This should trigger the naive cache, tailored specifically
        # to how Petrel reads data from ZGY.
        vprint("read one il, one brick at a time")
        ildata = np.zeros((1, 64, 64), dtype=np.float32)
        for xl in range(0, size[1], 64):
            for zz in range(0, size[2], 64):
                reader.read((0, xl, zz), ildata)
        assert len(trace.calls) >= 1
        # The cache was cleared after readv, so expect one and just one
        # read request to fill it.
        assert trace.calls[0].what in ("read", "readv", "cachemiss")
        assert all([t.what == "cachehit" for t in trace.calls[1:]])
        trace.reset()

        vprint("read one xl, one brick at a time")
        xldata = np.zeros((64, 1, 64), dtype=np.float32)
        for il in range(0, size[0], 64):
            for zz in range(0, size[2], 64):
                reader.read((il, 0, zz), ildata)
        assert len(trace.calls) >= 1
        assert all([t.what == "cachehit" for t in trace.calls[0:]])
        trace.reset()

    # Re-create the file with 7 MB segment size, to stress some more code.
    iocontext = SDCredentials(aligned=1, maxsize=64, maxhole=1, threads=1,
                              segsize=7, _debug_trace = trace
    )
    bricksize = np.array((64, 64, 64), dtype=np.int64)
    brick = np.product(bricksize) * np.dtype(np.float32).itemsize
    size = np.array((181, 241, 169), dtype=np.int64)
    numbricks = (size + bricksize - 1) // bricksize
    vprint("Creating. Expect header written twice and bulk data in 7 parts.")
    with newzgy.ZgyWriter(filename, iocontext=iocontext,
                          bricksize = tuple(bricksize),
                          size = tuple(size)) as writer:
        data = np.arange(np.product(size), dtype=np.float32).reshape(size)
        writer.write((0,0,0), data)

    # There may be several reads needed to generate lod 1 bricks
    # from data already flushed. Ignore those.
    calls = list([ e for e in trace.calls
                   if e.what not in ("readv", "cachehit", "cachemiss")])
    assert len(calls) == 9
    assert calls[0] == ("append", brick, brick, 1) # empty header
    assert calls[1] == ("append", 7 * brick, 7 * brick, 1)
    assert calls[2] == ("append", 7 * brick, 7 * brick, 1)
    assert calls[3] == ("append", 7 * brick, 7 * brick, 1)
    assert calls[4] == ("append", 7 * brick, 7 * brick, 1)
    assert calls[5] == ("append", 7 * brick, 7 * brick, 1)
    assert calls[6] == ("append", 7 * brick, 7 * brick, 1)
    assert calls[7] == ("write", brick, brick, 1) # actual header
    assert calls[8] == ("append", 3 * brick, 3 * brick, 1) # mop up.
    trace.reset()

    iocontext = SDCredentials(aligned=1, maxsize=64, maxhole=1, threads=1,
                              _debug_trace = trace
    )
    with newzgy.ZgyReader(filename, iocontext = iocontext) as reader:
        assert len(trace.calls) >= 1
        assert trace.calls[0].what in ("read", "readv", "cachemiss")
        assert all([t.what == "cachehit" for t in trace.calls[1:]])
        trace.reset()

        vprint("read one il,", numbricks[1] * numbricks[2], "bricks")
        ildata = np.zeros((1, size[1], size[2]), dtype=np.float32)
        reader.read((0,0,0), ildata)
        # There will be two reads since it crissed a segment boundary.
        assert len(trace.calls) == 1
        assert trace.calls[0] == ("readv", 12*brick, 12*brick, 2)
        trace.reset()

        vprint("read one xl,", numbricks[0], "*",  numbricks[2], "bricks")
        xldata = np.zeros((size[0], 1, size[2]), dtype=np.float32)
        reader.read((0,0,0), xldata)
        # Not contiguous, but a single scatter/gather read.
        # More that 3 parts due to crossing segment boundaries.
        assert len(trace.calls) == 1
        assert trace.calls[0] == ("readv", 9*brick, 9*brick, 4)
        trace.reset()

    vprint("done.")

def Main():
    np.seterr(all='raise')

    with TimeMe("ProgressWithDots"):
        testProgressWithDots()

    with TimeMe("BadArgumentsOnCreate"):
        testBadArgumentsOnCreate()

    with TimeMe("BadArgumentsOnReadWrite"):
        with LocalFileAutoDelete("somefile.zgy") as fn:
            testBadArgumentsOnReadWrite(fn.name)

    with TimeMe("AutoDelete"):
        testAutoDelete()

    if HasOldZgy():
        with TimeMe("HistogramRangeIsCenterNotEdge"):
            with LocalFileAutoDelete("histo.zgy") as fn:
                testHistogramRangeIsCenterNotEdge(fn.name)

    with TimeMe("EmptyFile_NN"):
        with LocalFileAutoDelete("emptyfile.zgy") as fn:
            testEmptyFile(fn.name, newzgy.ZgyWriter, newzgy.ZgyReader)

    if HasOldZgy():
        with TimeMe("EmptyFile_ON"):
            with LocalFileAutoDelete("emptyfile.zgy") as fn:
                testEmptyFile(fn.name, oldzgy.ZgyWriter, newzgy.ZgyReader)

        with TimeMe("EmptyFile_NO"):
            with LocalFileAutoDelete("emptyfile.zgy") as fn:
                testEmptyFile(fn.name, newzgy.ZgyWriter, oldzgy.ZgyReader)

        with TimeMe("EmptyFile_OO"):
            with LocalFileAutoDelete("emptyfile.zgy") as fn:
                testEmptyFile(fn.name, oldzgy.ZgyWriter, oldzgy.ZgyReader)

    with LocalFileAutoDelete("rmwfile.zgy") as fn:
        testRmwFile(fn.name, newzgy.ZgyWriter)

    with LocalFileAutoDelete("fatal-error.zgy") as fn:
        testFatalErrorFlag(fn.name)

    if False: # Disabled because it takes too long.
        with TimeMe("LargeSparseFile"):
            with LocalFileAutoDelete("largesparse.zgy") as fn:
                testLargeSparseFile(fn.name, newzgy.ZgyWriter, newzgy.ZgyReader)

    with TimeMe("Naan"):
        with LocalFileAutoDelete("naan.zgy") as fn:
            testNaan(fn.name)

    with TimeMe("WriteNaanToIntegerStorage"):
        with LocalFileAutoDelete("intnaan.zgy") as fn:
            testWriteNaanToIntegerStorage(fn.name)

    with TimeMe("ZeroCentric"):
        with LocalFileAutoDelete("zerocentric.zgy") as fn:
            testZeroCentric(fn.name)

    with TimeMe("FinalizeProgress"):
        with LocalFileAutoDelete("finalize.zgy") as fn:
            testFinalizeProgress(fn.name, abort = False)

    with TimeMe("FinalizeProgress"):
        with LocalFileAutoDelete("finalize.zgy") as fn:
            testFinalizeProgress(fn.name, abort = True)

    with TimeMe("HugeFile"):
        with LocalFileAutoDelete("huge.zgy") as fn:
            testHugeFile(fn.name)

    with LocalFileAutoDelete("oddsize.zgy") as fn:
        testDecimateOddSize(fn.name)

    with TimeMe("DecimateWeightedAverage"):
        with LocalFileAutoDelete("weighted.zgy") as fn:
            testDecimateWeightedAverage(fn.name)

    with TimeMe("MixingUserAndStorage"):
        with LocalFileAutoDelete("mixuserstorage.zgy") as fn:
            testMixingUserAndStorage(fn.name)

    with TimeMe("SmallConstArea"):
        with LocalFileAutoDelete("smallconstarea.zgy") as fn:
            testSmallConstArea(fn.name)

    with LocalFileAutoDelete("testhisto_f.zgy") as fn:
        testHistoCornercaseFloat(fn.name)

    with LocalFileAutoDelete("testhisto_i.zgy") as fn:
        testHistoCornercaseInt(fn.name)

    with TimeMe("FancyDefaultValue"):
        testFancyDefaultValue()
    with TimeMe("FancyReadConstant"):
        testFancyReadConstant()
    with TimeMe("FancyMisc"):
        testFancyMisc()
    with TimeMe("TestFancy1"):
        testFancy1()
    with TimeMe("TestFancy2"):
        testFancy2()
    with TimeMe("TestFancy3"):
        testFancy3()
    with TimeMe("TestFancy4"):
        testFancy4()
    if HasOldZgy():
        with TimeMe("TestFancy5"):
            testFancy5()
    with TimeMe("TestFancy6"):
        testFancy6()
    with TimeMe("TestFancy11"):
        testFancy11()
    with TimeMe("TestFancy12"):
        testFancy12()
    with TimeMe("TestFancy13"):
        testFancy13()

    # ZFP COMPRESSION

    if HasZFPCompression():
        with TimeMe("RegisteredCompressors"):
            testRegisteredCompressors()
        with TimeMe("TestFancy7"):
            testFancy7()
        with TimeMe("TestFancy8"):
            testFancy8()
        with TimeMe("TestFancy9"):
            testFancy9()
        with TimeMe("TestFancy10"):
            testFancy10()
        with TimeMe("TestFancy14"):
            testFancy14()
        with TimeMe("NoRmwInCompressedFile"):
            with LocalFileAutoDelete("no-rmw.zgy") as fn:
                testNoRmwInCompressedFile(fn.name)
        with TimeMe("Naan"):
            with LocalFileAutoDelete("naan.zgy") as fn:
                testNaan(fn.name, 70)

    # SEISMIC STORE

    if not HasSeismicStore():
        print("SKIPPING seismic store tests")
        return

    with TimeMe("testCloudAutoDelete"):
        testCloudAutoDelete()

    with TimeMe("testReadFromCloud"):
        testReadFromCloud(SDTestData("Synt2.zgy"))

    with TimeMe("testCloudWriter"):
        with CloudFileAutoDelete("openzgy-rules.zgy", SDCredentials()) as cad:
            testCloudWriter(cad.name)
            cad.disarm() # The test function cleans up itself, unless it throws.

    with TimeMe("EmptyFile"):
        with CloudFileAutoDelete("emptyfile.zgy", SDCredentials()) as fn:
            testEmptyFile(fn.name)

    # oldzgy probably doesn't have zgycloud set up in this test.
    if HasOldZgy() and False:
        with TimeMe("EmptyFile_ON"):
            with CloudFileAutoDelete("emptyfile.zgy", SDCredentials()) as fn:
                testEmptyFile(fn.name, oldzgy.ZgyWriter, newzgy.ZgyReader)

        with TimeMe("EmptyFile_NO"):
            with CloudFileAutoDelete("emptyfile.zgy", SDCredentials()) as fn:
                testEmptyFile(fn.name, newzgy.ZgyWriter, oldzgy.ZgyReader)

        with TimeMe("EmptyFile_OO"):
            with CloudFileAutoDelete("emptyfile.zgy", SDCredentials()) as fn:
                testEmptyFile(fn.name, oldzgy.ZgyWriter, oldzgy.ZgyReader)

    with TimeMe("EmptyExistingFile"):
        testEmptyExistingFile("sd://sntc/testdata/OldEmpty.zgy")

    with TimeMe("testRmwFile"):
        with CloudFileAutoDelete("rmwfile.zgy", SDCredentials()) as fn:
            testRmwFile(fn.name, newzgy.ZgyWriter)

    with TimeMe("testLegalTag"):
        with CloudFileAutoDelete("legaltag.zgy", SDCredentials()) as fn:
            testLegalTag(fn.name)

    with CloudFileAutoDelete("consolidate.zgy", SDCredentials()) as fn:
        with TimeMe("ConsolidateBricks"):
            testCloudConsolidateBricks(fn.name, verbose = False)

if __name__ == "__main__":
    Main()

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
