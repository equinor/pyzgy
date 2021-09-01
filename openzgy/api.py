#!/usr/bin/env python3

"""
This file contains the public OpenZGY API.

The API is modeled roughly after the API exposed by the Python wrapper
around the existing C++ ZGY-Public API. This is probably just as good
a starting point than anything else. And it makes testing simpler for
those tests that compare the old and new behavior.

    api.ZgyMeta:

        * Has a private reference to a impl.meta.ZgyInternalMeta instance.
        * Contains a large number of properties exposing meta data,
          most of which will present information from the ZgyInternalMeta
          in a way that is simpler to use and doesn't depend on the
          file version.
        * End users will access methods from this class. Most likely
          via the derived ZgyReader and ZgyWriter classes. The end users
          might not care that there is a separate base class.

    api.ZgyMetaAndTools(ZgyMeta):

        * Add coordinate conversion routines to a ZgyMeta instance.

    api.ZgyReader(ZgyMetaAndTools): # + read, readconst, close
    api.ZgyWriter(ZgyMetaAndTools): # + write, writeconst, finalize, close

        * Has a private reference to a impl.meta.ZgyInternalBulk instance.
        * Add bulk read and write functionality, forwarding the requests
          to the internal bulk instance.
        * These classes with their own and inherited methods and properties
          comprise the public OpenZGY API.
        * Currently the ZgyWriter does not expose the bulk read() function
          but it does allow accessing all the metadata. Allowing read()
          might be added later if it appears to be useful.
          In practice this just means to let ZgyWriter inherit ZgyReader.
"""

##@package openzgy
#@brief The top level only has package members.
##@package openzgy.api
#@brief User visible apis are here.

##
# \mainpage
#
# The %OpenZGY Python API allows read/write access to files
# stored in the ZGY format. The main part of the API is here:
#
# \li ZgyReader and its ZgyMeta base class.
# \li ZgyWriter also extending ZgyMeta.
# \li ZgyUtils for anything not read or write.
# \li \ref exception.ZgyError
#  you might want to catch.
# \li ProgressWithDots example of progress reporting.
# \li \ref Example Example application.
#
# If you are reading this document from doxygen/pure/apidoc.pdf
# in the source tree then please see doxygen/README.md for an
# explanation of why the documentation produced by the build might
# be better.
#
# \if IMPL
# If you are viewing the full Doxygen documentation then this
# covers both the API and most of the implementation. So if you
# look at the list of classes and methods this might seem a bit
# daunting. All you really need to use the API should be in the
# above list. Excluding trivial structs that will be cross
# referenced as needed. So you don't need to go looking for them.
# Of course, if you want to work om %OpenZGY itself then you
# probably need everything.
#
# See also the following related pages:
#
# \li \ref physicalformat
# \li \ref implementation
# \li \ref lowres
# \li \ref migration
#
# \endif
# \page Example
# \include simplecopy.py

import numpy as np
import json
import sys
from collections import namedtuple
from enum import Enum

from .exception import *
from .impl import enum as impl_enum
from .impl.meta import ZgyInternalMeta
from .impl.bulk import ZgyInternalBulk, ScalarBuffer
from .impl.transform import generalTransform
from .impl.file import FileFactory
from .impl.stats import StatisticData
from .impl.lodalgo import DecimationType
from .impl.compress import CompressFactoryImpl
from .impl.zfp_compress import ZfpCompressPlugin
from .impl.genlod import GenLodC

##@cond IMPL

##@brief Explicit control of imported symbols.
class _internal:
    """
    This class is only used to rename some internal code.

    I want to be explicit about which classes I need from impl.*
    but at the same time I don't want to pollute the user-visible api
    namespace with names not even starting with an underscore that the
    user has no business accessing. I am not sure whether my way of
    achieving this is considered a kludge.

    What I really want is to say something like:

        from .impl.genlod import GenLodC as _internal.GenLodC

    but apparently this is not allowed.
    """
    pass
_internal.enum             = impl_enum;        del impl_enum
_internal.ZgyInternalMeta  = ZgyInternalMeta;  del ZgyInternalMeta
_internal.ZgyInternalBulk  = ZgyInternalBulk;  del ZgyInternalBulk
_internal.ScalarBuffer     = ScalarBuffer;     del ScalarBuffer
_internal.generalTransform = generalTransform; del generalTransform
_internal.FileFactory      = FileFactory;      del FileFactory
_internal.StatisticData    = StatisticData;    del StatisticData
_internal.DecimationType   = DecimationType;   del DecimationType
_internal.CompressFactoryImpl = CompressFactoryImpl; del CompressFactoryImpl
_internal.ZfpCompressPlugin   = ZfpCompressPlugin;   del ZfpCompressPlugin
_internal.GenLodC          = GenLodC;          del GenLodC

##@endcond

##@brief Sample data type used in the public API.
class SampleDataType(Enum):
    """
    Sample data type used in the public API.
    Corresponds to RawDataType used in the ZGY file format.
    """
    unknown = 1000
    int8    = 1001
    int16   = 1002
    float   = 1003

##@brief Horizontal or vertical dimension as used in the public API.
class UnitDimension(Enum):
    """
    Horizontal or vertical dimension as used in the public API.

    Horizontal dimension may be length or arc angle, although most
    applications only support length. Vertical dimension may be time
    or length. Vertical length is of course the same as depth.
    Arguably there should have been separate enums for horizontal and
    vertical dimension since the allowed values differ.
    """
    unknown  = 2000
    time     = 2001
    length   = 2002
    arcangle = 2003

##@brief Base class shared betewwn ZgyReader and ZgyWriter.
class ZgyMeta:
    """
    Base class shared betewwn ZgyReader and ZgyWriter.
    """
    def __init__(self, meta):
        """Create an instance, providing the ZgyInternalMeta to use."""
        assert meta is not None
        assert isinstance(meta, _internal.ZgyInternalMeta)
        self._meta = meta

    @property
    def size(self): # (iii)
        """
        Number of inlines, crosslines, and samples in that order.
        """
        return self._meta._ih._size

    @property
    def datatype(self): # s -> Enum
        """
        Sample data type.
        The ZGY-Public API uses enums: "int8", "int16", "float".
        In some cases these are also passed as strings.
        The old Python wrapper for ZGY-Public is stringly typed.
        Instead of returning a real enum it returns the name.
        """
        return _map_DataTypeToSampleDataType(self._meta._ih._datatype)

    @property
    def datarange(self): # (ff), a.k.a. DataMinMax
        """
        For integral data this is the lowest and highest sample value
        than can be represented in storage. The lowest storage value
        (e.g. -128 for SignedInt8 data) will be returned as DataMinMax[0]
        when read as float. Similarly the highest storage value e.g. +127
        will be returned as DataMinMax[1]. When integer data is read as
        the "native" integral type then no automatic scaling is applied.
        Note that in this case the actual range of the samples on file might
        be smaller (for int8, not all of the range -128..+127 might be used)
        but it cannot be larger.

        For floating point data these numbers are supposed to be the actual
        value range of the samples on file. It is a good idea to enforce
        this here, as the values stored by older writers cannot be trusted.
        Note: Also enforced on write in impl.meta.InfoHeaderV2.calculate_write.
        TODO-Worry: In some float32 datasets the bulk data might have
        ridiculously large spikes wich will be included in the statistical
        range but not in the codingrange. So, codingrange is actually the
        one that is correct. Also, can we have a situation where stats
        are not filled in while the codingrange is set? I am not sure
        this is currently handled.
        """
        if self._meta._ih._datatype == _internal.enum.RawDataType.Float32:
            return (self._meta._ih._smin, self._meta._ih._smax)
        else:
            return (self._meta._ih._safe_codingrange[0], self._meta._ih._safe_codingrange[1])

    @property
    def raw_datarange(self): # (ff), a.k.a. DataMinMax
        """
        As datarange, but the actual values read from the file before
        they might have been changed to try to fix a bad file.
        Only use this property if you want to handle such files
        differently than the library does.
        """
        if self._meta._ih._datatype == _internal.enum.RawDataType.Float32:
            return (self._meta._ih._smin, self._meta._ih._smax)
        else:
            return (self._meta._ih._file_codingrange[0], self._meta._ih._file_codingrange[1])

    @property
    def zunitdim(self):
        """
        Dimension in the vertical direction. "time" or "length".
        "time" might on file be "SeismicTWT" or "SeismicOWT".
        The old Python wrapper for ZGY-Public is stringly typed.
        Instead of returning a real enum it returns the name.
        """
        return _map_VerticalDimensionToUnitDimension(self._meta._ih._vdim)

    @property
    def hunitdim(self):
        """
        Dimension in the horizontal direction. Should always be "length".
        The original specification called for supporting "arcangle" as well,
        i.e. coordinates in degrees instead of a projection. But most
        application code that use ZGY will not support this.
        The old Python wrapper for ZGY-Public is stringly typed.
        Instead of returning a real enum it returns the name.
        """
        return _map_HorizontalDimensionToUnitDimension(self._meta._ih._hdim)

    @property
    def zunitname(self):
        """
        Unit in the horizontal direction. E.g. "ms", "m", or "ft".
        Note that Petrel might ignore this settings and instead
        prompt the user to state what the unit should be.
        """
        return self._meta._ih._vunitname

    @property
    def hunitname(self):
        """
        Unit in the horizontal direction. E.g. "m" or "ft".
        """
        return self._meta._ih._hunitname

    @property
    def zunitfactor(self):
        """
        Factor to multiply stored vertical values with to get SI units.
        E.g. 0.001 for ms, 1.0 for m or 0.3048 for ft.
        """
        return self._meta._ih._vunitfactor

    @property
    def hunitfactor(self):
        """
        Factor to multiply stored horizontal values with to get SI units.
        """
        return self._meta._ih._hunitfactor

    @property
    def zstart(self):
        """
        Distance from surface/MSL to first sample, given in the vertical unit.
        """
        return self._meta._ih._orig[2]

    @property
    def zinc(self):
        """
        Sample interval, given in the vertical unit.
        """
        return self._meta._ih._inc[2]

    @property
    def annotstart(self):
        """
        First inline and crossline numbers.
        """
        return self._meta._ih._orig[0:2]

    @property
    def annotinc(self):
        """
        Inline and crossline number increments between adjacent
        sections of the cube.
        """
        return self._meta._ih._inc[0:2]

    @property
    def corners(self):
        """
        World XY coordinates of each of the 4 corners.
        The same coordinates in ordinal numbers are
        ((0, 0), (Size[0]-1, 0), (0, Size[1]-1), (Size[0]-1, Size[0]-1))
        """
        return self._meta._ih._ocp_world

    @property
    def indexcorners(self):
        """
        Redundant with Size.
        Ordinal coordinates of each of the 4 corners, ordered as "corners".
        """
        return self._meta._ih._ocp_index

    @property
    def annotcorners(self):
        """
        Redundant with Start, Inc, Size.
        Annotation coordinates of each of the 4 corners, ordered as HCorners.
        """
        return self._meta._ih._ocp_annot

    @property
    def bricksize(self):
        """
        Size of a brick. Should always be (64, 64, 64).
        """
        return self._meta._ih._bricksize

    @property
    def brickcount(self):
        """
        Number of bricks (including empties) ordered by [lod][dimension].
        """
        return self._meta._ih._lodsizes

    @property
    def nlods(self):
        """
        Number of level-of-detail layers, including lod 0 a.k.a. full resolution.
        Unlike the C++ version, nlods is NOT returned as 1 if lowres data is missing.
        """
        return self._meta._ih._nlods

    @staticmethod
    def _formatUUID(uuid):
        """
        Convert a little-endian binary UUID to a big-endian string version.
        See the C++ version for details.

        First part byteswaps as an uint32_t.
        Second and third part byteswaps as two uint16_t.
        Remaining two parts are not byteswapped.
        Hyphens added between parts.
        """
        return ("{3:02x}{2:02x}{1:02x}{0:02x}-" +
                "{5:02x}{4:02x}-{7:02x}{6:02x}-" +
                "{8:02x}{9:02x}-" +
                "{10:02x}{11:02x}{12:02x}{13:02x}{14:02x}{15:02x}").format(*uuid)

    #@property
    #def dataid(self):
    #    """
    #    GUID set on file creation.
    #    """
    #    return self._formatUUID(self._meta._ih._dataid)

    @property
    def verid(self):
        """
        GUID set each time the file is changed.
        """
        return self._formatUUID(self._meta._ih._verid)

    #@property
    #def previd(self):
    #    """
    #    GUID before last change.
    #    """
    #    return self._formatUUID(self._meta._ih._previd)

    @property
    def meta(self):
        """
        A dictionary of all the meta information, which can
        later be passed as **kwargs to the ZgyWriter constructor.
        and "indexcorners", "annotcorners", "brickcount", "nlods"
        are all derived properties that will never be settable.
        "numthreads" is a property of the implementation, not the file.
        """
        return {
            "size":        self.size,
            "bricksize":   self.bricksize,
            "datatype":    self.datatype,
            "datarange":   self.datarange,
            "zunitdim":    self.zunitdim,
            "hunitdim":    self.hunitdim,
            "zunitname":   self.zunitname,
            "hunitname":   self.hunitname,
            "zunitfactor": self.zunitfactor,
            "hunitfactor": self.hunitfactor,
            "zstart":      self.zstart,
            "zinc":        self.zinc,
            "annotstart":  self.annotstart,
            "annotinc":    self.annotinc,
            "corners":     self.corners,
        }

    @property
    def numthreads(self):
        """
        How many threads to use when reading. Currently ignored.
        """
        return 1
    @numthreads.setter
    def numthreads(self, x):
        print("Warning: numthreads is ignored.")

    def dump(self, file=None):
        file = file or sys.stdout
        print("{", file=file)
        for e in sorted(self.meta.items()):
            value = '"'+e[1]+'"' if isinstance(e[1], str) else str(e[1])
            print('  "{0}": {1},'.format(e[0], value), file=file)
        print("}", file=file)

    ### New in OpenZGY ###
    _statisticsType = namedtuple("Statistics", "cnt sum ssq min max")
    @property
    def statistics(self):
        """
        Return the statistics stored in the file header as a named tuple.
        NOTE, I might want to change this to another type if there is a
        need to implement the same method in the ZGY-Public wrapper,
        as it might be trickier to define a namedtuple there.
        """
        return self._statisticsType(self._meta._ih._scnt,
                                    self._meta._ih._ssum,
                                    self._meta._ih._sssq,
                                    self._meta._ih._smin,
                                    self._meta._ih._smax)

    _histogramType = namedtuple("Histogram", "cnt min max bin")
    @property
    def histogram(self):
        """
        Return the statistics stored in the file header as a named tuple.
        NOTE, I might want to change this to another type if there is a
        need to implement the same method in the ZGY-Public wrapper,
        as it might be trickier to define a namedtuple there.
        """
        if not self._meta._hh: return None
        return self._histogramType(self._meta._hh._cnt,
                                   self._meta._hh._min,
                                   self._meta._hh._max,
                                   self._meta._hh._bin)

##@brief Base class shared betewwn ZgyReader and ZgyWriter.
class ZgyMetaAndTools(ZgyMeta):
    """
    Base class shared betewwn ZgyReader and ZgyWriter.
    Adds coordinate conversion tools.
    """

    @staticmethod
    def transform(A, B, data):
        """
        Linear transformation of an array of double-precision coordinates.
        The coordinate systems to convert between are defined by
        three arbitrary points in the source system and the target.
        Arguments: ((ax0,ay0), (ax1,ay1), (ax2,ay2)),
                   ((bx0,by0), (bx1,by1), (bx2,by2)),
                   data
        where data is a 2d array of size (length, 2)
        """
        # performance note: In Python it can be far more efficient to
        # build and cache 6 transformation matrices between index/annot/world
        # and use those for the 6 transforms. But if we only transform
        # a few values at a time anyway, or if we are planning to convert
        # the accessor back to C++ fairly soon, this is a non-issue.
        _internal.generalTransform(
            A[0][0], A[0][1], A[1][0], A[1][1], A[2][0], A[2][1],
            B[0][0], B[0][1], B[1][0], B[1][1], B[2][0], B[2][1],
            data)

    @staticmethod
    def transform1(A, B, point):
        data = [[point[0], point[1]]]
        ZgyMetaAndTools.transform(A, B, data)
        return tuple(data[0])

    def annotToIndex(self, point):
        """Convert inline, crossline to ordinal"""
        return self.transform1(self.annotcorners, self.indexcorners, point)

    def annotToWorld(self, point):
        """Convert inline, crossline to world X,Y"""
        return self.transform1(self.annotcorners, self.corners, point)

    def indexToAnnot(self, point):
        """Convert ordinal to inline, crossline"""
        return self.transform1(self.indexcorners, self.annotcorners, point)

    def indexToWorld(self, point):
        """Convert ordinal to world X,Y"""
        return self.transform1(self.indexcorners, self.corners, point)

    def worldToAnnot(self, point):
        """Convert world X,Y to inline, crossline"""
        return self.transform1(self.corners, self.annotcorners, point)

    def worldToIndex(self, point):
        """Convert world X,Y to ordinal"""
        return self.transform1(self.corners, self.indexcorners, point)

##@brief Main entry point for reading ZGY files.
class ZgyReader(ZgyMetaAndTools):
    """
    Main entry point for reading ZGY files.

    Obtain a concrete instance by calling the constructor.
    You can then use the instance to read both meta data and bulk data.
    It is recommended to explicitly close the file when done with it.
    """
    def __init__(self, filename, *, _update=False, iocontext=None):
        # No "with" statement for the FileFactory, so we must remember
        # to close it ourselves in our own __exit__.
        self._fd = _internal.FileFactory(filename, ("r+b" if _update else "rb"), iocontext)
        self._meta = _internal.ZgyInternalMeta(self._fd)
        # self._meta._ih and friends will all be allocated.
        # Prove that all the tests for "._ih is not None" are redundant.
        self._meta._assert_all_headers_allocated()
        # At the implementation level the bulk- and meta access are separate,
        # and the bulk accessor needs some of the meta information to work.
        self._accessor = _internal.ZgyInternalBulk(self._fd, self._meta)
        # This causes an assignment to the parent's self._meta
        # which in Python is a no-op but in C++ the parent might
        # have its own _meta that we shadow here. Or not.
        super().__init__(self._meta)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Note that if the block was exited due to an exception, and if we
        # also get an exception from close, then it is the second one
        # that gets caught in a try/catch block placed outside the "while".
        # Callers will in this case often want to report just the first
        # exception since the close() probably failed as a cause of it.
        # Caller needs to do e.g. "ex2 = ex.__cause__ or ex.__context__".
        # It is possible to instead suppress any errors from close if
        # another exception is already pending. Simplifying the caller.
        # But then the close() would not be available at all. Bad idea.
        self.close()

    ##@brief Read bulk data into a caller specified buffer.
    def read(self, start, data, *, lod = 0, verbose = None, zeroed_data = False):
        """
        Read an arbitraty region of bulk data into a caller specified buffer.

        The buffer's type must be float, short, or char. Any file may
        be read as float. If the buffer is of type short or char then
        the file must be of precisely that type.

        Arguments: (i0,j0,k0), buffer, lod=0
        """
        file_dtype = _internal.enum._map_DataTypeToNumpyType(self._meta._ih._datatype)
        if (not isinstance(data, np.ndarray) or
            data.dtype not in (np.float32, file_dtype) or
            len(data.shape) != 3 or
            not data.flags.writeable
        ):
            raise ZgyUserError("Expected a writeable 3d numpy array of np.{0} or np.{1}".format(
                np.dtype(np.float32).name, np.dtype(file_dtype).name))
        if not self._accessor: raise ZgyUserError("ZGY file is not open for read.")
        self._accessor.readToExistingBuffer(data, start, lod=lod,
                                            as_float=(data.dtype==np.float32),
                                            verbose=verbose, zeroed_result=zeroed_data)

    ##@brief Get hint about all constant region.
    ##@image html readconst-fig1.png
    ##@image latex readconst-fig1.png
    def readconst(self, start, size, *, lod = 0, as_float = True, verbose = None):
        """
        Get hint about all constant region.

        Check to see if the specified region is known to have all samples
        set to the same value. Returns that value, or None if it isn't.

        The function only makes inexpensive checks so it might return
        None even if the region was in fact constant. It will not make
        the opposite mistake. This method is only intended as a hint
        to improve performance.

        For int8 and int16 files the caller may specify whether to scale
        the values or not.
        """
        if not self._accessor: raise ZgyUserError("ZGY file not open for read")
        return self._accessor.readConstantValue(start, size, lod=lod,
                                                as_float=as_float,
                                                verbose=verbose)

    def close(self):
        if self._fd:
            self._fd.xx_close()
            self._fd = None
        if self._accessor:
            self._accessor = None # No other cleanup needed.
        # Metadata remains accessible. Not sure whether this is a good idea.


##@brief Main API for creating ZGY files.
class ZgyWriter(ZgyMetaAndTools):
    """
    Main API for creating ZGY files.

    Obtain a concrete instance by calling the constructor.
    All meta data is specified in the call to open(), so meta data
    will appear to be read only. You can use the instance to write
    bulk data. The file becomes read only once the instance is closed.

    It is recommended to call finalize() and close() after all bulk
    has been written. But if you forget then this will be done when
    the writer goes out of scope, provided of course that you used a
    "with" block.
    """
    def __init__(self, filename, *,
                 iocontext=None, compressor=None, lodcompressor=None, **kwargs):
        """
        Create a new ZGY file.

        Optionally pass templatename = otherfile to create a new file
        similar to otherfile. Explicit keyword argumants override
        information from otherfile.

        Optionally pass templatename = filename to erase all data blocks
        from filename but keep the metadata. New data blocks can then
        be written to the file. Petrel/BASE might need this feature,
        due to the way it writes new files. They tend to get opened
        several times to add meta information. Caveat: Behind the
        scenes the file is simply deleted and re-created. This is
        slightly less efficient than opening the file for read/write.

        templatename: string
            Optionally create a file similar to this one.
            TODO-Low: In the future might also accept a ZgyReader instance.
            This is convenient if a file is being copied, so as to not
            needing to open it twice.

        filename: string
            The local or cloud file to create.

        size: (int, int, int)
            Number of inlines, crosslines, samples.

        bricksize: (int, int, int)
            Size of a single brick. Defaults to (64, 64, 64).
            Please use the default unless you really know what
            you are doing. In any case, each size needs to be
            a power of 2.

        datatype: SampleDataType
            Specify int8, int16, or float.

        datarange = (float, float)
            Used only if datatype is integral, to convert from storage to
            actual sample values. The lowest possible storage value, i.e.
            -128 or -32768, maps to datarange[0] and the highest possible
            storage value maps to datarange[1].

        zunitdim:    UnitDimension. time, length, or unknown.
        zunitname:   string
        zunitfactor: float
            Describe how to convert between storage units and SI units
            in the vertical direction. Petrel ignores these settings and
            prompts the user.

        hunitdim:    UnitDimension. length, arcangle, or unknown.
        hunitname:   string
        hunitfactor: float
            Describe how to convert between storage units and SI units
            in the horizontal direction. Most applications cannot handle
            arcangle. Petrel ignores these settings and prompts the user.

        zstart: float
            The time or depth corresponding to the shallowest sample.

        zinc: float
            The vertical time or depth distance between neighboring samples.

        annotstart: (float, float)
            The inline and crossline numbers corresponding to the ordinal
            position (0, 0) i.e. the first sample on the file.

        annotinc: (float, float)
            The inline / crossline step between neighboring samples.
            The samples at ordinal (1, 1) will have annotation
            annotstart + annotinc.

        corners: (float, float)[4]
            World coordinates of each corner, order as
                First inline / first crossline,
                last inline / first crossline,
                first inline / last crossline,
                last inline / last crossline.

        compressor, lodcompressor: callable
            If set, attempt to compress each block with this callable.
            Typically this should be a lambda or a class, because it
            needs to capture the compression parameters.
            Example:
                compressor = ZgyCompressFactory("ZFP", snr = 30)
            If different compression parameters are desired for
            full- and low resolution bricks then lodcompressor can
            be provided as well. It defaults to compressor. Using
            two different instances, even if the parameters match,
            may also cause statistics to be reported separately
            for fullres and lowres.
            TODO-Low: Future: passing zfp_compressor = snr is equivalent
            to compressor = ZgyCompressFactory("ZFP", snr = snr).
            Unlike the compressor keyword this also works in the wrapper.
        """
        # The following arguments are not passed on to _create:
        #   -  templatename is handled locally, using the template for defaults.
        #   -  iocontext is only made available to the FileADT layer
        #   -  compressor is explicitly passed as an argument to those
        #      functions (write, writeconst, finalize) that need it.

        if "templatename" in kwargs:
            with ZgyReader(kwargs["templatename"]) as t:
                for k, v in t.meta.items():
                    if not k in kwargs:
                        kwargs[k] = v
            del kwargs["templatename"]

        # Compressing a file as integer is not useful, it just adds more noise
        # at least as long as we are talking about seismic. Even if we allowed
        # int8 or int16 here the code in impl_zfp_compress will currently
        # use the ZFP float interface.
        #
        # Flagging as int8 or int16 would save memory at the cost of adding
        # even more noise. Converting the decompressed data to integral before
        # returning it as that type. But if the applicaton wants this then
        # it can easily convert the result itself.
        #
        # There are other subtle issues with int8 / int16. Even with enabled
        # compression, individual bricks are allowed to be stored uncompressed
        # if neither lossy nor lossless compression works as desired. The
        # declared value type then controls how these blocks are stored.
        # Storing those (presumably very few) blocks as integral means
        # more that can go wrong and more unit  tests. And keep in mind
        # that float->int8 and float->int16 are also a form of compression
        # (4x and 2x respectively) but is a lot noisier than ZFP at the same
        # compression factor.
        #
        # We may need to revisit this if switching to another compression
        # algorithm where integral compression works better.
        if compressor and kwargs.get("datatype", SampleDataType.float) != SampleDataType.float:
            raise ZgyUserError("Compressed files need to be stored as float.")

        # After this, self._meta._ih and friends exists but will be None.
        self._meta = _internal.ZgyInternalMeta(None)
        # This causes an assignment to the parent's self._meta
        # which in Python is a no-op but in C++ the parent might
        # have its own _meta that we shadow here. Or not.
        super().__init__(self._meta)
        self._create(filename, compressed = bool(compressor or lodcompressor), **kwargs)
        # Now self._meta._ih and friends will all be allocated.
        # Prove that all the tests for "._ih is not None" are redundant.
        self._meta._assert_all_headers_allocated()
        # The file creation was deferred until after the consistency checks.
        # No "with" statement for the FileFactory, so we must remember
        # to close it ourself in our own __exit__.
        self._fd = _internal.FileFactory(filename, "w+b", iocontext)
        self._meta._flush_meta(self._fd)
        # The accessor needs to know whether we will do compression or not,
        # because this determines whether bricks will be written aligned
        # and possibly whether updates are allowed. The actual
        # compression algorithm is passed on each write etc.
        # TODO-Low consider storing the compressor as context of the accessor
        # instead. Less precise control though. We might want a different
        # snr on the low resolution bricks.
        self._accessor = _internal.ZgyInternalBulk(
            self._fd, self._meta,
            compressed = bool(compressor or lodcompressor))
        self._dirty = False # If True need LOD, stats, histogram.
        self._compressor = compressor or lodcompressor
        self._lodcompressor = lodcompressor or compressor

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Note that if the block was exited due to an exception, and if we
        # also get an exception from close, then it is the second one
        # that gets caught in a try/catch block placed outside the "while".
        # Callers will in this case often want to report just the first
        # exception since the close() probably failed as a cause of it.
        # Caller needs to do e.g. "ex2 = ex.__cause__ or ex.__context__".
        # This is mitigated by the close() method skipping work and/or
        # suppressing exceptions if the file has already been flagged bad.
        self.close()

    def _create(self, filename, *, size = None, compressed = False,
                bricksize = None,
                datatype = SampleDataType.float, datarange = None,
                zunitdim = UnitDimension.unknown,
                hunitdim = UnitDimension.unknown,
                zunitname = None, hunitname = None,
                zunitfactor = 0.0, hunitfactor = 0.0, zstart = 0.0, zinc = 0.0,
                annotstart = (0, 0), annotinc = (0, 0),
                corners = ((0,0),(0,0),(0,0),(0,0))):
        """
        Called from __init__. Do not call directly.
        The user should use a "using" statement when creating the reader.
        Datatype can be "int8", "int16", "float".
        Dimension can be "time", "length", or "arcangle".
        """
        self._meta._init_from_scratch(filename = filename,
                                size = size,
                                compressed = compressed, # If True this is V4.
                                bricksize = bricksize,
                                datatype = _map_SampleDataTypeToDataType(datatype),
                                datarange = datarange,
                                zunitdim = _map_UnitDimensionToVerticalDimension(zunitdim),
                                hunitdim = _map_UnitDimensionToHorizontalDimension(hunitdim),
                                zunitname = zunitname,
                                hunitname = hunitname,
                                zunitfactor = zunitfactor,
                                hunitfactor = hunitfactor,
                                zstart = zstart,
                                zinc = zinc,
                                annotstart = annotstart,
                                annotinc = annotinc,
                                corners = corners)

    ##@brief Write an arbitrary region.
    def write(self, start, data, *, verbose = None):
        """
        Write bulk data. Type must be np.float32, np.int16, or np.int8.
        np.float32 may be written to any file and will be converted
        if needed before storing. Writing an integral type implies that
        values are to be written without conversion. In that case the
        type of the buffer must match exactly the file's storage type.
        You cannot write int8 data to an int16 file or vice versa.

        Arguments:
            start: tuple(i0,j0,k0) where to start writing.
            data: np.ndarray of np.int8, np.int16, or np.float32
        """
        file_dtype = _internal.enum._map_DataTypeToNumpyType(self._meta._ih._datatype)
        if (not isinstance(data, np.ndarray) or
            data.dtype not in (np.float32, file_dtype) or
            len(data.shape) != 3
        ):
            raise ZgyUserError("Expected a 3d numpy array of np.{0} or np.{1}".format(
                np.dtype(np.float32).name, np.dtype(file_dtype).name))
        if not self._accessor: raise ZgyUserError("ZGY file is not open.")
        if self._accessor._is_bad or self._meta._is_bad:
            raise ZgyCorruptedFile("Cannot continue due to previous errors.")
        self._accessor._writeRegion(data, start, lod=0,
                                    compressor=self._compressor,
                                    is_storage=(data.dtype != np.float32),
                                    verbose=verbose)
        self._dirty = True

    ##@brief Write a single value to a region of the file.
    def writeconst(self, start, value, size, is_storage, *, verbose = None):
        """
        Write a single value to a region of the file.

        This is equivalent to creating a constant-value array with np.full()
        and write that. But this version might be considerably faster.

        If is_storage is false and the input value cannot be converted to
        storage values due to being outside range after conversion then
        the normal rules (use closest valid value) apply. If
        is_storage is True then an error is raised if the supplied value
        cannot be represented.

        Arguments:
            start:     tuple(i0,j0,k0) where to start writing.
            size:      tuple(ni,nj,nk) size of region to write.
            value:     Scalar to be written.
            is_storge: False if the value shall be converted to storage
                       True if it is already storage and should be written
                       unchanged. Ignored if the storage type is float.
        """
        if self._meta._ih._datatype == _internal.enum.RawDataType.Float32:
            dtype = np.float32 # "Convert" to user now. Actually a no-op.
            is_storage = False
        elif not is_storage:
            dtype = np.float32 # Force conversion.
        elif self._meta._ih._datatype == _internal.enum.RawDataType.SignedInt16:
            dtype = np.int16
        elif self._meta._ih._datatype == _internal.enum.RawDataType.SignedInt8:
            dtype = np.int8
        else:
            raise ZgyFormatError("Unrecognized datatype on file")
        if np.issubdtype(dtype, np.integer) and not np.isfinite(value):
            raise ZgyUserError("Cannot store {0} in a {1}".format(value, np.dtype(dtype)))
        self._accessor._writeRegion(_internal.ScalarBuffer(size, value, dtype),
                                    start, lod=0,
                                    compressor=self._compressor,
                                    is_storage=is_storage,
                                    verbose=verbose)
        self._dirty = True

    def finalize(self, *, decimation=None, progress=None, force=False, verbose=None):
        """
        Generate low resolution data, statistics, and histogram.
        This will be called automatically from close(), but in
        that case it is not possible to request a progress callback.

        If the processing raises an exception the data is still marked
        as clean. Called can force a retry by passing force=True.

        Arguments:
            decimation: Optionally override the decimation algorithms by
                        passing an array of impl.lodalgo.DecimationType
                        with one entry for each level of detail. If the
                        array is too short then the last entry is used for
                        subsequent levels.
                        TODO-Low: The expected enum type is technically internal
                        and ought to have been mapped to an enum api.XXX.
            progress:   Function(done, total) called to report progress.
                        If it returns False the computation is aborted.
                        Will be called at least one, even if there is no work.
            force:      If true, generate the low resolution data even if
                        it appears to not be needed. Use with caution.
                        Especially if writing to the cloud, where data
                        should only be written once.
            verbose:    optional function to print diagnostic information.
        """
        if self._dirty or force:
            self._dirty = False
            stats, histo = _internal.GenLodC(
                accessor   = self._accessor,
                compressor = self._lodcompressor,
                decimation = decimation,
                progress   = progress,
                verbose    = verbose)()
            # TODO-Low: Refactor:
            # violating encapsulation rather more than usual.
            # Note that _accessor._metadata is private; it is a copy
            # or actually a reference letting ZgyInternalBulk use
            # some parts of the metadata.
            (a, b) = self._accessor._scaleDataFactorsStorageToFloat()
            stats.scale(a, b)
            histo.scale(a, b)
            histo.resize(256)
            self._meta._ih._scnt = stats._cnt
            self._meta._ih._ssum = stats._sum
            self._meta._ih._sssq = stats._ssq
            self._meta._ih._smin = stats._min
            self._meta._ih._smax = stats._max
            self._meta._hh._min = histo.vv_range[0]
            self._meta._hh._max = histo.vv_range[1]
            self._meta._hh._bin = histo.bins
            self._meta._hh._cnt = np.sum(histo.bins)
        else:
            if progress:
                progress(0, 0)
        # For debugging and measurements only.
        if False:
            if self._compressor:
                self._compressor.dump(msg="Compress")
            if self._lodcompressor and not self._lodcompressor is self._compressor:
                self._lodcompressor.dump(msg="LOD_data")

        # TODO-Low: Refactor: the accessor should logically have a close(),
        # shouldn't it? And maybe I shouldn't set self._fd here.
        # Allowing the accessor to manage it.

    def close(self):
        """
        Close the currently open file.
        Failure to close the file will corrupt it.
        """
        if self._fd:
            # Both the meta accessor and the bulk accessor has an _is_bad
            # flag that is set if we got an error while writing to the file.
            # If set then don't bother with statistics, histogram, and lowres.
            # The file will probably just be discarded amyway.
            if not (self._meta._is_bad or self._accessor._is_bad):
                self.finalize()

            # Flushing metadata is a bit more ambiguous. Parts of the file
            # might still be salvageable, so especially when testing we might
            # still want to flush a file marked as bad. But, ignore any
            # secondary errors as the user already knows something is wrong.
            if not (self._meta._is_bad or self._accessor._is_bad):
                self._meta._flush_meta(self._fd)
            else:
                try:
                    self._meta._is_bad = False
                    self._accessor._is_bad = False
                    self._meta._flush_meta(self._fd)
                except Exception:
                    pass
                finally:
                    self._meta._is_bad = True
                    self._accessor._is_bad = True

            # TODO-Low it would be more consistent if the final write was
            # of the updated headers, in case the app crashes before
            # xx_close. This is true for local file access. But for
            # cloud access the last bulk data segment might be written
            # on xx_close(). Difficult to change without complicating
            # the internal FileADT api.

            # Closing the local or cloud handle is always needed as there
            # might be resources that need to be cleaned up.
            self._fd.xx_close()
            self._fd = None

            # The client code is strongly advised to delete the file if it was
            # opened for create. OpenZGY might have deleted the file itself but
            # this is probably too harsh. A file opened for update (which is not
            # actually supported yet) might still be usable. So in the future we
            # may need an additional flag telling whether the writer has ever
            # had any successful writes to the file. If not then the file is
            # still good. Note that the suggestions below need more work.
            # TODO-Low: Later: this is in the bells & whistles category.
            #if not self._precious_set_on_open:
            #    self._fd.xx_delete_on_close(); self._fd.xx_close()
            #    self._fd.xx_close_of_needed_and_delete()
            #    ZgyUtils(saved_iocontext).delete(self._filename)

    ##@cond IMPL
    @property
    def errorflag(self):
        """
        If true this means that an error happened during a critical operation
        such as a write. This means the file being written is probably bad.
        It is recommended to close and delete the file. In the future
        OpenZGY might even do that automatically.
        """
        return self._meta._is_bad or self._accessor._is_bad

    @errorflag.setter
    def errorflag(self, value):
        """
        Set or reset the flag indicating that the file got corrupted.
        """
        self._meta._is_bad = value
        self._accessor._is_bad = value
    ##@endcond

##@brief Simple progress bar.
class ProgressWithDots:
    """
    Progress bar that writes dots (51 by default) to standard output.
    This can be user as-is for simple command line apps, or you can use
    the source code as an example on how to write your own.

    The default of 51 dots will print one dot at startup and then one
    additional dot for each 2% work done.

    If you are using this to write to the cloud a file that is smaller
    than ~10 GB then the progress bar will probably move in larger
    jumps. Because writing to a cloud back-end uses very large buffers.
    Most cloud back-ends cannot report progress inside a "write block".

    When passing a progress reporter to a function, make sure you do not
    pass the class itself. You need to create an instance of it.
    """
    def __init__(self, length=51, outfile=sys.stderr):
        self._dots_printed = 0
        self._length = length
        self._outfile = outfile
    def __call__(self, done, total):
        #print("Progress: {0}/{1}".format(done, total))
        if self._dots_printed == 0:
            print("[" + (" " * self._length) + "]\r[", end='', flush=True)
        needed = 1 if not total else 1 + ((done * (self._length-1)) // total)
        if needed > self._dots_printed:
            print("." * (needed - self._dots_printed),
                  file=self._outfile, flush=True, end='')
            self._dots_printed = needed
        if done == total:
            print("", file=self._outfile)
        return True

##@brief Operations other than read and write.
class ZgyUtils:
    """
    Operations other than read and write.

    Any operations that don't fit into ZgyReader or ZgyWriter go here.
    Such as deleting a file. Or any other operation that does not need
    the file to be open first.
    """
    ##@brief Create a new concrete instance of ZgyUtils.
    def __init__(self, iocontext=None):
        """
        Create a new concrete instance of ZgyUtils.

        The reason you need to supply a file name or a file name prefix is that
        you need to provide enough information to identify the back-end that
        this instance will be bound to. So if you have registered a back-end
        named "xx", both "xx://some/bogus/file.zgy" and just "xx://" will
        produce an instance that works for your XX backend,

        For performance reasons you should consider caching one ZgyUtils
        instance for each back end you will be using. Instead of just creating
        a new one each time you want to invoke a method. Just remember that
        most operations need an instance created with the same prefix.
        """
        self._iocontext = iocontext

    ##@brief Delete a file. Works both for local and cloud files.
    def delete(self, filename):
        """
        Delete a file. Works both for local and cloud files.
        Note that the instance must be of the correct (local or cloud) type.
        """
        with _internal.FileFactory(filename, "d", self._iocontext) as f:
            f.xx_close()

def ZgyCompressFactory(name, *args, **kwargs):
    """
    Look up a compression algorithm by name and instanciate a compressor,
    passing the required compression parameters. Using this approach
    reduces the coupling between client code and the compressor.
    """
    return _internal.CompressFactoryImpl.factory(name, *args, **kwargs)

def ZgyKnownCompressors():
    """
    Return the names of all compressors known to the system.
    This is primarily for logging, but might in principle be used
    in a GUI to present a list of compressors to choose from.
    The problem with that is how to handle the argument list.
    """
    return _internal.CompressFactoryImpl.knownCompressors()

def ZgyKnownDecompressors():
    """
    Return the names of all compressors known to the system.
    This is primarily for logging.
    """
    return _internal.CompressFactoryImpl.knownDecompressors()

#############################################################################
###   Define enums used in the public API. These are separate from the    ###
###   enums used inside ZGY files, to improve isolation.                  ###
#############################################################################

def _map_DataTypeToSampleDataType(e):
    return _mapEnum(e, {
        _internal.enum.RawDataType.SignedInt8:  SampleDataType.int8,
        _internal.enum.RawDataType.SignedInt16: SampleDataType.int16,
        _internal.enum.RawDataType.Float32:     SampleDataType.float,
        None:                       SampleDataType.unknown,
    })

def _map_SampleDataTypeToDataType(e):
    return _mapEnum(e, {
        SampleDataType.int8:  _internal.enum.RawDataType.SignedInt8,
        SampleDataType.int16: _internal.enum.RawDataType.SignedInt16,
        SampleDataType.float: _internal.enum.RawDataType.Float32,
    })

def _map_HorizontalDimensionToUnitDimension(e):
    return _mapEnum(e, {
        _internal.enum.RawHorizontalDimension.Length:   UnitDimension.length,
        _internal.enum.RawHorizontalDimension.ArcAngle: UnitDimension.arcangle,
        None:                               UnitDimension.unknown,
    })

def _map_VerticalDimensionToUnitDimension(e):
    return _mapEnum(e, {
        _internal.enum.RawVerticalDimension.Depth:      UnitDimension.length,
        _internal.enum.RawVerticalDimension.SeismicTWT: UnitDimension.time,
        _internal.enum.RawVerticalDimension.SeismicOWT: UnitDimension.time,
        None:                               UnitDimension.unknown,
    })

def _map_UnitDimensionToHorizontalDimension(e):
    return _mapEnum(e, {
        UnitDimension.length:   _internal.enum.RawHorizontalDimension.Length,
        UnitDimension.arcangle: _internal.enum.RawHorizontalDimension.ArcAngle,
        UnitDimension.unknown: _internal.enum.RawHorizontalDimension.Unknown,
    })

def _map_UnitDimensionToVerticalDimension(e):
    return _mapEnum(e, {
        UnitDimension.time:   _internal.enum.RawVerticalDimension.SeismicTWT,
        UnitDimension.length: _internal.enum.RawVerticalDimension.Depth,
        UnitDimension.unknown: _internal.enum.RawVerticalDimension.Unknown,
    })

def _mapEnum(e, lookup):
    """
    Internal method to map between impl.enum tags used in the file format
    and those used in the API, to better isolate the API from changes.

    An unrecognized tag when mapping from file format to api is
    usually treated as a warning. The application might be able to
    handle it. In the lookup table, if there is an entry with key None
    then this is considered to be the default value. If there is no
    such entry then we are expected to raise an exception on error.

    When mapping in the other direction the value came
    from user code so raising an exception is usually warranted.
    """
    if e in lookup: return lookup[e]
    if None in lookup: return lookup[None]
    valid = ", ".join([str(e) for e in sorted(lookup, key = lambda x: x.value)])
    raise ZgyUserError("Value " + str(e) + " not accepted." +
                       " It should be one of " + valid + ".")

if __name__ == "__main__":
    help(ZgyReader)
    help(ZgyWriter)

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
