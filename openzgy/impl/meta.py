#!/usr/bin/env python3

"""
Meta data read/write

This file contains a number of classes and free functions dealing with
meta data access, i.e. everything in the ZGY file except the actual
bulk data. Nothing in this file should be directly visible to users of
the public API.

    impl.meta.ZgyInternalMeta:

        * Holds a collection of header instances, one of each type.
        * A function to populate all the instances, mainly be invoking
          read() in each header instance, in the right order.
          With the current format there are several chicken and egg problems.
        * A function to populate all the header instances when creating
          a new file.
        * A function to flush all headers instances to file after writing.
        * The user visible ZgyReader and ZgyWriter classes should contain
          an instance of this class, using composition not inheritance
          because there are no methods or data in this class that the user
          is allowed to use directly.

    FileHeader
    OffsetHeader{V1,V2,V3,V4}
    InfoHeader{V1,V2,V3,V4}
    StringList{V1,V2,V3,V4}
    HistHeader{V1,V2,V3,V4}
    AlphaLUP{V1,V2,V3,V4}
    BrickLUP{V1,V2,V3,V4}:

        * All these represent a particular version of one of the headers
          that make up a complete ZGY file. Each header knows about the
          physical layout on the file. In C++ the class should be a POD
          with an exact match to what is found on file. If additional
          data members are needed they should be introduced in a derived
          class. In Python there are no POD types so specifying the
          physical layout is a bit more involved. And any derived data
          might as well be included in the same class.

    OffsetHeaderFactory, InfoHeaderFactory, etc.

        * Free functions to create an instance of this type, with the
          numerical version number passed in.

    HeaderBase, LookupTable

        * These are "convenience" base classes for sharing code;
          they are not intended to provide polymorphic behavior.
"""
##@package openzgy.impl
#@brief All implementation details are in this namespace.
##@package openzgy.impl.meta
#@brief Meta data read/write.

import struct
import sys
import numpy as np
import json
import math
import random
from enum import Enum

from ..impl.transform import acpToOcp
from ..impl import enum as impl_enum
from ..impl import file as impl_file
from ..exception import *

def _checked_read(f, offset, size, *, usagehint=impl_file.UsageHint.Header):
    data = f.xx_read(offset, size, usagehint=usagehint)
    if len(data) != size:
        if len(data) < size:
            raise ZgyFormatError("Got EOF reading header.")
        else:
            raise ZgyInternalError("Got too much data reading header.")
    return data

def _fix_codingrange(r, datatype):
    """
    Sanity check. If the codingrange for an int cube is bad, silently
    use a range that causes no conversion between storage and float.

    This avoids several corner cases both inside OpenZGY and in applications.

    Rationale: A non-finite range is always bad. A range with min==max
    is technically valid when reading, as all storage values would map
    to the same float value. But this is almost certainly not what the
    writer intended. Similarly a range with min>max technically means
    that increasing storage values correspond to decreasing float values.
    Again, the values are more likely to be completely bogus.

    Leave the range alone if this is a float cube. There is no conversion
    involved, and for float the codingrange is ignored by the API anyway.

    For files written by the OpenZGY library an exception wouls be thrown
    on create. So the codingrange should always be ok for those.

    Note: The sanity check could also be applied to the histogram range.
    That fix would also apply to float cubes. The histogram is less
    important though, and it should be ok to let the application worry
    about a bad histogram.
    """
    file_dtype = impl_enum._map_DataTypeToNumpyType(datatype)
    if np.issubdtype(file_dtype, np.integer):
        iinfo = np.iinfo(file_dtype)
        if (not r or
            not np.isfinite(r[0]) or
            not np.isfinite(r[1]) or
            r[1] <= r[0]):
            #print("Bad codingrange", r, "-> use", (iinfo.min, iinfo.max))
            r = (float(iinfo.min), float(iinfo.max))
    return r

class ErrorsWillCorruptFile:
    """
    Duplicated between impl.bulk and impl.meta. Maybe fix sometime.
    """
    def __init__(self, parent): self._parent = parent
    def __enter__(self): return None
    def __exit__(self, type, value, traceback):
        if type:
            #print("Meta: Exit critical section", str(value))
            self._parent._is_bad = True

class HeaderBase:
    """
    Convenience base class for implementing classes that map 1:1 to a
    specific header with a specific version that exists in a ZGY file.
    The constructor should unpack a supplied byte buffer into a new
    instance of this header. Caller is responsible for all I/O, so no
    methods in this class need an iocontext except read(). The various
    headers don't need to inherit this base class if they don't want to.
    """

    @classmethod
    def _formats(cls):
        """
        Define the physical header layout. This method needs to be
        overridden in a subclass. Otherwise the header will be empty.
        The second field is the format as recognized by the 'struct' module.
        Data on the file is stored packed and little-endian. So for use with
        'struct', a '<' should be prepended to the string.
        """
        return []

    @classmethod
    def _format(cls):
        """
        Describe the layout of this header block on the file,
        showing only the offsets as a string recognized by 'struct'.
        May be used as an argument to struct.calcsize(). Can technically
        also be used with struct.unpack to read the entire struct at
        once. But, that will returns a flat list which is tricky to
        assign to the respective attributes.
        """
        return ("<" + " ".join([ e[1] for e in cls._formats() ])).strip()

    @classmethod
    def headersize(cls):
        """
        Return the size this header has on disk.
        """
        return struct.calcsize(cls._format())

    @classmethod
    def checkformats(cls, verbose = False, *, file = None):
        """
        Helper to compare the python definition of the header layout with
        the C++ version. Also check that the same attribute isn't listed twice.
        """
        file = file or sys.stdout
        mapping = {
            "char*":   "",
            "enum":    "B",
            "float32": "f",
            "float64": "d",
            "int32":   "i",
            "int64":   "q",
            "uint32":  "I",
            "uint64":  "Q",
            "uint8":   "B",
        }
        errors = 0
        seen = set()
        byteoffset = 0
        for e in cls._formats():
            if e[0] in seen:
                print("# ERROR: attribute {0} is listed twice.".format(e[0]), file=file)
            seen.add(e[0])
            ctype = e[2]
            cname = e[0]
            csize = None
            p1 = ctype.find("[")
            p2 = ctype.find("]")
            if p1 > 0 and p2 > p1:
                csize = ctype[p1+1:p2]
                cname = cname + "[" + csize + "]"
                ctype = ctype[0:p1]
            expect = (csize if csize else '') + mapping[ctype]
            if expect == "16B": expect = "16s"
            if expect == "4B": expect = "4s"
            actual = e[1]
            if actual and actual != expect:
                print("# ERROR: Expected code {0}, got {1}".format(expect, e[1]), file=file)
                errors += 1
            byteoffset += struct.calcsize(e[1])
        assert not errors

    def pack(self):
        """
        Convert the contents of this class to a byte array suitable for
        storing in the ZGY file.
        """
        self.checkformats()
        buf = bytearray(self.headersize())
        offset = 0
        for e in self._formats():
            if e[1]:
                value = getattr(self, e[0])
                if isinstance(value, Enum):
                    value = value.value
                if isinstance(value, (list, np.ndarray)):
                    value = tuple(value)
                if not type(value) is tuple: value = (value,)
                data = struct.pack_into("<" + e[1], buf, offset, *value)
                offset += struct.calcsize("<" + e[1])
        assert offset == self.headersize()
        return buf

    def unpack(self, buf = None):
        """
        Convert a byte array as read from the ZGY file into a Python object.
        Normally a call to unpack() will immediately be followed by a call
        to calculate() to fill in any derived information, convert enume, etc.
        If buf is None, unpacking is done on an all zero buffer. This ensures
        that all data fields are present in the object. Simplifying things
        if the application is creating an instance from scratch.
        """
        self.checkformats()
        if not buf: buf = bytes(self.headersize())
        offset = 0
        for e in self._formats():
            if e[1]:
                data = struct.unpack_from("<" + e[1], buf, offset=offset)
                offset += struct.calcsize("<" + e[1])
                setattr(self, e[0], data if len(data) > 1 else data[0])
            else:
                setattr(self, e[0], None)
        assert offset == self.headersize()

    @classmethod
    def read(cls, f, offset):
        """
        Read the header from disk and parse it, returning a new instance.
        """
        return cls(_checked_read(f, offset, cls.headersize()))

    def dump(self, prefix = "", file = None):
        """
        Print the entire contents of the object, including derived fields.
        """
        file = file or sys.stdout
        print("\n{0} takes up {1} bytes with format '{2}'".format(
            self.__class__.__name__, self.headersize(), self._format()), file=file)
        for e in self._formats():
            print("{0}{1:12} = {2}".format(prefix, e[0], getattr(self, e[0])), file=file)

#****************************************************************************
#** FileHeader **************************************************************
#****************************************************************************

class FileHeader(HeaderBase):
    """
    Unpack a byte buffer into a new instance of this header.
    Caller is responsible for all I/O, so we don't need an iocontext.
    """

    def __init__(self, buf = None, *, compressed = None):
        assert self.headersize() == 8
        # Python isn't really good at overloaded methods.
        assert (buf is None and compressed is not None) or (buf is not None and compressed is None)
        if buf:
            self.unpack(buf)
            assert buf == self.pack()
            if self._magic == b'VCS\x00': raise ZgyFormatError("Old ZGY compressed files are not supported")
            if self._magic != b'VBS\x00': raise ZgyFormatError("Not an uncompressed ZGY file")
            if self._version < 1 or self._version > 4: raise ZgyFormatError("Unsupported ZGY version " + str(version))
        else:
            # Prepare for writing a new file.
            self._magic = b'VBS\x00'
            self._version = 4 if compressed else 3

    @staticmethod
    def _formats():
        return [
            ('_magic', '4s', 'uint8[4]', 'Always VBS\\0.'),
            ('_version', 'I', 'uint32', 'Current version is 3 or 4.'),
        ]

#****************************************************************************
#** OffsetHeader ************************************************************
#****************************************************************************

class OffsetHeaderV1:
    def __init__(self):
        """
        Unpack a byte buffer into a new instance of this header.
        Caller is responsible for all I/O, so we don't need an iocontext.
        """
        self._infsize = 0
        self._strsize = 0
        self._histsize = 0
        self._alphalupsize = 0
        self._bricklupsize =0
        self._infoff = -1
        self._stroff = 0 # not used in V1
        self._histoff = -1
        self._alphalupoff = -1
        self._bricklupoff = -1
        self.calculate()

    @classmethod
    def _format(self):
        return "<8I"

    @classmethod
    def size(self):
        return struct.calcsize(self._format())

    @classmethod
    def load(self, buf):
        data = struct.unpack(self._format(), buf)
        r = self()
        # Weird combination of big-endian and little-endian.
        # Also beware that it was cast to signed int.
        # Negative offsets probably yield undefined behavior.
        # But that should not have been possible anyway.
        r._infoff = (data[0] << 32) + data[1]
        r._stroff = 0
        r._alphalupoff = (data[2] << 32) + data[3]
        r._bricklupoff = (data[4] << 32) + data[5]
        r._histoff = (data[6] << 32) + data[7]
        #print("###", r._infoff)
        return r

    def calculate(self, ih = None):
        """
        Offsets are stored explicitly, so this only needs to set size.
        """
        self._infsize = InfoHeaderV1.headersize()
        self._histsize = HistHeaderV1.headersize()
        self._alphalupsize = 8 * ih._alphaoffsets[-1] if ih else 0
        self._bricklupsize = 8 * ih._brickoffsets[-1] if ih else 0

    @classmethod
    def read(cls, f):
        # The OffsetHeader always starts immediately after the FileHeader,
        # which always starts at the beginning of the file.
        return cls.load(_checked_read(f, FileHeader.headersize(), cls.size()))

    def dump(self, *, file=None):
        file = file or sys.stdout
        print("\n{0} takes up {1} bytes with format '{2}'".format(self.__class__.__name__, self.size(), self._format()), file=file)
        print("  Offsets: info {0:x} str {1:x} hist {2:x} alpha {3:x} brick {4:x}".format(
            self._infoff, self._stroff, self._histoff, self._alphalupoff, self._bricklupoff), file=file)

    @staticmethod
    def _formats():
        """
        Describe the layout of this header block on the file.
        This class doesn't inherit HeaderBase and is coded by hand.
        The implementation won't use _formats but it is included
        for the benefit of some tools.
        """
        return [
            ('_infoff',      'Q', 'int64', 'InfoHeader position in file.'),
            ('_stroff',      '',  'int64', 'String table position, N/A in V1 and pulled from InfoHeader in V2.'),
            ('_alphalupoff', 'Q', 'int64', 'Alpha tile lookup table position in file.'),
            ('_bricklupoff', 'Q', 'int64', 'Brick data lookup table position in file.'),
            ('_histoff',     'Q', 'int64', 'Histogram position in file.'),
            ]

class OffsetHeaderV2:

    def __init__(self, buf = None):
        """
        Unpack a byte buffer into a new instance of this header.
        Caller is responsible for all I/O, so we don't need an iocontext.
        """
        self._infsize = 0
        self._strsize = 0
        self._histsize = 0
        self._alphalupsize = 0
        self._bricklupsize =0
        self._infoff = -1
        self._stroff = -1
        self._histoff = -1
        self._alphalupoff = -1
        self._bricklupoff = -1
        self.calculate()

    def pack(self):
        return bytes(1)

    @classmethod
    def _format(self):
        return "<B"

    @classmethod
    def size(self):
        return struct.calcsize(self._format())

    @classmethod
    def read(cls, f):
        # This will read and discard a single byte.
        # The OffsetHeader always starts immediately after the FileHeader,
        # which always starts at the beginning of the file.
        return cls(_checked_read(f, FileHeader.headersize(), cls.size()))

    def dump(self, *, file=None):
        file = file or sys.stdout
        print("\n{0} takes up {1} bytes with format '{2}'".format(self.__class__.__name__, self.size(), self._format()), file=file)
        print(("  Offsets = info {_infoff} str {_stroff} hist {_histoff} alpha {_alphalupoff} brick {_bricklupoff}\n" +
               "  Sizes   = info {_infsize} str {_strsize} hist {_histsize} alpha {_alphalupsize} brick {_bricklupsize}" +
               "").format(**self.__dict__), file=file)

    def calculate(self, ih = None):
        """
        Calculate offsets and sizes for the various headers and tables.
        Some information requires the InfoHeader to be already known.
        If it isn't we will just calculate as much as we can.

        In general the size of a header as written to file might be
        larger than the size that the header expects to unpack.
        This allows adding more data fields at the end of the header.
        Older readers will just unpack the fields they know about.

        For ZGY V{2,3,4} this is moot, as all the offsets are implicit
        with all the headers written sequentially. So the size needs
        to match exactly or the headers following this will be corrupt.
        """
        self._infoff = FileHeader.headersize() + OffsetHeaderV2.size()
        self._infsize = InfoHeaderV2.headersize()
        if ih:
            self._strsize = ih._slbufsize
            self._histsize = HistHeaderV2.headersize()
            self._alphalupsize = 8 * ih._alphaoffsets[-1]
            self._bricklupsize = 8 * ih._brickoffsets[-1]
            self._stroff = self._infoff + self._infsize
            self._histoff = self._stroff + self._strsize
            self._alphalupoff = self._histoff + self._histsize
            self._bricklupoff = self._alphalupoff + self._alphalupsize

    @staticmethod
    def _formats():
        """
        Describe the layout of this header block on the file.
        This class doesn't inherit HeaderBase and is coded by hand.
        The implementation won't use _formats but it is included
        for the benefit of some tools.
        """
        return [
            ('_infoff',      '', 'int64', 'InfoHeader position in file.'),
            ('_stroff',      '', 'int64', 'String table position, N/A in V1 and pulled from InfoHeader in V2.'),
            ('_alphalupoff', '', 'int64', 'Alpha tile lookup table position in file.'),
            ('_bricklupoff', '', 'int64', 'Brick data lookup table position in file.'),
            ('_histoff',     '', 'int64', 'Histogram position in file.'),
            ]

class OffsetHeaderV3(OffsetHeaderV2):
    pass

class OffsetHeaderV4(OffsetHeaderV3):
    pass

def OffsetHeaderFactory(version):
    try:
        return [OffsetHeaderV1, OffsetHeaderV2, OffsetHeaderV3, OffsetHeaderV4][version-1]
    except IndexError:
        raise ZgyFormatError("Version " + str(version) + " is not supported")

#****************************************************************************
#** InfoHeader **************************************************************
#****************************************************************************

class InfoHeaderV1(HeaderBase):

    def __init__(self, buf = None):
        """
        Unpack a byte buffer into a new instance of this header.
        Caller is responsible for all I/O, so we don't need an iocontext.
        """
        assert self.headersize() == 146
        self.unpack(buf)
        self.calculate()
        if buf:
            # Check the roundtrip.
            assert buf == self.pack()

    @classmethod
    def load(self, buf):
        result = self()
        data = struct.unpack(self._format(), buf)
        #print(data)
        assert len(data) == 30
        return result

    def calculate(self, sl = None, hh = None):
        # See FileVersion<1>::InfoHeader::Interpret in the old library.
        self._bricksize = (64, 64, 64)
        try:
            if type(self._datatype) == int: self._datatype = impl_enum.RawDataType(self._datatype)
        except ValueError as ex:
            raise ZgyFormatError("Invalid enumeration found in ZGY file: " + str(ex))
        try:
            if type(self._coordtype) == int: self._coordtype = impl_enum.RawCoordType(self._coordtype)
        except ValueError as ex:
            self._coordtype = impl_enum.RawCoordType.Unknown
        self._ocp_world, self._ocp_annot, self._ocp_index = (
            _CalcOrderedCorners(self._orig, self._inc, self._size,
                                self._gpiline, self._gpxline,
                                self._gpx, self._gpy))
        self._lodsizes = _CalcLodSizes(self._size, self._bricksize)
        self._nlods = len(self._lodsizes)
        self._brickoffsets = _CalcLutOffsets(self._lodsizes, False)
        self._alphaoffsets = _CalcLutOffsets(self._lodsizes, True)
        # In V1 there is just a single value range, used both for the
        # histogram and for scaling integral values to float and for
        self._file_codingrange = (hh._min, hh._max) if hh else None
        # Sanity check. If codingrange is bad, silently use a range
        # that causes no conversion between storage and float.
        self._safe_codingrange = _fix_codingrange(self._file_codingrange, self._datatype)
        self._smin = hh._min if hh else None
        self._smax = hh._max if hh else None
        # Convert the V1 "coordtype" member to the V2 equivalent.
        if self._coordtype == impl_enum.RawCoordType.Meters:
            self._hdim = impl_enum.RawHorizontalDimension.Length
            self._hunitname = 'm'
            self._hunitfactor = 1.0
        elif self._coordtype == impl_enum.RawCoordType.Feet:
            self._hdim = impl_enum.RawHorizontalDimension.Length
            self._hunitname = 'ft'
            self._hunitfactor = 0.3048
        elif self._coordtype == impl_enum.RawCoordType.ArcSec:
            self._hdim = impl_enum.RawHorizontalDimension.ArcAngle
            self._hunitname = 'arcsec'
            self._hunitfactor = 1.0
        elif self._coordtype == impl_enum.RawCoordType.ArcDeg:
            self._hdim = impl_enum.RawHorizontalDimension.ArcAngle
            self._hunitname = 'deg'
            self._hunitfactor = 3600.0
        elif self._coordtype == impl_enum.RawCoordType.ArcDegMinSec:
            # value = deg*10000 + min*100 + sec
            # Not supported, nor does it work in the old code.
            self._hdim = impl_enum.RawHorizontalDimension.Unknown
            self._hunitname = '?'
            self._hunitfactor = 1.0
        else:
            self._hdim = impl_enum.RawHorizontalDimension.Unknown
            self._hunitname = ''
            self._hunitfactor = 1.0
        # V1 had no vertical unit information.
        self._vdim = impl_enum.RawVerticalDimension.Unknown
        self._vunitname = ''
        self._vunitfactor = 1.0

    @staticmethod
    def _formats():
        """
        Describe the layout of this header block on the file.
        The second field is the format as recognized by the 'struct' module.
        Data on the file is stored packed and little-endian. So for use with
        'struct', a '<' should be prepended to the string.
        """
        return [
            ('_size',      '3i', 'int32[3]', 'Integer size in inline, crossline, vertical directions.'),
            ('_orig',      '3i', 'int32[3]', 'First inline, crossline, time/depth. Only integral values allowed.'),
            ('_inc',       '3i', 'int32[3]', 'Integer increment in inline, crossline, vertical directions.'),
            ('_incfactor', '3f', 'float32[3]', 'Unused. Write as (1,1,1), ignore on read.'),
            ('_gpiline',   '4i', 'int32[4]', 'Inline component of 4 control points.'),
            ('_gpxline',   '4i', 'int32[4]', 'Crossline component of 4 control points.'),
            ('_gpx',       '4d', 'float64[4]', 'X coordinate of 4 control points.'),
            ('_gpy',       '4d', 'float64[4]', 'Y coordinate of 4 control points.'),
            ('_datatype',  'B', 'uint8', 'Type of samples in each brick: int8 = 0, int16 = 2, float32 = 6.'),
            ('_coordtype', 'B', 'uint8', 'Coordinate type: unknown = 0, meters = 1, feet = 2, degrees*3600 = 3, degrees = 4, DMS = 5.'),

            # Derived, as they are missing in v1 but present in v2.

            ('_bricksize', '', 'int32[3]', 'Brick size. Values other than (64,64,64) will likely not work.'),
            ('_file_codingrange', '', 'float32[2]', 'Rescaling interval used if datatype is non-float. Value range hint otherwise.'),

            # Data identifiers
            ('_dataid', '', 'uint8[16]', 'Data GUID, set on file creation.'),
            ('_verid', '', 'uint8[16]', 'Data version GUIDm set each time the file is changed.'),
            ('_previd', '', 'uint8[16]', 'Previous data version GUID.'),

            # Data source
            ('_srcname', '', 'char*', 'Source name.'),
            ('_srcdesc', '', 'char*', 'Source descriotion.'),
            ('_srctype', '', 'uint8', 'Source datatype.'),

            # Extent
            ('_curorig', '', 'int32[3]', 'Zero-based origin of extent spanned by the data currently in the file. Unused?'),
            ('_cursize', '', 'int32[3]', 'Size of extent spanned by the data currently in the file. Unused?'),

            # Statistics
            ('_scnt', '', 'int64', 'Count of values used to compute statistics.'),
            ('_ssum', '', 'float64', 'Sum of all "scnt" values.'),
            ('_sssq', '', 'float64', 'Sum of squared "scnt" values.'),
            ('_smin', '', 'float32', 'Statistical (computed) minimum value.'),
            ('_smax', '', 'float32', 'Statistical (computed) maximum value.'),

            ('_srvorig', '', 'float32[3]', 'Unused?'),
            ('_srvsize', '', 'float32[3]', 'Unused?'),

            # Grid definition
            ('_gdef', '', 'uint8', 'Grid definition type. Ignored on read.'),
            ('_gazim', '', 'float64[2]', 'Unused.'),
            ('_gbinsz', '', 'float64[2]', 'Unused.'),

            # Horizontal domain
            ('_hprjsys', '', 'char*', 'Free form description of the projection coordinate system. Usually not parseable into a well known CRS.'),
            ('_hdim', '', 'uint8', 'Horizontal dimension.'),
            ('_hunitfactor', '', 'float64', 'Multiply by this factor to convert from storage units to SI units.'),
            ('_hunitname', '', 'char*', 'For annotation only. Use the factor to convert to or from SI.'),

            # Vertical domain
            ('_vdim', '', 'uint8', 'Vertical dimension.'),
            ('_vunitfactor', '', 'float64', 'Multiply by this factor to convert from storage units to SI units.'),
            ('_vunitname', '', 'char*', 'For annotation only. Use the factor to convert to or from SI.'),

            # Derived information, both in v1 and v2.
            ('_ocp_world', '', 'enum', 'Ordered corner points: ((i0,j0),(iN,j0),(i0,jM),(iN,jM)'),
            ('_lodsizes', '', 'int32[lod]', 'Size of the survey at reduced level of detail.'),
            ('_nlods', '', 'int32', 'How many levels of details. 1 means only full resolution. Currently nlods will always be just enough to make the highest LOD (i.e. lowest resolution) fit in a single brick.'),
            ('_brickoffsets', '', 'int64[lod]', 'How many entries in the lookup table to skip when dealing with level N.'),
            ('_alphaoffsets', '', 'int64[lod]', 'How many entries in the lookup table to skip when dealing with level N.'),
        ]

class InfoHeaderV2(HeaderBase):

    def __init__(self, buf = None):
        """
        Unpack a byte buffer into a new instance of this header.
        Caller is responsible for all I/O, so we don't need an iocontext.
        """
        assert self.headersize() == 337
        self.unpack(buf)
        if buf:
            # Check the roundtrip.
            if buf != self.pack():
                print("FAIL",
                      " ".join([hex(x) for x in buf]),
                      " ".join([hex(x) for x in self.pack()]),
                      sep="\n")
            assert buf == self.pack()
        # Need to do this at end, because calculate() may patch the header.
        self.calculate()

    @staticmethod
    def _cast_enum(e, cls, default):
        """
        Convert an integer read from file to the correctly typed enum.
        The numerical values of these enums have been chosen to match
        what the file contains. If the file contains an impossible
        value then return the supplied default. It is safe to call this
        method twice. If the value is already an enum then it is returned
        unchanged.
        """
        if type(e) == int:
            try:
                return cls(e)
            except ValueError:
                return default
        else:
            return e

    def calculate(self, sl = None, hh = None):
        # Convert plain ints to enums. If they are already enums then this is a no-op.
        try:
            if type(self._datatype) == int:
                self._datatype = impl_enum.RawDataType(self._datatype)
        except ValueError as ex:
            # If the data type is not recognized the file will be unuseable.
            raise ZgyFormatError("Invalid enumeration found in ZGY file: " + str(ex))
        # The other enums on the file are not important; replace corrupt
        # values with a suitable dsefault.
        self._srctype = self._cast_enum(
            self._srctype, impl_enum.RawDataType, self._datatype)
        self._gdef    = self._cast_enum(
            self._gdef, impl_enum.RawGridDefinition, impl_enum.RawGridDefinition.Unknown)
        self._hdim    = self._cast_enum(
            self._hdim, impl_enum.RawHorizontalDimension, impl_enum.RawHorizontalDimension.Unknown)
        self._vdim    = self._cast_enum(
            self._vdim, impl_enum.RawVerticalDimension, impl_enum.RawVerticalDimension.Unknown)

        # Set the 5 strings if a StringHeader is provided, else clear them.
        # Normally we get called exactly once without a StringList, followed
        # by exactly one call that has the strings.
        self._srcname = sl._list[0] if sl else None
        self._srcdesc = sl._list[1] if sl else None
        self._hprjsys = sl._list[2] if sl else None
        self._hunitname = sl._list[3] if sl else None
        self._vunitname = sl._list[4] if sl else None

        # Geometry can be specified either as 4 ordered corner points
        # (_gdef = RawGridDefinition.FourPoint) or 3 arbitrary points
        # (_gdef = RawGridDefinition.ThreePoint) where both annotation
        # and world coordinates are given. There is also .Parametric
        # which specifies azimuth and spacing, but that is no longer
        # supported. If it ever was. With FourPoint the writer is still
        # required to store the annotation coordinates just in case
        # there is some disagreement over how the corners are ordered.
        # So, we treat FourPoint the same as ThreePoint. Ignore the last
        # point (nominally this is the corner opposite the origin)
        # and treat the remaining 3 points as if they were arbitraty.
        self._ocp_world, self._ocp_annot, self._ocp_index = (
            _CalcOrderedCorners(self._orig, self._inc, self._size,
                                self._gpiline, self._gpxline,
                                self._gpx, self._gpy))
        self._lodsizes = _CalcLodSizes(self._size, self._bricksize)
        self._nlods = len(self._lodsizes) # NOT _CalcNumberOfLODs(self._size)
        self._brickoffsets = _CalcLutOffsets(self._lodsizes, False)
        self._alphaoffsets = _CalcLutOffsets(self._lodsizes, True)
        # Sanity check. If codingrange is bad, silently use a range
        # that causes no conversion between storage and float.
        self._safe_codingrange = _fix_codingrange(self._file_codingrange, self._datatype)


    def _calculate_strings(self):
        strings = [self._srcname or '',
                   self._srcdesc or '',
                   self._hprjsys or '',
                   self._hunitname or '',
                   self._vunitname or '']
        strings = '\x00'.join(strings) + '\x00'
        strings = bytes(strings, "ASCII", errors='replace')
        return strings

    def calculate_write(self):
        """
        Call this when important parts of this struct has changed,
        Note that on write it is we that compute slbufsize that gets copied
        to offsetheader and used when the string list is written.
        Not the other way aroud.
        """
        self._lodsizes = _CalcLodSizes(self._size, self._bricksize)
        self._nlods = len(self._lodsizes)
        self._brickoffsets = _CalcLutOffsets(self._lodsizes, False)
        self._alphaoffsets = _CalcLutOffsets(self._lodsizes, True)
        self._slbufsize = len(self._calculate_strings())
        # For floating point data the coding range set by the user
        # on file creation is ignored; see _init_from_scratch().
        # on write it is set to the actual min/max data range as
        # measured in the statistics. This is very useful if the
        # file is later converted to an integral type without
        # explicitly setting the range. Also in general it is bad
        # to have an attribute which might be set incorrectly by
        # the user but is 99% unused so the consequence of setting
        # it wrong is unknown.
        # Note that overriding the codingrange is also done on read,
        # see openzgy.api.ZgyMeta.datarange.
        # TODO-Low, this does not support the case where data is written
        # containing huge spikes and where the codingrange is then
        # set to the correct range the data ought to be clipped to.
        if self._datatype == impl_enum.RawDataType.Float32:
            if self._smin <= self._smax:
                self._file_codingrange = (self._smin, self._smax)
            else: # No valid samples in file, still need a valid range
                self._file_codingrange = (-1, +1)
            self._safe_codingrange = self._file_codingrange

    @staticmethod
    def _formats():
        """
        Describe the layout of this header block on the file.
        The second field is the format as recognized by the 'struct' module.
        Data on the file is stored packed and little-endian. So for use with
        'struct', a '<' should be prepended to the string.

        In the Info header, the fields bricksize, datatype, and size
        are particularly important and immutable because their values
        affect the size of other blocks on the file. Codingrange is
        also immutable when storage type is integral. Because once
        bulk has been written, changing the codingrange would change
        all the sample values. Also, dataid is immutable by definition.

        Annotation origin and increment could technically be updated
        but this doesn't make much sense. Data may have been loaded
        indexed by annotation coordinates, and in that case changing
        the annotation would invalidate the data.
        """
        return [
            # Coding
            ('_bricksize', '3i', 'int32[3]', 'Brick size. Values other than (64,64,64) will likely not work.'),
            ('_datatype', 'B', 'uint8', 'Type of samples in each brick: int8 = 0, int16 = 2, float32 = 6.'),
            ('_file_codingrange', '2f', 'float32[2]', 'If datatype is integral, this is the value range samples will be scaled to when read as float. In this case it must be specified on file creation. If datatype is float then this is the value range of the data and should be set automatically when writing the file.'),

            # Data identifiers
            ('_dataid', '16s', 'uint8[16]', 'GUID set on file creation.'),
            ('_verid', '16s', 'uint8[16]', 'GUID set each time the file is changed.'),
            ('_previd', '16s', 'uint8[16]', 'GUID before last change.'),

            # Data source
            ('_srcname', '', 'char*', 'Optional name of this data set. Rarely used.'), # In StringList[0]
            ('_srcdesc', '', 'char*', 'Optional description of this data set. Rarely used.'), # in StringList[1]
            ('_srctype', 'B', 'uint8', 'Optional datatype the samples had before being stored in this file.'),

            # Extent
            ('_orig', '3f', 'float32[3]', 'First inline, crossline, time/depth. Unlike v1 these are now floating point.'),
            ('_inc', '3f', 'float32[3]', 'Increment in inline, crossline, vertical directions.'),
            ('_size', '3i', 'int32[3]', 'Size in inline, crossline, vertical directions.'),
            ('_curorig', '3i', 'int32[3]', 'Unused. Set to (0,0,0) on write and ignore on read.'),
            ('_cursize', '3i', 'int32[3]', 'Unused. Set to size on write and ignore on read.'),

            # Statistics. Not set directly from user code.
            ('_scnt', 'q', 'int64', 'Count of values used to compute statistics.'),
            ('_ssum', 'd', 'float64', 'Sum of all "scnt" values.'),
            ('_sssq', 'd', 'float64', 'Sum of squared "scnt" values.'),
            ('_smin', 'f', 'float32', 'Statistical (computed) minimum value.'),
            ('_smax', 'f', 'float32', 'Statistical (computed) maximum value.'),

            # Survey extent. Ignore on read.
            ('_srvorig', '3f', 'float32[3]', 'Unused. Set equal to orig on write. Ignore on read.'),
            ('_srvsize', '3f', 'float32[3]', 'Unused. Set to inc*size on write. Ignore on read.'),

            # Grid definition
            ('_gdef', 'B', 'uint8', 'Grid definition type. Set to 3 (enum: "FourPoint") on write. Ignored on read. See notes for a longer explanation.'),
            ('_gazim', '2d', 'float64[2]', 'Unused.'),
            ('_gbinsz', '2d', 'float64[2]', 'Unused.'),
            ('_gpiline', '4f', 'float32[4]', 'Inline component of 4 control points.'),
            ('_gpxline', '4f', 'float32[4]', 'Crossline component of 4 control points.'),
            ('_gpx', '4d', 'float64[4]', 'X coordinate of 4 control points.'),
            ('_gpy', '4d', 'float64[4]', 'Y coordinate of 4 control points.'),

            # Horizontal domain
            ('_hprjsys', '', 'char*', 'Free form description of the projection coordinate system. Usually not parseable into a well known CRS.'), # in StringList[2]
            ('_hdim', 'B', 'uint8', 'Horizontal dimension. Unknown = 0, Length = 1, ArcAngle = 2. Few applications support ArcAngle.'),
            ('_hunitfactor', 'd', 'float64', 'Multiply by this factor to convert from storage units to SI units. Applies to gpx, gpy.'),
            ('_hunitname', '', 'char*', 'For annotation only. Use hunitfactor, not the name, to convert to or from SI.'), # in StringList[3]

            # Vertical domain
            ('_vdim', 'B', 'uint8', 'Vertical dimension. Unknown = 0, Depth = 1, SeismicTWT = 1, SeismicOWT = 3.'),
            ('_vunitfactor', 'd', 'float64', 'Multiply by this factor to convert from storage units to SI units. Applies to orig[2], inc[2].'),
            ('_vunitname', '', 'char*', 'For annotation only. Use vunitfactor, not the name, to convert to or from SI.'), # in StringList[4]

            # Miscellaneous. Not set directly from user code.
            ('_slbufsize', 'I', 'uint32', 'Size of the StringList section.'),

            # Derived information, not stored in the file.
            ('_ocp_world', '', 'enum', 'Ordered corner points: ((i0,j0),(iN,j0),(i0,jM),(iN,jM)'),
            ('_lodsizes', '', 'int32[lod]', 'Size of the survey at reduced level of detail.'),
            ('_nlods', '', 'int32', 'How many levels of details. 1 means only full resolution. Currently nlods will always be just enough to make the highest LOD (i.e. lowest resolution) fit in a single brick.'),
            ('_brickoffsets', '', 'int64[lod]', 'How many entries in the lookup table to skip when dealing with level N.'),
            ('_alphaoffsets', '', 'int64[lod]', 'How many entries in the lookup table to skip when dealing with level N.'),
        ]

class InfoHeaderV3(InfoHeaderV2):
    pass

class InfoHeaderV4(InfoHeaderV3):
    pass

def InfoHeaderFactory(version):
    try:
        return [InfoHeaderV1, InfoHeaderV2, InfoHeaderV3, InfoHeaderV4][version-1]
    except IndexError:
        raise ZgyFormatError("Version " + str(version) + " is not supported")

#****************************************************************************
#** StringList **************************************************************
#****************************************************************************

class StringListV1:
    """
    ZGY V1 had no string list, so this section is empty.
    """
    def __init__(self, di):
        pass

    @classmethod
    def _format(cls):
        return ""

    @classmethod
    def size(cls, oh = None, ih = None):
        return 0

    @classmethod
    def load(cls, di, buf):
        return cls()

    @classmethod
    def read(cls, f, oh, ih):
        return cls(None)

    def dump(self, *, file=None):
        file = file or sys.stdout
        print("\n{0} takes up {1} bytes with format '{2}'".format(self.__class__.__name__, self.size(), self._format()), file=file)

class StringListV2:
    """
    The string list holds 5 null terminated strings:
    srcname, srcdesc, hprjsys, hunit.name, vunit.name.
    The reader will ignore trailing garbage. This is
    the one place we might add additional information
    without breaking existing readers.
    """
    def __init__(self, buf = None, oh = None, ih = None):
        """
        Unpack a byte buffer into a new instance of this header.
        Caller is responsible for all I/O, so we don't need an iocontext.
        """
        self._oh = oh
        self._ih = ih
        if buf:
            buf = buf.split(b'\x00')
            self._list = list([e.decode("ASCII", errors='replace') for e in buf])
        else:
            self._list = ['','','','','']

    @classmethod
    def _format(cls):
        return ""

    @classmethod
    def size(cls, oh = None, ih = None):
        """
        On read the size is stored in the offsetheader,
        which may have gotten it from the infoheader.
        On write the size depends on the strings written
        and we might not be able to trust the offsetheader.
        """
        # TODO-Worry handle write, especially the case where one of the strings
        # have been updated. Or isn't this needed because we don't allow
        # updating these?
        return oh._strsize if oh else 0

    @classmethod
    def read(cls, f, oh, ih):
        if oh._strsize > 0:
            return cls(_checked_read(f, oh._stroff, oh._strsize), oh, ih)
        else:
            return cls()

    def dump(self, *, file=None):
        file = file or sys.stdout
        print("\n{0} takes up {1} bytes with format '{2}'".format(self.__class__.__name__, self.size(self._oh, self._ih), self._format()), file=file)
        print("    " + "\n    ".join(self._list), file=file)

class StringListV3(StringListV2):
    pass

class StringListV4(StringListV3):
    pass

def StringListFactory(version):
    try:
        return [StringListV1, StringListV2, StringListV3, StringListV4][version-1]
    except IndexError:
        raise ZgyFormatError("Version " + str(version) + " is not supported")

#****************************************************************************
#** HistHeader **************************************************************
#****************************************************************************

class HistHeaderV1(HeaderBase):

    def __init__(self, buf = None):
        """
        Unpack a byte buffer into a new instance of this header.
        Caller is responsible for all I/O, so we don't need an iocontext.
        """
        assert self.headersize() == 256*4 + 8
        self.unpack(buf)
        #self.calculate()
        self._cnt = np.sum(self._bin) # In V1 there is no explicit all-count.
        if buf:
            # Check the roundtrip.
            assert buf == self.pack()

    @staticmethod
    def _formats():
        """
        Describe the layout of this header block on the file.
        The second field is the format as recognized by the 'struct' module.
        Data on the file is stored packed and little-endian. So for use with
        'struct', a '<' should be prepended to the string.
        """
        return [
            ('_max', 'f', 'float32', 'Center point of first bin.'),
            ('_min', 'f', 'float32', 'Center point of last bin.'),
            ('_bin', '256I', 'uint32[256]', 'Histogram.'),
        ]

class HistHeaderV2(HeaderBase):

    def __init__(self, buf = None):
        """
        Unpack a byte buffer into a new instance of this header.
        Caller is responsible for all I/O, so we don't need an iocontext.
        """
        assert self.headersize() == 256*8 + 16
        self.unpack(buf)
        #self.calculate()
        if buf:
            # Check the roundtrip.
            assert buf == self.pack()

    @staticmethod
    def _formats():
        """
        Describe the layout of this header block on the file.
        The second field is the format as recognized by the 'struct' module.
        Data on the file is stored packed and little-endian. So for use with
        'struct', a '<' should be prepended to the string.
        """
        return [
            ('_cnt', 'q', 'int64', 'Total number of samples.'),
            ('_min', 'f', 'float32', 'Center point of first bin.'),
            ('_max', 'f', 'float32', 'Center point of last bin.'),
            ('_bin', '256q', 'int64[256]', 'Histogram.'),
        ]

class HistHeaderV3(HistHeaderV2):
    pass

class HistHeaderV4(HistHeaderV3):
    pass

def HistHeaderFactory(version):
    try:
        return [HistHeaderV1, HistHeaderV2, HistHeaderV3, HistHeaderV4][version-1]
    except IndexError:
        raise ZgyFormatError("Version " + str(version) + " is not supported")

#****************************************************************************
#** Alpha lookup table ******************************************************
#****************************************************************************

class LookupTable:
    """
    Both the Alpha lookup table and the Brick lookup table hold a 64-bit
    file offset for each tile or brick in the file. Alpha tiles are bitmaps
    used to flag dead traces, and only have (i,j) coordinates. Bricks
    contain the actual samples and are indexed with (i, j, k).

    The size of the lookup tables depend on the survey size. The first
    entry in the lookup table is for the brick or tile (always just one)
    holding the lowest resolution. This is immediately followed by one
    or more entries for the bricks or tiles at level of detail N-1, and
    so on until the entries for lod 0. Within one lod level the first
    entry is for the lowest numbered i,j,[k]. For subsequent entries the
    i numbers vary fastest and the k (or j in the alpha case) varies
    slowest. Note that this is somewhat non intuitive as it is the
    opposite of the ordering of samples within a tile.

    In version 1 of the lookup tables the file offsets are stored in a
    somewhat quirky manner. The high 32 bits and the low 32 bits are
    both stored as little-endian integers, but the high part is stored
    first. So it is part big-endian, part little-endian.

    An offset of 0 means the corresponding brick or tile does not exist.
    An offset of 1 means the brick or tile contains all zeros and does
    not take any space on the file. An offset with the most significant
    bit set also means the brick or tile has a constant value. In this
    case the actual value is encoded in the least significant 8/16/32
    bits (depending on valuetype) of the stored offset.
    Offsets 0x8000000000000000 and 0x0000000000000001 are equivalent.
    Actually, v2 and later uses the first form while v1 used the second.
    For robustness both forms should be accepted regardless of version.
    """

    def __init__(self, buf, lupsize, mustflip):
        """
        Unpack a byte buffer into a new instance of this header.
        Caller is responsible for all I/O, so we don't need an iocontext.
        """
        self._mustflip = mustflip
        self._lupsize = lupsize
        if buf:
            self._lookup = list(struct.unpack(self._format(lupsize), buf))
            if self._mustflip: self._lookup = self._flip_array(self._lookup)
            self._lookend = self._calc_lookupsize(self._lookup, None, None)
        else:
            self._lookup = [0] * (lupsize//8)
            self._lookend = [0] * (lupsize//8)

    def pack(self):
        if self._mustflip: self._lookup = self._flip_array(self._lookup)
        result = struct.pack(self._format(len(self._lookup)*8), *self._lookup)
        if self._mustflip: self._lookup = self._flip_array(self._lookup)
        return result

    @staticmethod
    def _calc_lookupsize(lookup, eof, maxsize):
        """
        Given an index => start_offset lookup table, produce an
        index => end_offset table by assuming there are no holes
        in the allocated data.

        The function understands constant-value and compressed blocks.

        If eof and maxsize are known, the code can also make the
        following checks:

        Blocks that have a start offset > eof are unreadable and
        should be ignored. Set them to start and end at eof.
        The same applies to offsets of unknown type i.e. the most
        significant bit is 1 but the most significant byte is
        neither 0x80 (constant) nor 0xC0 (compressed).
        Blocks ending past eof should be assumed to end at eof.

        Blocks that appear to be larger than an uncompressed block are
        probably too large. This may be caused by holes in the allocated
        data. Assume the block is the same size as an uncompressed block.
        If a compressed block takes up more room than an uncompressed one
        then the writer should simply refrain from compressing it.
        But for extra robustness the code that makes use of this
        information shoukd be prepared to retry the access of the block
        really turned out to be larger.

        This method might be called unconditionally on file open, or
        called only if at least one compressed brick was found, or it
        might be deferred until the first time we read a compressed brick.

        TODO-Low: If alpha tiles are present then both brick and alpha offsets
        ought to be considered in the same pass. The way it will work now
        is that for bricks, a few bricks will appear too large because
        they are followed by some alpha tiles. This is harmless.
        For aplha tiles the end offsets will be hopelessly wrong.
        We will need to just assume 4 KB for those.
        """
        #print("@@@ _calc_lookupsize", list(map(hex, lookup[:5])))
        # make array of (offset, endoffset, type, ordinal)
        # Note, the only reason I use a structured array instead of a
        # 2d array [][4] is to get sort() to work the way I want it.
        # Is there a simpler way? Or is this just as performant?
        dtype = [('offset', np.uint64),
                 ('endpos', np.uint64),
                 ('type', np.uint64),
                 ('ordinal', np.uint64)]
        tmp = np.zeros(len(lookup), dtype=dtype)
        codeshift = np.uint64(56)
        codemask = np.uint64(0xFF) << codeshift
        tmp['offset'] = lookup
        tmp['ordinal'] = np.arange(len(lookup), dtype=np.uint64)
        tmp['type'] = tmp['offset']; tmp['type'] &= codemask
        tmp['offset'][tmp['type'] == (np.uint64(0x80) << codeshift)] = 0
        tmp['offset'][tmp['type'] == (np.uint64(0xC0) << codeshift)] &= ~(codemask)
        tmp.sort(order='offset')
        # The end of block i is the start of block i+1,
        # except the last block which ends at EOF, just use a huge number.
        tmp[:-1]['endpos'] = tmp[1:]['offset']
        tmp[-1]['endpos'] = eof or ~(np.uint64(1)<<np.uint64(63))
        tmp.sort(order='ordinal')
        # Several entries may be flagged as starting at offset 0.
        # With the above algorithm, an arbitrary one of these will
        # have its end set to the start of the first real block.
        # The rest will appear to end at zero. Get rid of this
        # odd behavior.
        tmp['endpos'][tmp['offset']==0] = 0
        if eof is not None and maxsize is not None:
            #endpos = max(offset, min(endpos, eof, offset + maxsize))
            np.minimum(tmp['endpos'], eof, out=tmp['endpos'])
            np.minimum(tmp['endpos'], tmp['offset'] + maxsize, out=tmp['endpos'])
            np.maximum(tmp['endpos'], tmp['offset'], out=tmp['endpos'])
            # This test as written also catches unrecognized block types
            # because the offsets are unsigned and all the unrecognized
            # types will have the most significant bit set.
        #print("@@@ _calc_lookupsize", tmp[:5])
        # Returning a numpy array of np.int64 is problematic because
        # _lookup is a normal Python array, and arithmetic such as
        # int(foo) - np.int64(bar) returns a Python float.
        #return tmp['endpos'].copy()
        return list(map(int, tmp['endpos']))

    @classmethod
    def _flip(cls, n):
        return ((n >> 32) & 0xFFFFFFFF) | ((n & 0xFFFFFFFF) << 32)

    @classmethod
    def _flip_array(cls, a):
        return list([cls._flip(n) for n in a])

    @classmethod
    def _format(cls, lupsize):
        fmt = "<" + str(lupsize//8) + "Q"
        assert struct.calcsize(fmt) == lupsize
        return fmt

    @classmethod
    def read(cls, f, offset, lupsize):
        return cls(_checked_read(f, offset, lupsize), lupsize)

    def dump(self, prefix = "", prefix0 = "", *, file=None):
        file = file or sys.stdout
        print("\n{0}{1} takes up {2} bytes with format '{3}'".format(
            prefix0,
            self.__class__.__name__,
            self._lupsize,
            self._format(self._lupsize)), file=file)
        print(prefix+("\n"+prefix).join(map(hex, self._lookup)), file=file)

class AlphaLUPV1(LookupTable):
    def __init__(self, buf, lupsize):
        super().__init__(buf, lupsize, mustflip = True)

class AlphaLUPV2(LookupTable):
    def __init__(self, buf, lupsize):
        super().__init__(buf, lupsize, mustflip = False)

class AlphaLUPV3(AlphaLUPV2):
    pass

class AlphaLUPV4(AlphaLUPV3):
    pass

def AlphaLUPFactory(version):
    try:
        return [AlphaLUPV1, AlphaLUPV2, AlphaLUPV3, AlphaLUPV4][version-1]
    except IndexError:
        raise ZgyFormatError("Version " + str(version) + " is not supported")

#****************************************************************************
#** Brick lookup table ******************************************************
#****************************************************************************

class BrickLUPV1(LookupTable):
    def __init__(self, buf, lupsize):
        super().__init__(buf, lupsize, mustflip = True)

class BrickLUPV2(LookupTable):
    def __init__(self, buf, lupsize):
        super().__init__(buf, lupsize, mustflip = False)

class BrickLUPV3(BrickLUPV2):
    pass

class BrickLUPV4(BrickLUPV3):
    pass

def BrickLUPFactory(version):
    try:
        return [BrickLUPV1, BrickLUPV2, BrickLUPV3, BrickLUPV4][version-1]
    except IndexError:
        raise ZgyFormatError("Version " + str(version) + " is not supported")

#****************************************************************************
#** All headers combined in a single instance *******************************
#****************************************************************************

class ZgyInternalMeta:
    """
    Holds references to all the individual headers needed to access ZGY.
    The information is stored in the instance in a format that is tightly
    coupled to the file format, so there will be one layer (but hopefully
    just one) above us that is responsible for the public API.
    """
    def __init__(self, myfile):
        self._is_bad = False
        if myfile: # when reading
            self._init_from_open_file(myfile)
        else: # when writing
            self._init_from_headers(None, None, None, None, None, None, None)

    def _assert_all_headers_allocated(self):
        """
        The following asserts might seem paranoid, but for a long time
        there was an assupption that every single access to those headers
        had to check first whether they were None. I am just proving to
        myself that all those checks can be removed.
        """
        assert self._fh is not None
        assert self._oh is not None
        assert self._ih is not None
        assert self._sl is not None
        assert self._hh is not None
        assert self._alup is not None
        assert self._blup is not None

    def _init_from_headers(self, fh, oh, ih, sl, hh, alup, blup):
        self._fh = fh
        self._oh = oh
        self._ih = ih
        self._sl = sl
        self._hh = hh
        self._alup = alup
        self._blup = blup

    def _init_from_open_file(self, f):
        """
        Read all the headers and save pointers to each of them in this instance.
        Some care is needed to read them in the correct order, as there are
        several dependencies between them. The next version of the file format
        will hopefully be a bit simpler in this respect.
        """

        # The file header contains only a magic string and the file version.
        # In V{2,3,4} the headers are stored consecutively:
        #     FileHeader OffsetHeader InfoHeader StringList HistHeader
        #     AlphaLUT BrickLUT
        # In V1 and most likely in the upcoming V4 only FileHeader and
        # OffsetHeader are known to be consecutive.
        fh = FileHeader.read(f, 0)

        # The offset header immediately follows the file header.
        # Changed in v2 and v3: Offset header is no longer used
        # (all offsets are now implicit), but due to a quirk in the
        # implementation it still occupies one byte on the file.
        # This is actually a bug waiting to happen, because that
        # byte (which is the size of a class with no members) is
        # probably compiler dependant.
        # Removing the OffsetHeader actually made the files tricker
        # to read, as the size of some sections depend on contents
        # of other sections.
        oh = OffsetHeaderFactory(fh._version).read(f)

        # 'oh' at this point may be incomplete (V{2,3,4}) but the offset
        # to the InfoHeader should be known by now. In V1 is is mostly
        # complete but is missing a few section sizes.
        ih = InfoHeaderFactory(fh._version).read(f, oh._infoff)

        # For V{2,3,4}, fill in the rest of the offsets now that
        # the InfoHeader is known.
        oh.calculate(ih)

        # Variable length strings are stored in a separate header.
        # Most (currently all) of these logically belong to InfoHeader.
        sl = StringListFactory(fh._version).read(f, oh, ih)

        hh = HistHeaderFactory(fh._version).read(f, oh._histoff)

        # For V2 and later, fill in the missing strings in InfoHeader
        # now that the StringList is known.
        # For V1 we also need to copy the value range (used for scaling)
        # from the histogram header.
        ih.calculate(sl, hh)

        alup = AlphaLUPFactory(fh._version).read(f, oh._alphalupoff, oh._alphalupsize)
        blup = BrickLUPFactory(fh._version).read(f, oh._bricklupoff, oh._bricklupsize)

        self._init_from_headers(fh, oh, ih, sl, hh, alup, blup)

    def _init_from_scratch(self, filename, *, size = None, compressed = False,
                           bricksize = None,
                           datatype = impl_enum.RawDataType.Float32,
                           datarange = None,
                           zunitdim = impl_enum.RawVerticalDimension.Unknown,
                           hunitdim = impl_enum.RawHorizontalDimension.Unknown,
                           zunitname = None, hunitname = None,
                           zunitfactor = None, hunitfactor = None,
                           zstart = 0.0, zinc = 0.0,
                           annotstart = (0, 0), annotinc = (0, 0),
                           corners = ((0,0),(0,0),(0,0),(0,0))):

        #--- Sanity checks and defaults management ---#

        if not size:
            raise ZgyUserError("size must be specified.")
        elif any([s<1 for s in size]):
            raise ZgyUserError("size must be at least 1 in each dimension.")

        if not bricksize:
            bricksize = (64, 64, 64)
        elif len(bricksize) != 3:
            raise ZgyUserError("bricksize must be specified in 3 dimensions.")
        elif any([(s<4 or not self._is_power_of_two(s)) for s in bricksize]):
            raise ZgyUserError("bricksize must be >= 4 and a power of 2.")

        # The codingrange for floating point data is special. The user is
        # not allowed to set it, and its value is not used internally.
        # To ensure that any user supplied range is really ignored we set
        # the range to NaN. In _calculate_write it will be set to the
        # statistical range before being written out. As an additional
        # bullet-proofing, to avoid surprises with older files, this
        # rule can also be enforced in api.ZgyMeta.datarange.
        # Note: For integral types I might have defaulted the datarange
        # to no conversion (-128..+127 or -32768..+32767) and also
        # silently re-ordered min and max if needed. But in both cases
        # the application is buggy. So, make a fuss.
        # A data range for integral data covering just a single value
        # (presumably the user will just write that single value to
        # the file) is also forbidden because it just isn't useful
        # and it triggers several corner cases.
        if datatype == impl_enum.RawDataType.Float32:
            datarange = (math.nan, math.nan)
        elif not datarange or len(datarange) != 2:
            raise ZgyUserError("datarange must be specified for integral types.")
        elif datarange[0] >= datarange[1]:
            raise ZgyUserError("datarange must have min < max.")
        elif not np.isfinite(datarange[0]) or not np.isfinite(datarange[1]):
            raise ZgyUserError("datarange must be finite.")

        zunitname = zunitname or ""
        hunitname = hunitname or ""
        zunitfactor = zunitfactor or 1
        hunitfactor = hunitfactor or 1

        #--- End sanity checks and defaults --#

        fh = FileHeader(buf = None, compressed = compressed) # Inits to version 3, or 4 if potentially compressed.
        oh = OffsetHeaderFactory(fh._version)() # Sets infoff, infsize only.
        ih = InfoHeaderFactory(fh._version)() # Inits to all zero

        # Fill in the parts of InfoHeader that affects headers elsewhere.
        ih._bricksize = tuple(bricksize) or (64, 64, 64)
        ih._datatype = datatype
        ih._size = tuple(size)
        ih._slbufsize = 0

        # Meta information caller is allowed to specify.
        # Naming is not 100% consistent; this is because
        # the parameters to this function wree set to match
        # the existing Python wrapper for the old ZGY.

        ih._file_codingrange = (datarange[0], datarange[1])
        ih._safe_codingrange = ih._file_codingrange
        ih._vdim = zunitdim
        ih._hdim = hunitdim
        ih._vunitname = zunitname
        ih._hunitname = hunitname
        ih._vunitfactor = zunitfactor
        ih._hunitfactor = hunitfactor
        ih._orig = (annotstart[0], annotstart[1], zstart)
        ih._inc = (annotinc[0], annotinc[1], zinc)
        ih._ocp_world = ((corners[0][0], corners[0][1]),
                         (corners[1][0], corners[1][1]),
                         (corners[2][0], corners[2][1]),
                         (corners[3][0], corners[3][1]))
        ih._gpx = (corners[0][0], corners[1][0], corners[2][0], corners[3][0])
        ih._gpy = (corners[0][1], corners[1][1], corners[2][1], corners[3][1])
        beg = (ih._orig[0], ih._orig[1])
        end = (ih._orig[0] + ih._inc[0] * (ih._size[0] - 1),
               ih._orig[1] + ih._inc[1] * (ih._size[1] - 1))
        ih._gpiline = (beg[0], end[0], beg[0], end[0])
        ih._gpxline = (beg[1], beg[1], end[1], end[1])

        # Meta information that might be updated after creation.
        # Except for dataid.
        def makeUUID():
            # See the C++ version for details.
            # TODO-Worry: Is the entropy of the random seed good enough?
            uuid = bytearray([random.randint(0,255) for i in range(16)])
            uuid[8] = (uuid[8] & 0x3f) | 0x80 # variant 1 (DCE)
            uuid[7] = (uuid[7] & 0x0f) | 0x40 # version 4 (random)
            return uuid
        ih._dataid = makeUUID()
        ih._verid = makeUUID()
        ih._previd = bytes(16)
        ih._srcname = ""
        ih._srcdesc = ""
        ih._srctype = datatype
        ih._hprjsys = ""

        ih._scnt = 0
        ih._ssum = 0.0
        ih._sssq = 0.0
        ih._smin = 0.0
        ih._smax = 0.0

        # Unused fields, required to be set this way for compatibility.
        ih._curorig = (0, 0, 0)
        ih._cursize = ih._size
        ih._srvorig = ih._orig
        ih._srvsize = (ih._size[0] * ih._inc[0], ih._size[1] * ih._inc[1], ih._size[2] * ih._inc[2])
        ih._gdef = impl_enum.RawGridDefinition.FourPoint
        ih._gazim = (0.0, 0.0)
        ih._gbinsz = (0.0, 0.0)

        # Derived information, see calculate()
        ih.calculate_write()

        # Fill in the rest of the offsets now that the InfoHeader is known.
        # Requires _slbufsize, ih._alphaoffsets, _brickoffsets
        oh.calculate(ih)

        # TODO-Low: Refactor: a more elegant way of handling this?
        sl = StringListFactory(fh._version)(ih._calculate_strings(), oh, ih)

        # Histogram gets initialized to empty
        hh = HistHeaderFactory(fh._version)()

        # Lookup tables get initialized to empty
        alup = AlphaLUPFactory(fh._version)(None, oh._alphalupsize)
        blup = BrickLUPFactory(fh._version)(None, oh._bricklupsize)

        self._init_from_headers(fh, oh, ih, sl, hh, alup, blup)

        # If "assert contig" in _flush_meta fails, this is also wrong:

        # ZGY aligns data to the basic block size, which depends on
        # data type and bricksize. This simplifies caching data.
        # If all headers are written sequentially from the start
        # of the file then it is simpler to add the padding as part
        # of the header data instead of before the first data block.
        # Forget about using leftover space in the header to store the
        # first few alpha tiles. We probably won't be writing those anyway.
        code = impl_enum._map_DataTypeToStructFormatCode(self._ih._datatype)
        bs = self._ih._bricksize
        bs = bs[0] * bs[1] * bs[2] * struct.calcsize(code)
        hdrsize = self._oh._bricklupoff + self._oh._bricklupsize
        padsize = (((hdrsize+bs-1)//bs)*bs)-hdrsize

        self._data_beg = hdrsize + padsize
        self._data_end = hdrsize + padsize

    def _flush_meta(self, f):

        # Make sure derived information is up to date and consistent.
        self._ih.calculate_write()
        self._oh.calculate(self._ih)

        # fileheader, infoheader, histheader all derive from HeaderBase
        # so they inherit pack() and unpack() from there. alup and blup
        # inherit pack() from LookupTable.
        fh   = self._fh.pack()
        oh   = self._oh.pack()
        ih   = self._ih.pack()
        sl   = self._ih._calculate_strings()
        hh   = self._hh.pack()
        alup = self._alup.pack()
        blup = self._blup.pack()

        assert len(ih) == self._oh._infsize
        assert len(sl) == self._oh._strsize
        assert len(hh) == self._oh._histsize
        assert len(alup) == self._oh._alphalupsize
        assert len(blup) == self._oh._bricklupsize

        contig = (self._oh._infoff      == len(fh) + len(oh) and
                  self._oh._stroff      == self._oh._infoff + len(ih) and
                  self._oh._histoff     == self._oh._stroff + len(sl) and
                  self._oh._alphalupoff == self._oh._histoff + len(hh) and
                  self._oh._bricklupoff == self._oh._alphalupoff + len(alup))

        # ZGY aligns data to the basic block size, which depends on
        # data type and bricksize. This simplifies caching data.
        # If all headers are written sequentially from the start
        # of the file then it is simpler to add the padding as part
        # of the header data instead of before the first data block.
        # Forget about using leftover space in the header to store the
        # first few alpha tiles. We probably won't be writing those anyway.
        pad = bytes()
        self._first_data = None
        if contig:
            code = impl_enum._map_DataTypeToStructFormatCode(self._ih._datatype)
            bs = self._ih._bricksize
            bs = bs[0] * bs[1] * bs[2] * struct.calcsize(code)
            hdrsize = self._oh._bricklupoff + self._oh._bricklupsize
            padsize = (((hdrsize+bs-1)//bs)*bs)-hdrsize
            pad = bytes(padsize)
            self._first_data = hdrsize + padsize

        # If not contiguous then just write the blocks one at a time.
        # For V{2,3,4} this will never happen. So I won't add code
        # for that case (which I will never be able to test).
        assert contig
        with ErrorsWillCorruptFile(self):
            f.xx_write(fh + oh + ih + sl + hh + alup + blup + pad,
                       0,
                       usagehint=impl_file.UsageHint.Header)

    def dumpRaw(self, *, file=None):
        self._fh.dump(file=file)
        self._oh.dump(file=file)
        self._ih.dump('    ', file=file)
        self._sl.dump(file=file)
        self._hh.dump('    ', file=file)
        self._alup.dump('   ', file=file)
        self._blup.dump('   ', file=file)

    @staticmethod
    def _is_power_of_two(n):
        """Not efficient, but will only be called a couple of times."""
        for shift in range(0, 32):
            if n == 1<<shift:
                return True
        return False

#****************************************************************************
#** Free functions **********************************************************
#****************************************************************************

def _CalcLodSizes(size, bricksize):
    """
    Compute the size of the file in bricks, for all LOD levels
    and all 3 dimensions. Indirectly also compute the number
    of LOD levels, since by definition the last LOD level
    is the one that has size (1, 1, 1) i.e. the entire cube
    is decimated enought to fit inside a single brick.
    """
    if min(bricksize) < 1: return [(0, 0, 0)]
    size = ((size[0] + bricksize[0] - 1) // bricksize[0],
            (size[1] + bricksize[1] - 1) // bricksize[1],
            (size[2] + bricksize[2] - 1) // bricksize[2])
    result = [ size ]
    while max(result[-1]) > 1:
        size = ((size[0]+1)//2, (size[1]+1)//2, (size[2]+1)//2)
        result.append(size)
    #print("## CalcLodSize", result)
    return tuple(result)

def _CalcLutOffsets(lods, isalpha):
    """
    Compute the offset into the lookup tables by LOD level.
    Return an array of offsets indexed by LOD. Also return
    (appended to the end of the result) the lookuop table size.
    The result differs depending on whether this is the alpha
    or brick LUT.
    The highest LOD is at the start of each table so it by
    definition has offset 0. Since the highest LOD level
    just has a single brick, the second highest will start
    at offset 1.
    """
    result = []
    pos = 0
    for e in reversed(lods):
        result.append(pos)
        pos = pos + e[0]*e[1]*(1 if isalpha else e[2])
    result.reverse()
    result.append(pos)
    #print("## CalcLutOffsets", result)
    return tuple(result)

def _CalcNumberOfLODs(size):
    """
    For illustration only. Use len(_CalcLodSize(...)) instead.
    This method suffers from numerical accuracy issues.
    """
    assert False
    return math.ceil(math.log(max(size))/math.log(2)) + 1 if size and max(size) >= 1 else 0

def _CalcOrderedCorners(orig, inc, size, gpiline, gpxline, gpx, gpy):
    """
    Convert three arbitrary control points to 4 ordered corner points ordered
    first il, first xl; last il, first xl, first il, last xl, last il, last xl.
    Also return the same 4 corners in annotation- and ordinal coordinates.
    The last two are redundant with size and annot but are provided as a
    convenience for the api layer.

    If the file contains world corners but no annotation, assume the writer
    used RawGridDefinition.FourPoint but without storing the apparently
    redundant annotation corners. This is contrary to spec but we will handle
    that case for robustness.

    If the conversion fails for another reason then return None because
    in that case coordinate conversion will not be possible.
    """
    no_world = math.isclose(gpx[0], gpx[1]) and math.isclose(gpy[0], gpy[1])
    no_annot = math.isclose(gpiline[0], gpiline[1]) and math.isclose(gpxline[0], gpxline[1])
    if no_annot and not no_world:
        ocp_world = list([[gpx[i], gpy[i]] for i in range(4)])
    else:
        try:
            ocp_world = acpToOcp(orig, inc, size, gpiline, gpxline, gpx, gpy)
        except RuntimeError:
            #raise ZgyFormatError("Invalid coordinates in ZGY file.")
            ocp_world = None
            #ocp_world = list([[gpx[i], gpy[i]] for i in range(4)])
    ends = (orig[0] + inc[0] * (size[0]-1),
            orig[1] + inc[1] * (size[1]-1))
    ocp_annot = ((orig[0], orig[1]),
                 (ends[0], orig[1]),
                 (orig[0], ends[1]),
                 (ends[0], ends[1]))
    ocp_index = ((0, 0),
                 (size[0]-1, 0),
                 (0, size[1]-1),
                 (size[0]-1, size[1]-1))
    # Make immutable so it is safe to return to application code.
    if ocp_world is not None: ocp_world = tuple(map(tuple, ocp_world))
    return (ocp_world, ocp_annot, ocp_index)

def checkAllFormats(*, verbose = False, file = None):
    FileHeader.checkformats(verbose=verbose, file=file)
    InfoHeaderV1.checkformats(verbose=verbose, file=file)
    InfoHeaderV2.checkformats(verbose=verbose, file=file)
    InfoHeaderV3.checkformats(verbose=verbose, file=file)
    InfoHeaderV4.checkformats(verbose=verbose, file=file)
    HistHeaderV1.checkformats(verbose=verbose, file=file)
    HistHeaderV2.checkformats(verbose=verbose, file=file)
    HistHeaderV3.checkformats(verbose=verbose, file=file)
    HistHeaderV4.checkformats(verbose=verbose, file=file)

# For now, run the consistency checks also when the module is imported.
checkAllFormats()

if __name__ == "__main__":
    pass
    #me = InfoHeaderV1()
    #me.dump("    ")
    #print(me.pack())

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
