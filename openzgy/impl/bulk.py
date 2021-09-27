#!/usr/bin/env python3

"""
This file contains a number of classes and free functions dealing with
bulk data access. There is a similar file impl.meta that deals with
metadata. Nothing in this file should be directly visible to users of
the public API.

    impl.bulk.ZgyInternalBulk:

        * Internal methods to read and write bulk.
        * Needs access to much of the meta data information.
          Currently this is handled by giving this class references to
          those file headers it needs, e.g. InfoHeader, BrickLUP, etc.
          TODO-Low: Refactor, would it be cleaner to instead provide a
          reference to the ZgyInternalMeta instance? This gives access to
          more metadata than the class actually needs. But would provide
          better isolation nevertheless.
        * Refactoring notes: There are a couple of sections in this
          class that ought to have been separated out. As it is now
          the class does too much.
            - Write support
            - Lod generation
            - Histogram generation

"""

##@package openzgy.impl.bulk
# \brief Reading and writing bulk data.

import numpy as np
import struct
import os
import math
import sys

from ..impl import enum as impl_enum
from ..impl import file as impl_file
from ..exception import *
from ..impl.lodalgo import decimate, DecimationType
from ..impl.stats import StatisticData
from ..impl.compress import CompressFactoryImpl

# Changing this should be done ONLY FOR TESTING.
# Use a different value for 'outside survey area' than the default.
_padding_fill_value = None

def dprint(*args, **kwargs):
    return None # comment out to get debug logging
    #print(*args, **kwargs)

class ErrorsWillCorruptFile:
    """
    Duplicated between impl.bulk and impl.meta. Maybe fix sometime.
    Technically the two differ because they set different flags.
    In ZgyInternalBulk and ZgyInternalMeta respectively. But that
    distinction isn't easy to see in Python.

    Start a critical section where any exception means that the
    owner class should be permanently flagged with _is_bad = True.
    Typically this is used to prevent secondary errors after a
    write failure that has most likely corrupted the entire file.
    The exception itself will not be caught.

    The _is_bad flag normally means that any further attempts
    to access this class, at least for writing, will raise a
    ZgyCorruptedFile exception. Regardless of what the exception
    was that caused the flag to be set.
    """
    def __init__(self, parent): self._parent = parent
    def __enter__(self): return None
    def __exit__(self, type, value, traceback):
        if type:
            #print("Bulk: Exit critical section", str(value))
            self._parent._is_bad = True

class ScalarBuffer:
    """
    Represents an ndarray where all elements have the same value.
    This is basically a scalar but additionally has a .shape
    attribute telling how large an array it represents.
    """
    def __init__(self, shape, value, dtype):
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)
        self._value = self._dtype.type(value)
    @property
    def shape(self): return self._shape
    @property
    def dtype(self): return self._dtype
    @property
    def value(self): return self._value
    @property
    def itemsize(self): return self._dtype.itemsize
    def __len__(self): return 0
    def __str__(self): return str(self._value)
    def __repr__(self): return "numpy.{0}({1})".format(self._dtype.name, self._value)
    def inflate(self): return np.full(self._shape, self._value, dtype=self._dtype)

class ZgyInternalBulk:
    """
    Read or write bulk data. The meta data needs to have been read
    already. The user-callable API will forward its read requests here.
    """
    def __init__(self, f, metadata, *, compressed = False):
        self._is_bad = False
        self._file = f
        # Note, ZgyInternalBulk._metadata and ZgyMeta._meta will point to
        # the same instance of ZgyInternalMeta. The members deliberately
        # have different names; I believe this will to reduce confusion.
        self._metadata = metadata
        self._compressed = compressed # If true, do not align bricks.
        # Technically only needed if self._metadata._fh._version == 1
        # but do it unconditionally to get better test coverage.
        self._unscramble_init()
        # Keep track of value range as LOD 0 samples are written.
        # If samples are overwritten the range might be too wide.
        # This range will be used when computing the histogram.
        # The range is always user values, not storage. This migh
        # be of academic interest only, as the information is
        # currently ignored when the data on the file is integral.
        self._sample_min = np.inf
        self._sample_max = -np.inf
        # TODO-Low some way for application code to configure this.
        # For now don't check for cloud/local because uncompressed
        # local files are reasonable to have as Always, especially
        # since it is so backward to change.
        #self._update_mode = impl_enum.UpdateMode.Always if not compressed and not f.xx_iscloud else impl_enum.UpdateMode.Constant
        self._update_mode = impl_enum.UpdateMode.Always if not compressed else impl_enum.UpdateMode.Constant

    @staticmethod
    def _argType(parameter, accept, name = "parameter"):
        """
        Check that the parameter has one of the acceped types.
        If None is allowed, include type(None) in the accepted list.
        Raise an exception on error.
        The main reason I am using this function is that I changed the
        convention for passing data buffers around. This used to be
        either a np.ndarray or a scalar, with the latter often requiring
        a "size" parameter as well. The new convention is to pass either
        np.ndarray of ScalarBuffer. Given Python's weak typing this
        might easily introduce some bugs.
        """
        if not isinstance(parameter, accept):
            if not isinstance(accept, tuple): accept = (accept,)
            oknames = tuple([t.__name__ for t in accept])
            myname = type(parameter).__module__ + "." + type(parameter).__name__
            err = "{0} must be in {1} but was {2}".format(name, oknames, myname)
            raise TypeError(err)

    def _checkCompatibleBufferTypes(self, user_dtype, is_storage):
        """
        Several methods now have both an explicit is_storage argument
        and a return-buffer value type that can be used to infer whether
        the buffer or scalar needs to be converted from a float 'user'
        value to an integer 'storage' values. The is_storage parameter
        has precedence but to help trap bugs the value type must match.
        """
        file_dtype = impl_enum._map_DataTypeToNumpyType(self._metadata._ih._datatype)
        if is_storage and user_dtype != file_dtype:
            raise ZgyInternalError("Wrong data type " + str(data.dtype) +
                                   " for a file with " + str(file_dtype))
        elif not is_storage and user_dtype != np.float32:
            raise ZgyInternalError("Wrong data type " + str(data.dtype) +
                                   ", when is_storage=False it should be float")

    @staticmethod
    def _slice(beg, end):
        """For debugging output only."""
        return "[{0}:{3},{1}:{4},{2}:{5}]".format(*beg, *end)
        #return "[{0}:{3},{1}:{4},{2}:{5}] (size {6},{7},{8})".format(*beg, *end, *(end-beg))

    @classmethod
    def _getOverlap(cls, src_beg, src_count, dst_beg, dst_count, survey_beg, survey_count):
        """
        Determine the overlap between two or three N-dimensional regions.
        Typically used to compare what the user wanted with the content
        of one particular 3d data block, and possibly also with the
        declared total area of the survey. With the intent of copying
        only the overlapping area. Return the offsets relative to both
        the source and the target region.
        """
        src_beg = np.array(src_beg, dtype=np.int64)
        src_end = src_beg + np.array(src_count, dtype=np.int64)
        dst_beg = np.array(dst_beg, dtype=np.int64)
        dst_end = dst_beg + np.array(dst_count, dtype=np.int64)
        # Get the overlap, relative to cube origin, between the brick
        # we just read (src) and the area that the user requested (dst).
        overlap_beg = np.maximum(src_beg, dst_beg)
        overlap_end = np.minimum(src_end, dst_end)
        # Optionally clip to the survey size as well.
        if survey_beg is not None and survey_count is not None:
            survey_beg = np.array(survey_beg, dtype=np.int64)
            survey_end = survey_beg + np.array(survey_count, dtype=np.int64)
            overlap_beg = np.maximum(overlap_beg, survey_beg)
            overlap_end = np.minimum(overlap_end, survey_end)
        #print("// src relative to cube:", cls._slice(src_beg, src_end))
        #print("// dst relative to cube:", cls._slice(dst_beg, dst_end))
        #print("// overlap to be copied:", cls._slice(overlap_beg, overlap_end))
        # Now convert from offsets relative to the cube orign
        # to offsets into the two buffers.
        local_src_beg = overlap_beg - src_beg
        local_src_end = overlap_end - src_beg
        local_dst_beg = overlap_beg - dst_beg
        local_dst_end = overlap_end - dst_beg
        #print("// src relative to buffer:", cls._slice(local_src_beg, local_src_end))
        #print("// dst relative to buffer:", cls._slice(local_dst_beg, local_dst_end))
        return local_src_beg, local_src_end, local_dst_beg, local_dst_end

    @classmethod
    def _partialCopy(cls, src, src_beg, src_count, dst, dst_beg, dst_count, survey_beg, survey_count, *, verbose = None):
        """
        Copy bulk data from src to dst, but only in the overlapping region.
        The src buffer will typically be a single brick read from file.
        Src can also be a scalar, in which case the overlapping region
        will have all its samples set to this particular value.
        The method will typically be called several times, once for each
        brick that was read. Both src and dest need to be either numpy
        arrays or (in the case of src) a scalar. Or a ScalarBuffer.
        """
        if isinstance(src, (np.ndarray, ScalarBuffer)): assert tuple(src.shape) == tuple(src_count)
        if isinstance(dst, np.ndarray): assert tuple(dst.shape) == tuple(dst_count)
        dst_count = dst.shape
        (local_src_beg, local_src_end,
         local_dst_beg, local_dst_end) = cls._getOverlap(
             src_beg, src_count, dst_beg, dst_count, survey_beg, survey_count)
        if np.all(local_src_end > local_src_beg):
            if isinstance(src, np.ndarray):
                if verbose:
                    verbose("set result",
                          cls._slice(local_dst_beg, local_dst_end), "= brick",
                          cls._slice(local_src_beg, local_src_end))
                dst[local_dst_beg[0]:local_dst_end[0],
                    local_dst_beg[1]:local_dst_end[1],
                    local_dst_beg[2]:local_dst_end[2]] = (
                src[local_src_beg[0]:local_src_end[0],
                    local_src_beg[1]:local_src_end[1],
                    local_src_beg[2]:local_src_end[2]])
            else:
                if verbose:
                    verbose("set result",
                          cls._slice(local_dst_beg, local_dst_end),
                          "= constant", src)
                value = src.value if isinstance(src, ScalarBuffer) else src
                dst[local_dst_beg[0]:local_dst_end[0],
                    local_dst_beg[1]:local_dst_end[1],
                    local_dst_beg[2]:local_dst_end[2]] = value
        else:
            if verbose:
                verbose("no ovelap with this brick")

    @staticmethod
    def _scaleFactorsStorageToFloat(codingrange, file_dtype):
        """
        Get the linear transform y = a*x + b for converting from
        storage values to actual floating point values.
        The smallest value in storage (e.g. -128) should map to codingrange[0]
        and the largest value (e.g. +127) should map to codingrange[1].
        If file_dtype is a floating point type there is never any conversion.
        """
        if not np.issubdtype(file_dtype, np.integer):
            (a, b) = (1, 0)
        else:
            iinfo = np.iinfo(file_dtype)
            a = (codingrange[1] - codingrange[0]) / (iinfo.max - iinfo.min)
            b = codingrange[0] - a * iinfo.min
        return (a, b)

    def _scaleDataFactorsStorageToFloat(self):
        ih = self._metadata._ih
        file_dtype = impl_enum._map_DataTypeToNumpyType(ih._datatype)
        return self._scaleFactorsStorageToFloat(ih._safe_codingrange, file_dtype)

    @classmethod
    def _scaleToFloat(cls, data, codingrange, file_dtype):
        """
        Scale integral data from storage according to the coding range.
        Input can be either a scalar or a numpy array.
        The operation is a no-op if file_dtype is float.
        The data must be known to be in storage domain, we don't try
        to guess based on its valuetype.
        """
        if data is None: return None
        # The compression module might not like the following assert
        # because for compressed integral data the decompressor might
        # convert from integral to float but leave the scaling for
        # later. Currently not a problem.
        if isinstance(data, (ScalarBuffer, np.ndarray)): assert file_dtype == data.dtype
        if not np.issubdtype(file_dtype, np.integer): return data
        (a, b) = cls._scaleFactorsStorageToFloat(codingrange, file_dtype)
        if isinstance(data, np.ndarray):
            # Numerical accuracy notes:
            #
            # To match the (arguably broken) old ZGY implementation
            # do all the computations, including the codingrange to
            # (slope, intercept) calculation, using single precision
            # float. The result isn't seriously wrong but values
            # close to zero may see a noticeable shift.
            #
            # To get the result as accurate as possible, convert the
            # entire array to float64, then apply the conversion,
            # then convert back to float32. The compromise currently
            # used is to accurately compute slope, intercept but do
            # the scaling on float32 values. The difference compared
            # to the most accurate case is hardly noticeable.
            #
            # See also InfoHeaderAccess::storagetofloat() in OpenZGY/C++.
            #
            data = data.astype(np.float32)
            # In-place operators to avoid copying arrays again.
            data *= a
            data += b
            return data
        elif isinstance(data, ScalarBuffer):
            return ScalarBuffer(data.shape, a * float(data.value) + b, np.float32)
        else:
            return np.float32(a * float(data) + b)

    def _scaleDataToFloat(self, data):
        """
        Convert and/or scale the storage value type to user-supplied data.
        Unlike _scaleToFloat() this is an instance method so it already
        knows the storage type. The data must be known to be in storage
        domain, we don't try to guess based on its valuetype.
        """
        file_dtype = impl_enum._map_DataTypeToNumpyType(self._metadata._ih._datatype)
        return self._scaleToFloat(data, self._metadata._ih._safe_codingrange, file_dtype)

    @staticmethod
    def _clipAndCast(data, dtype):
        """
        Cast to the specified numpy type, clipping any values outside
        the valid range to the closest edge. Works both for scalars
        and numpy arrays. Can be made to support lists and tuples also,
        but this is a case of yagni.
        If the data is a scalar then the result will be of the reqested
        numpy type and not the generic python 'int'.
        """
        cliprange = (np.iinfo(dtype).min, np.iinfo(dtype).max)
        if isinstance(data, np.ndarray):
            np.clip(data, cliprange[0], cliprange[1], out=data) # in-place clip
            # Can I save a few cycles by passing dtype to np.rint and
            # dropping the call to astype? Not a big deal and the
            # documentation is a bit unclear.
            data = np.rint(data).astype(dtype)
        elif isinstance(data, ScalarBuffer):
            value = np.rint(max(cliprange[0], min(cliprange[1], data.value)))
            data = ScalarBuffer(data.shape, value, dtype)
        else:
            data = np.rint(max(cliprange[0], min(cliprange[1], data)))
            data = dtype(data)
        return data

    @classmethod
    def _scaleToStorage(cls, data, codingrange, file_dtype):
        """
        Scale floating point data to integral storage values according
        to the coding range. Input can be either a scalar or a numpy array.
        Or the new ScalarBuffer type.
        Input values that cannot be encoded will be clipped to the closest
        value that can be represented.
        The operation is a no-op if file_dtype is float and not an integral type.
        The input data must be np.float32 or an array of same; this is to
        avoid accidentally passing an input that is already in storage.
        """
        # Test: unittests.test_scaleToStorage()
        vt = data.dtype if isinstance(data, (np.ndarray, ScalarBuffer)) else type(data)
        if vt != np.float32:
            raise ZgyInternalError("Input to _scaleToStorage must be np.float32")
        if not np.issubdtype(file_dtype, np.integer): return data
        (a, b) = cls._scaleFactorsStorageToFloat(codingrange, file_dtype)
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32)
            # In-place operators to avoid copying arrays again.
            data -= b
            data *= (1.0/a)
        elif isinstance(data, ScalarBuffer):
            value = (float(data.value) - b) / a
            data = ScalarBuffer(data.shape, value, np.float32)
        else:
            data = (float(data) - b) / a
        return cls._clipAndCast(data, file_dtype)

    def _defaultValue(self, as_float):
        """
        Get the "real" or "storage" value to be used when samples that
        have never been written. as_float=False means the return will be
        a "storage" value as found in the file. Otherwise convert to "real".
        If the data is a scalar then the result will be of the reqested
        numpy type and not the generic python 'int'.
        TODO-Performance, compute this when the file is opened.
        """
        defaultstorage = self._scaleDataToStorage(np.float32(0))
        defaultreal = self._scaleDataToFloat(defaultstorage)
        return defaultreal if as_float else defaultstorage

    @staticmethod
    def _bricksNeeded(start, size, bricksize):
        """
        Return the list of bricks needed to cover the entire area given by
        start and size. Each entry in the list gives both the sample position
        and the brick position, each as a 3-tuple of integers.
        """
        brick0 = [start[0] // bricksize[0],
                  start[1] // bricksize[1],
                  start[2] // bricksize[2]]
        brickN = [(start[0] + size[0] - 1) // bricksize[0],
                  (start[1] + size[1] - 1) // bricksize[1],
                  (start[2] + size[2] - 1) // bricksize[2]]
        result = []
        for ii in range(brick0[0], brickN[0]+1):
            for jj in range(brick0[1], brickN[1]+1):
                for kk in range(brick0[2], brickN[2]+1):
                    result.append(((ii * bricksize[0],
                                    jj * bricksize[1],
                                    kk * bricksize[2]),
                                   (ii, jj, kk)))
        return result

    def _validatePosition(self, i, j, k, lod):
        if lod < 0 or lod >= self._metadata._ih._nlods:
            raise ZgyUserError("Requested LOD {0} is outside the valid range {1} to {2} inclusive".format(lod, 0, self._metadata._ih._nlods - 1))
        size = self._metadata._ih._lodsizes[lod]
        if (i<0 or i>=size[0] or j<0 or j>=size[1] or k<0 or k>=size[2]):
            raise ZgyUserError("Requested brick position {0} is outside the valid range {1} to {2} inclusive at lod {3}".format((i, j, k), (0, 0, 0), tuple(np.array(size) - 1), lod))

    def _getBrickLookupIndex(self, i, j, k, lod):
        self._validatePosition(i, j, k, lod)
        size = self._metadata._ih._lodsizes[lod]
        index = (self._metadata._ih._brickoffsets[lod] +
  	         i + (size[0] * j) + (size[0] * size[1] * k))
        if index < 0 or index >= self._metadata._ih._brickoffsets[-1]:
            raise ZgyInternalError("Internal error in _getBrickLookupIndex")
        return index

    def _getAlphaSizeInBytes(self):
        return int(self._metadata._ih._bricksize[0] * self._metadata._ih._bricksize[1])

    def _getBrickSizeInBytes(self):
        """
        Get the size of an uncompressed brick in bytes.
        TODO-Performance, this should be cached on file open and
        should probably be a derived attribute of self._metadata._ih.
        NOTE-Performance: np.product() might be preferable to spelling out
        the multiply and needing a temp. But it could be 100 times slower.
        """
        file_dtype = np.dtype(impl_enum._map_DataTypeToNumpyType(self._metadata._ih._datatype))
        bs = self._metadata._ih._bricksize
        maxsize = bs[0] * bs[1] * bs[2] * file_dtype.itemsize
        return int(maxsize)

    def _getBegAndSize(self, lup, ix, maxsize):
        """
        Use only for bricks known to be compressed. Unless testing.
        Return both the file offset of the compressed block and its size.
        The latter is a hint that should not be fully trusted.
        For testing pass sane = False, this skips the test for too large
        bricks etc. which means I can use the tool to find holes in the
        allocated data.
        """
        # TODO-Low consider deferring _calc_lookupsize() until first needed.
        # The simple version below id NOT THREADSAFE.
        #if self._metadata._blup._lookend is None:
        #    self._metadata._blup._lookend = (
        #        self._metadata._blup._calc_lookupsize(self._lookup, eof, maxsize))
        raw_beg = lup._lookup[ix]
        raw_end = lup._lookend[ix]
        btype = (raw_beg >> 56) & 0xFF
        if btype == 0x80 or raw_beg < 2:
            return 0, 0
        beg = raw_beg & ~(0xFF << 56)
        end = raw_end
        # Need extra tests if _calc_lookupsize() was called without
        # giving  the eof and maxsize parameters.
        end = max(beg, min(end, self._file.xx_eof, beg + maxsize))
        return (beg, end - beg)

    def _getAlphaBegAndSize(self, ix, *, sane = True):
        return self._getBegAndSize(self._metadata._alup, ix, self._getAlphaSizeInBytes() if sane else (0xFF<<56))

    def _getBrickBegAndSize(self, ix, *, sane = True):
        return self._getBegAndSize(self._metadata._blup, ix, self._getBrickSizeInBytes() if sane else (0xFF<<56))

    def _getBrickFilePosition(self, i, j, k, lod):
        """
        returns (brickstatus, fileoffset, constvalue, bricksize).
        """
        ix = self._getBrickLookupIndex(i, j, k, lod)
        pos = self._metadata._blup._lookup[ix]
        btype = (pos >> 56) & 0xFF
        if pos == 0:
            return (impl_enum.BrickStatus.Missing, None, 0, 0)
        elif pos == 1:
            return (impl_enum.BrickStatus.Constant, None, 0, 0)
        elif btype == 0x80:
            # TODO-Worry probably won't work with big-endian.
            # I suspect the old C++ reader won't either,
            # but that is an academic question since we
            # only support that library for x86.
            fmt = impl_enum._map_DataTypeToStructFormatCode(self._metadata._ih._datatype)
            tmp = struct.pack("<Q", pos)
            constant = struct.unpack_from(fmt, tmp)[0]
            return (impl_enum.BrickStatus.Constant, None, constant, 0)
        elif btype == 0xC0:
            beg, size = self._getBrickBegAndSize(ix)
            return (impl_enum.BrickStatus.Compressed, beg, None, size)
        elif pos & (1<<63):
            raise ZgyFormatError("Unknown brick type " + str(btype))
        else:
            return (impl_enum.BrickStatus.Normal, pos, None, self._getBrickSizeInBytes())

    def _setBrickFilePosition(self, i, j, k, lod, brickstatus, content, bricksize):
        ix = self._getBrickLookupIndex(i, j, k, lod)
        if brickstatus == impl_enum.BrickStatus.Missing:
            self._metadata._blup._lookup[ix] = 0
            self._metadata._blup._lookend[ix] = 0
        elif brickstatus == impl_enum.BrickStatus.Constant:
            fmt = impl_enum._map_DataTypeToStructFormatCode(self._metadata._ih._datatype)
            #print("_setBrickFilePosition: constant", type(content), content)
            tmp = struct.pack(fmt, content)
            tmp += bytes(8 - len(tmp))
            content = struct.unpack_from("<Q", tmp)[0]
            content |= (1<<63)
            self._metadata._blup._lookup[ix] = content
            self._metadata._blup._lookend[ix] = 0
        elif brickstatus == impl_enum.BrickStatus.Compressed:
            self._metadata._blup._lookend[ix] = content + bricksize
            content &= ~(0xFF << 56)
            content |= (0xC0 << 56)
            self._metadata._blup._lookup[ix] = content
        else:
            self._metadata._blup._lookup[ix] = content
            self._metadata._blup._lookend[ix] = content + bricksize

    def _getAlphaLookupIndex(self, i, j, lod):
        self._validatePosition(i, j, 0, lod)
        size = self._metadata._ih._lodsizes[lod]
        index = self._metadata._ih._alphaoffsets[lod] + i + (size[0] * j)
        if index < 0 or index >= self._metadata._ih._alphaoffsets[-1]:
            raise ZgyInternalError("Internal error in _getAlphaLookupIndex")
        return index

    def _getAlphaFilePosition(self, i, j, lod):
        # TODO-Low: Compression, same handling as in _getBrickFilePosition()
        pos = self._metadata._alup._lookup[self._getAlphaLookupIndex(i, j, lod)]
        if pos == 0:
            return (impl_enum.BrickStatus.Missing, None, 0)
        elif pos == 1:
            return (impl_enum.BrickStatus.Constant, None, 0)
        elif pos & (1<<63):
            constant = pos & 0xff
            return (impl_enum.BrickStatus.Constant, None, constant)
        else:
            return (impl_enum.BrickStatus.Normal, pos, None)

    def _unscramble_init(self):
        """
        Create static tables to convert between a regular 3d brick (v2, v3)
        and a brick that has 8x8 subtiling in the horizontal direction (v1).
        """
        self._srctodst = np.zeros(64*64*64, dtype=np.int32)
        self._dsttosrc = np.zeros(64*64*64, dtype=np.int32)
        pos = 0
        for ii in range(0, 64, 8):
            for jj in range(0, 64, 8):
                for subi in range(ii, ii+8):
                    for subj in range(jj, jj+8):
                        for subk in range(64):
                            src_pos = pos
                            pos += 1
                            dst_pos = subi*64*64 + subj*64 + subk
                            self._srctodst[src_pos] = dst_pos
                            self._dsttosrc[dst_pos] = src_pos

    def _unscramble(self, brick):
        return brick.flat[self._srctodst].reshape(brick.shape)

    def readConstantValue(self, start, size, lod = 0, as_float = True, *, verbose = None):
        """
        Check to see if the specified region is known to have all
        samples set to the same value. A return value that is not None
        signifies that a regular read would return all samples set to
        that value. A return value of None means we don't know. This
        method is only intended as a hint to improve performance.
        """
        defaultstorage = self._defaultValue(as_float=False)
        needed = self._bricksNeeded(start, size, self._metadata._ih._bricksize)
        bricks = [((e[0],) + self._getBrickFilePosition(*e[1], lod)) for e in needed]
        result = None
        for startpos, brickstatus, fileoffset, constvalue, bricksize in bricks:
            if brickstatus == impl_enum.BrickStatus.Constant:
                if result is not None and result != constvalue and not (np.isnan(result) and np.isnan(constvalue)):
                    return None
                result = constvalue
            elif brickstatus == impl_enum.BrickStatus.Missing:
                if result is not None and result != defaultstorage:
                    return None
                result = defaultstorage
            else:
                return None
        if as_float:
            result = self._scaleDataToFloat(result)
        return result

    def _deliverOneBrick(self, result, start, startpos, raw, brickstatus, as_float, *, verbose = None):
        """
        This is the final step in readToExistingBuffer(). The data has
        been read from storage, so now it needs to be copied back to
        the user. This function may be invoked multiple times if data
        was needed from more than one brick.
        Arguments:
          result   -- user's buffer which was passed to readToExistingBuffer()
          start    -- user's supplied position as a 3-tuple of index values
          startpos -- position of the start of this brick
          raw      -- bulk data for this brick, scalar or bytes or ...
          brickstatus -- normal, compressed, constant, ...
        This low level function deals with bytes (or similar) and scalars
        an input. Not the np.ndarray and ScalarBuffer used elsewhere.
        """
        self._checkCompatibleBufferTypes(result.dtype, not as_float)
        file_dtype = impl_enum._map_DataTypeToNumpyType(self._metadata._ih._datatype)
        # TODO-Low: Refactor: this is probably NOT the right place.
        if brickstatus == impl_enum.BrickStatus.Compressed:
            # Note that when the file contains compressed integral data
            # and the application asks for float data back, the decompressor
            # is responsible for the int to float conversion but NOT the
            # scaling from storage values to real values.
            onebrick = CompressFactoryImpl.decompress(raw, brickstatus, self._metadata._ih._bricksize, file_dtype, result.dtype)
            if onebrick is None:
                raise ZgyFormatError("Compression type not recognized")
        elif brickstatus == impl_enum.BrickStatus.Normal:
            if len(raw) != np.product(self._metadata._ih._bricksize) * np.dtype(file_dtype).itemsize:
                raise ZgyFormatError("Got wrong count when reading brick.")
            onebrick = np.frombuffer(raw, dtype=file_dtype)
            # Instead of describing the array as explicitly little-endian
            # I will immediately byteswap it to become native-endian.
            if np.dtype('<i4') != np.int32: # big-endian
                onebrick.byteswap(inplace=True)
            onebrick = onebrick.reshape(self._metadata._ih._bricksize, order="C")
            # Only needed for normal bricks. No compressed bricks exist in v1.
            if self._metadata._fh._version == 1:
                onebrick = self._unscramble(onebrick)
        else:
            _ = float(raw) # just to assert it is in fact a scalar.
            onebrick = ScalarBuffer(self._metadata._ih._bricksize, raw, file_dtype)
        if as_float and file_dtype != np.float32:
            onebrick = self._scaleDataToFloat(onebrick)
        #print(onebrick)

        # Note, we might pass in the survey range as well to force
        # all padding bytes to be set to the default value. Less
        # surprises for the caller. It may look a bit odd if the user
        # does a flood-fill of the entire survey to a given value and
        # later sees that the content is different in the padding area.
        # But, the caller should ignore the padding.
        #
        # On write the padding samples should also be forced to contain
        # the same value. If nothing else, to help compression. But for
        # efficiency reasons the value is not specified. Typically it
        # will be the default "absent" value but if the rest of the
        # brick has a const value and no allocated disk space then
        # they will inherit that constant.
        self._partialCopy(onebrick,
                          startpos, self._metadata._ih._bricksize,
                          result, start, result.shape,
                          (0, 0, 0), self._metadata._ih._size,
                          verbose=verbose)

    def readToExistingBuffer(self, result, start, lod, as_float, *, verbose = None, zeroed_result=False):
        """
        Read bulk data starting at "start" in index space and store the
        result in the provided 3d numpy array. Start should be in the range
        (0,0,0) to Size-1. The count of samples to read is implied by the
        size of the provided result array that is passed in. The valid data
        types for the result array are float32 (in which case samples stored
        as int8 or int16 will be scaled) or the files's storage value type
        (in which case there is no scaling). It is valid to pass a count
        that includes the padding area between the survey and the end
        of the current brick, but not past that point.
        """
        self._argType(result, (np.ndarray,))
        self._checkCompatibleBufferTypes(result.dtype, not as_float)
        file_dtype = impl_enum._map_DataTypeToNumpyType(self._metadata._ih._datatype)
        result_dtype = result.dtype
        count = result.shape

        # Need a default value to use when trying to read a brick that
        # was never written, or to fill in a brick that was only partly
        # written. To avoid non intuitive behavior the same value should
        # be used for both cases, and it should be a value that would
        # also be allowed to store as a regular sample. Use the value
        # that becomes 0.0 after conversion to float. Or as close as
        # possible to that value if the coding range is not zero centric.
        # Always use zero for floating point data. (Not NaN...)
        # Dead traces don't have any special handling apart from the
        # alpha flag. They still contain whatever data was written to them.
        defaultstorage = self._defaultValue(as_float=False)
        defaultvalue = self._defaultValue(as_float=True)

        # Make a separate pass to gather all the bricks we need to read.
        # FUTURE: we might fetch some of them in parallel and we might be
        # able to combine bricks to read larger blocks at a time. Both changes
        # can have a dramatic performance impact effect on cloud access.
        needed = self._bricksNeeded(start, count, self._metadata._ih._bricksize)
        bricks = [((e[0],) + self._getBrickFilePosition(*e[1], lod)) for e in needed]
        #print(bricks)

        #print("Default storage", defaultstorage, " and value", defaultvalue)
        if not zeroed_result:
            result.fill(defaultvalue if _padding_fill_value is None else _padding_fill_value)
        # After all bricks have been processed, the padding past the
        # end if the survey might still not have been touched. Just in
        # case the request did in fact include such samples we will
        # initialize the entire result buffer to the default value.
        if not zeroed_result:
            result.fill(_padding_fill_value if _padding_fill_value is not None else
                        defaultvalue if result.dtype == np.float32 else
                        defaultstorage)
        bricksize_bytes = np.prod(self._metadata._ih._bricksize) * file_dtype().itemsize
        requests = []
        for startpos, brickstatus, fileoffset, constvalue, real_bricksize in bricks:
            # Functor that accepts a raw "data" brick and copies it
            # into the correct place in result. Note that "startpos" and
            # "brickstatus" need to be captured for the current loop
            # iteration. Other captured variables should remain the same.
            deliverance = lambda data, startpos=startpos, brickstatus=brickstatus: self._deliverOneBrick(result, start, startpos, data, brickstatus, as_float=as_float, verbose=verbose)
            if brickstatus == impl_enum.BrickStatus.Constant:
                if verbose:
                    verbose(" Reading brick at {0} as constant {1}".format(
                        startpos, constvalue))
                deliverance(constvalue)
            elif brickstatus == impl_enum.BrickStatus.Missing:
                if verbose:
                    verbose(" Reading brick at {0} not found, use {1}".format(
                        startpos, defaultstorage))
                deliverance(defaultstorage)
            elif brickstatus == impl_enum.BrickStatus.Normal:
                if verbose:
                    verbose("Reading brick at {0} from file offset {1}".format(
                        startpos, hex(fileoffset)))
                requests.append((fileoffset, bricksize_bytes, deliverance))
            elif brickstatus == impl_enum.BrickStatus.Compressed:
                if verbose:
                    verbose("Reading compressed brick at {0} from file offset {1} size {2}".format(
                        startpos, hex(fileoffset), hex(real_bricksize)))
                # TODO-Worry obscure corner case, might need to re-try if we didn't get enough data.
                requests.append((fileoffset, real_bricksize, deliverance))
            else:
                raise ZgyInternalError("Internal error, bad brick status")
        # Send all pending read requests to the backend in one operation.
        if requests:
            self._file.xx_readv(requests,
                                parallel_ok=False,
                                immutable_ok=False,
                                transient_ok=True,
                                usagehint=impl_file.UsageHint.Data)

    # --- WRITE SUPPORT --- #
    def _scaleDataToStorage(self, data):
        """
        Convert and/or scale the user-supplied data to the storage value type.
        Unlike _scaleToStorage() this is an instance method so it already
        knows the storage type. Also be more strict about the input type.
        The method accepts np.ndarray or ScalarBuffer only.
        """
        file_dtype = impl_enum._map_DataTypeToNumpyType(self._metadata._ih._datatype)
        data_dtype = data.dtype

        if data_dtype != np.float32:
            raise ZgyInternalError("Input to _scaleDataToStorage must be np.float32, not " + str(data_dtype))
        if file_dtype != np.float32:
            data = self._scaleToStorage(data, self._metadata._ih._safe_codingrange, file_dtype)
        return data

    @classmethod
    def _setPaddingToEdge(cls, data, used, modulo, dim):
        """
        Pad unused parts of the data buffer by replicating the last samples,
        but only up to a multiple of 'modulo' samples. Handles just one
        dimension, so caller will typically invoke us three times.
        """
        cls._argType(data, (np.ndarray,))
        beg = used[dim]
        end = min(((used[dim]+modulo-1) // modulo) * modulo, data.shape[dim])
        if beg >= end: pass
        elif dim == 0: data[beg:end,:,:] = data[beg-1:beg,:,:]
        elif dim == 1: data[:,beg:end,:] = data[:,beg-1:beg,:]
        elif dim == 2: data[:,:,beg:end] = data[:,:,beg-1:beg]

    @classmethod
    def _setPaddingToConst(cls, data, used, missingvalue, dim):
        """
        Pad unused parts of the data buffer with a constant.
        Handles just one dimension, so caller should invoke us three times.
        """
        cls._argType(data, (np.ndarray,))
        beg = used[dim]
        end = data.shape[dim]
        if beg >= end: pass
        elif dim == 0: data[beg:end,:,:] = missingvalue
        elif dim == 1: data[:,beg:end,:] = missingvalue
        elif dim == 2: data[:,:,beg:end] = missingvalue

    @classmethod
    def _setPaddingSamples(cls, data, used, missingvalue, compressor):
        """
        Make sure the contants of the padding area, if any, is
        deterministic instead of whatever garbage the buffer holds.
        The method is allowed to update the buffer in place,
        but may also allocate and return a new one.
        """
        cls._argType(data, (np.ndarray,))

        if tuple(used) == tuple(data.shape):
            return data

        # TODO-Low let the compressor choose the optimal padding.
        # ZFP: replicate up to modulo-4, storage-zero outside.
        #if compressor: return compressor.pad(data, used)

        for dim in range(3):
            cls._setPaddingToConst(data, used, missingvalue, dim)

        # To reduce LOD edge artifacts, pad 'edge' up to even size.
        # To optimize ZFP compression, pad 'edge' up to modulo-4.
        for dim in range(3):
            cls._setPaddingToEdge(data, used, 4, dim)

        return data

    def _writeWithRetry(self, rawdata, brickstatus, fileoffset, brickpos, lod, *, verbose = None):
        dprint("@@ _writeWithRetry(pos={0}, count={1}, fileoffset={2:x})".format(brickpos, len(rawdata), fileoffset or self._file.xx_eof))
        if fileoffset is not None:
            try:
                self._file.xx_write(rawdata, fileoffset, usagehint=impl_file.UsageHint.Data)
            except ZgySegmentIsClosed:
                # The update mode doesn't need to be checked again here
                # unless we want to support UpdateMode.NoLeaks which
                # would require raising an exception.
                # TODO-Test: testing will require careful thought.
                fileoffset = None # Write a new brick, abandoning the old one.
        if fileoffset is None:
            # Write a new brick.
            fileoffset = self._file.xx_eof
            self._file.xx_write(rawdata, fileoffset, usagehint=impl_file.UsageHint.Data)
            self._setBrickFilePosition(brickpos[0], brickpos[1], brickpos[2],
                                       lod, brickstatus,
                                       fileoffset, len(rawdata))

    def _writeOneNormalBrick(self, data, fileoffset, brickpos, lod, compressor, *, verbose = None):
        """
        Handle padding of area outside the survey and optionally compression.
        Also convert from NumPy array to plain bytes.

        TODO-Performance

        This function is a candidate for running multi threaded due to the
        potentially expensive compression. While the next step, actually
        writing the file, needs to be serialized both because it may want
        to write at EOF and because even if the file offset is explicitly
        known the lower level write might not be thread safe. Also when
        serializing the order of bricks should be preserved. Otherwise
        performance on read might suffer.
        """
        self._argType(data, (np.ndarray,))
        dprint("@@ _writeOneNormalBrick(pos={0}, count={1})".format(brickpos, data.shape))
        # TODO-Medium: if scalar(data): return data, impl_enum.BrickStatus.Constant
        # Make it possible for this function to be called for all data
        # in _writeOneBrick(). Then move it even further up to
        # _writeAlignedRegion() which will hopefully enable some
        # parallelization on write.
        assert tuple(data.shape) == tuple(self._metadata._ih._bricksize)
        used = self._usedPartOfBrick(data.size, brickpos, lod)
        data = self._setPaddingSamples(data, used,
                                       missingvalue=self._defaultValue(as_float=False),
                                       compressor=compressor)
        # TODO-Low: Compression, might also want to check for all-constant
        # by calling _isUsedPartOfBrickAllConstant() and if true, return
        # (data.flat[0], impl_enum.BrickStatus.Constant) and do not attempt
        # to compress. This function would then be called from
        # _writeAlignedRegion() instead of _writeWithRetry().
        # There is a slight amount of added work because if the data block
        # already exists on file (which we don't know yet) the test would
        # not be needed.
        # With that change both _writeOneNormalBrick() and _writeOneBrick()
        # will need to know the current brick status, and _writeOneNormalBrick()
        # would return .Constant for constant data.
        brickstatus = impl_enum.BrickStatus.Normal
        rawdata = None
        if compressor:
            cdata = compressor(data)
            if cdata:
                rawdata = cdata
                brickstatus = impl_enum.BrickStatus.Compressed
        if rawdata is not None:
            pass
        elif np.dtype('<i4') != np.int32: # big-endian, TODO-Worry not tested.
            rawdata = data.byteswap().tobytes()
        else:
            rawdata = data.tobytes()
        self._writeWithRetry(rawdata, brickstatus, fileoffset, brickpos, lod, verbose = None)

    def _writeOneConstantBrick(self, data, brickpos, lod, *, verbose = None):
        self._argType(data, (ScalarBuffer,))
        dprint("@@ _writeOneConstantBrick(pos={0}, value={1})".format(brickpos, data))
        self._setBrickFilePosition(brickpos[0], brickpos[1], brickpos[2], lod,
                                   impl_enum.BrickStatus.Constant, data.value, 0)

    def _usedPartOfBrick(self, size, brickpos, lod):
        """
        Compute the size of the used (inside survey) area of a data buffer
        with size "size". size will probably always be equal to bricksize
        but the function should still work if it isn't.
        """
        # TODO-Low: Refactor, should I enforce storing the members as numpy arrays?
        bricksize  = np.array(self._metadata._ih._bricksize, dtype=np.int64)
        surveysize = np.array(self._metadata._ih._size, dtype=np.int64)
        brickpos   = np.array(brickpos, dtype=np.int64)
        surveysize = (surveysize + (1<<lod) - 1) // (1<<lod)
        available =  surveysize - (brickpos * bricksize)
        return np.maximum((0, 0, 0), np.minimum(available, size))

    def _isUsedPartOfBrickAllConstant(self, data, brickpos, lod):
        """
        Return True if all useful samples in this brick have the same value.
        Padding samples outside the survey are not useful and should not
        be checked since they could easily have been set to a different value.
        """
        self._argType(data, (np.ndarray,))
        used = self._usedPartOfBrick(data.shape, brickpos, lod)
        first = data.flat[0]
        if np.isnan(first):
            return np.all(np.isnan(data[:used[0],:used[1],:used[2]]))
        else:
            return np.all(data[:used[0],:used[1],:used[2]] == first)

    def _mustLeakOldBrick(self, data, compressor, brickstatus):
        """
        Return True if this block needs to be leaked by pretending
        that its block offset has not been allocated yet.
        Raise an exception if the update us disallowed by _update_mode.

        Note, in "Always" mode I am only supposed to leak the block
        if the new one is larger. But that is too much work to
        figure out here. So, treat "Always" as if it were "Pedantic".

        Note, there is one other place where bricks might leak:
        in _writeWithRetry() when a ZgySegmentIsClosed is caught.
        The update mode doesn't need to be checked again at that point
        unless we want to support UpdateMode.NoLeaks.
        """
        self._argType(data, (np.ndarray, ScalarBuffer))
        msg = "Updating a {0} BrickStatus.{1} brick with {2} data is illegal in UpdateMode.{3}.".format(
            "cloud" if self._file.xx_iscloud else "local",
            brickstatus.name,
            "Compressed" if compressor else "Normal" if isinstance(data, np.ndarray) else "Constant",
            self._update_mode.name)
        if brickstatus != impl_enum.BrickStatus.Missing:
            if self._update_mode in (impl_enum.UpdateMode.Never,):
                raise ZgyUserError(msg)
        if brickstatus == impl_enum.BrickStatus.Normal:
            if self._update_mode in (impl_enum.UpdateMode.Never, impl_enum.UpdateMode.Constant):
                raise ZgyUserError(msg)
        if brickstatus == impl_enum.BrickStatus.Compressed or (brickstatus == impl_enum.BrickStatus.Normal and compressor is not None):
            if self._update_mode in (impl_enum.UpdateMode.Pedantic, impl_enum.UpdateMode.Always):
                return True
            else:
                raise ZgyUserError(msg)
        return False

    def _writeOneBrick(self, data, brickpos, lod, compressor, *, verbose = None):
        """
        Write a single brick to storage. Normally the data should either
        be a numpy scalar (all samples have the same value) or a numpy array
        with the same shape as the file's brick size.

        brickpos is given relative to this lod level. For lod 0 the valid
        range is the survey size in bricks. For lod 1 it is half that,
        rounded upwards.

        The data must already have been scaled to storage values and
        converted to the appropriate value type. It must either be a
        scalar (to fill the entire brick) or a numpy array.

        Caveat: int and float scalars won't work. Be explicit and pass an
        np.int8, np.int16, or np.float32.
        """
        self._argType(data, (np.ndarray, ScalarBuffer))
        self._checkCompatibleBufferTypes(data.dtype, is_storage = True)
        brickstatus, fileoffset, constvalue, bricksize = self._getBrickFilePosition(brickpos[0], brickpos[1], brickpos[2], lod)

        if self._mustLeakOldBrick(data, compressor, brickstatus):
            brickstatus, fileoffset, constvalue, bricksize = (
                impl_enum.BrickStatus.Missing, None, 0, 0)

        data_const = not isinstance(data, np.ndarray)
        file_const = brickstatus not in (impl_enum.BrickStatus.Normal, impl_enum.BrickStatus.Compressed)
        if file_const: # also true if never written yet.
            if data_const:
                # Caller asked to store a constant value.
                dprint("@@ Explicit store constant")
                self._writeOneConstantBrick(data, brickpos, lod, verbose=verbose)
            elif self._isUsedPartOfBrickAllConstant(data, brickpos, lod):
                # Caller did not explicitly ask for a constant value,
                # but all useable samples (i.e. all samples that are
                # inside the survey boundaries) have ths same value.
                tmp = ScalarBuffer(data.shape, data[0,0,0], data.dtype)
                self._writeOneConstantBrick(tmp, brickpos, lod, verbose=verbose)
            else:
                # Allocate a new disk block and write out this block.
                self._writeOneNormalBrick(data, None, brickpos, lod, compressor=compressor, verbose=verbose)
        else:
            if data_const:
                dprint("@@ Const data expanded before storing")
                # The brick has already been allocated. Cannot set it to
                # constant-value because the file storage for this brick
                # would then be leaked.
                data = np.full(self._metadata._ih._bricksize[:3], data.value, dtype=data.dtype)
            # No compression, for the same reason as above.
            self._writeOneNormalBrick(data, fileoffset, brickpos, lod, compressor=None, verbose=verbose)

    def _writeAlignedRegion(self, data, start, lod, compressor, *, verbose = None):
        """
        Write an arbitrary region covering one or more full bricks.
        The data must already be converted to storage.
        Start must be a multiple of brick size. start + count must
        either be a multiple of block size or set to the end of the
        survey.
        With current usage the data may or may not have been padded to
        the brick size depending on whether a read/modify/write was used.
        """
        self._argType(data, (np.ndarray, ScalarBuffer))
        # Meta information from the file only.
        bs = np.array(self._metadata._ih._bricksize, dtype=np.int64)
        file_dtype = impl_enum._map_DataTypeToNumpyType(self._metadata._ih._datatype)
        self._checkCompatibleBufferTypes(data.dtype, is_storage=True)
        survey_beg = np.array((0, 0, 0), dtype=np.int64)
        survey_end = np.array(self._metadata._ih._size, dtype=np.int64)
        survey_end = (survey_end + (1<<lod) - 1) // (1<<lod)
        defaultstorage = self._defaultValue(as_float=False)
        # Massaging of arguments to this function
        dprint("@@ _writeRegion(start {0}, count {1}, type {2} -> {3})".format(
            start, data.shape, data.dtype, np.dtype(file_dtype)))
        start = np.array(start, dtype=np.int64)
        count = np.array(data.shape, dtype=np.int64)
        beg_brick = (start // bs) * bs
        end_brick = ((start + count + bs - 1) // bs) * bs
        if isinstance(data, np.ndarray):
            brick = np.zeros(bs, dtype=file_dtype)
        else:
            brick = ScalarBuffer(bs, data.value, data.dtype)
        for ii in range(beg_brick[0], end_brick[0], bs[0]):
            for jj in range(beg_brick[1], end_brick[1], bs[1]):
                for kk in range(beg_brick[2], end_brick[2], bs[2]):
                    this_beg = np.array([ii, jj, kk], dtype=np.int64)
                    brickpos = this_beg // bs
                    if isinstance(brick, np.ndarray):
                        brick.fill(defaultstorage)
                        self._partialCopy(data, start, data.shape,
                                          brick, this_beg, bs,
                                          survey_beg, survey_end - survey_beg,
                                          verbose=verbose)
                    # TODO-Medium Note Compression:
                    # This might be a good extension point, as this is the
                    # last place where we still have a list of bricks to
                    # be written. So it would be easier to parallelize here.
                    # But we don't yet know which bricks will be written as
                    # all-constant so there would need to be some refactoring.
                    # _writeOneNormalBrick() probably needs to be called from here.
                    # See comments in that function; it will need some
                    # changes. And _writeOneBrick() needs to be told whether
                    # the data has been compressed or not. Instead of
                    # being told which compressor (if any) to use.

                    # If there are any errors during _writeOneBrick() this
                    # probably means the entire file is a lost cause.
                    # This is true also for ZgyUserError raised in the file
                    # layer, because at that layer the "user" is really
                    # OpenZGY and not some client code. The only acceptable
                    # error is ZgySegmentIsClosed, and that will be caught
                    # and handled at lower levels.

                    # TODO-Low the logic here can probably be refined.
                    # If there is no buffering (on-prem files) then a
                    # write can probably just be re-tried if we make
                    # sure the write is done before updating metadata.
                    # But do we really want that complexity? The risk
                    # of transient errors is way higher with cloud access.
                    # And due to buffering in the file layer those would
                    # also be a lot harder to recover from.

                    with ErrorsWillCorruptFile(self):
                        self._writeOneBrick(brick, brickpos,
                                            lod=lod,
                                            compressor=compressor,
                                            verbose=verbose)

    def _writeRegion(self, data, start, lod, compressor, is_storage, *, verbose = None):
        """
        Write an arbitrary region. If start and end are not aligned to
        the brick size this will do a read/modify/write. Note that when
        writing to the cloud it is highly recommended to write aligned
        regions only. Otherwise some disk space might be wasted.

        Performance note: The read/modify/write could also have been done
        one brick at a time. Doing it here means that for large requests
        a number of bricks which were already full will also be read.
        On the other hand a single read might help parallelism. Either
        way the recommendation is for the user to write brick aligned data.
        So this might not be a big deal.
        """
        self._argType(data, (np.ndarray, ScalarBuffer))
        self._checkCompatibleBufferTypes(data.dtype, is_storage)

        if not is_storage:
            data = self._scaleDataToStorage(data)

        beg = np.array(start, dtype=np.int64)
        end = np.array(data.shape, dtype=np.int64) + beg
        bs = np.array(self._metadata._ih._bricksize, dtype=np.int64)
        survey_beg = np.array((0, 0, 0), dtype=np.int64)
        survey_end = (np.array(self._metadata._ih._size, dtype=np.int64) + ((1<<lod) - 1)) // (1<<lod)
        need_rmw = any([(beg[i]%bs[i]) != 0 or ((end[i]%bs[i]) != 0 and end[i] < survey_end[i]) for i in range(3)])
        if need_rmw:
            # Since we are doing a read/modify/write, also read any padding
            # outside the survey. _writeAlignedRegion will see full
            # bricks also at the end of the survey. If no r/m/w needed
            # the latter is not guaranteed.
            new_start = (beg // bs) * bs
            new_count = (((end + bs - 1) // bs) * bs) - new_start
            new_data = np.zeros(new_count, dtype=data.dtype)
            dprint("@@ _writeRegion r/m/w: user {0} {1} padded {2} {3}".format(
                start, data.shape, new_start, new_count))
            # The reader takes care of any defaultvalue.
            # as_float=False because we converted, and data.dtype==file_dtype
            self.readToExistingBuffer(new_data, new_start, lod=lod,
                                      as_float=False,
                                      verbose=verbose)
            self._partialCopy(data, start, data.shape,
                              new_data, new_start, new_data.shape,
                              survey_beg, survey_end - survey_beg,
                              verbose=verbose)
            data = new_data
            start = new_start
        else:
            dprint("@@ _writeRegion direct: user {0} {1}".format(start, data.shape))

        self._writeAlignedRegion(data, start, lod=lod,
                                 compressor=compressor,
                                 verbose=verbose)

        data = data.value if isinstance(data, ScalarBuffer) else data
        if np.issubdtype(data.dtype, np.integer):
            tmp_min = np.amin(data)
            tmp_max = np.amax(data)
            if self._sample_min > self._sample_max:
                # Changes _sample_min / _sample_max to the correct value type.
                self._sample_min = tmp_min
                self._sample_max = tmp_max
            else:
                self._sample_min = min(self._sample_min, tmp_min)
                self._sample_max = max(self._sample_max, tmp_max)
        else:
            valid = np.isfinite(data).astype(np.bool)
            self._sample_min = np.amin(data, where=valid, initial=self._sample_min)
            self._sample_max = np.amax(data, where=valid, initial=self._sample_max)

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
