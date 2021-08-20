#!/usr/bin/env python3

"""
Efficient iterator for reading an entire ZGY file.

Naming conventions used in this file for parameters:

    surveysize: (i,j,k)-- Size of entire survey without padding.
    bricksize: (i,j,k) -- How data is stored. Usually (64, 64, 64).
    blocksize: (i,j,k) -- How much to read (at most) in one call.
    chunksize: (i,j,k) -- How much to return (at most) at once.
    maxbytes: int64    -- Max bytes to read at once, defaults to 1 GB
    dtype: np.dtype    -- Sample data type to return (float or storage).
    readfn: callable   -- To be called when reading a block.
    progress: callable -- Invoked after each read from disk.
"""
##@package openzgy.iterator
#@brief Efficient iterator for reading an entire ZGY file.

import numpy as np
from .api import _map_SampleDataTypeToDataType
from .impl.enum import _map_DataTypeToNumpyType

def _roundup(x, step):
    return ((x + step - 1) // step) * step

def _choose_blocksize(surveysize, bricksize, samplesize, maxbytes):
    """
    Get a reasonable blocksize (a.k.a. buffer size) to use when reading.
    Always read full traces, even if this means exceeding the specified
    maxbytes. Always read full bricks. Read more if it will fit inside
    the specified maxbytes.

    In each dimension the block size will either be the full survey size
    padded to bricksize or a power-of-two multiple of the brick size.
    In the simplest case caller can pass maxbytes=0 meaning read as little
    as possible; this will return (bricksize[0], bricksize[1], surveysize[2])
    a.k.a. one brick column.

    Note: It is desirable to have blocksize being a multiple of chunksize
    and chunksize being a multiple of bricksize. _choose_blocksize() tries
    to help with this: If there is room for more than one brick-column it
    will multiple the size in some direction, but only with a power of 2.
    This means that the adjustment will not break this condition.
    If it had not been for the power-of-two limitation this might not work.

    If the condition is met then the consumer knows that if a chunk smaller
    than the requested chunksize is received then this can only be due to
    reading at the end or bottom of the survey.
    """
    surveysize = list([_roundup(surveysize[i], bricksize[i]) for i in range(3)])
    bs = [bricksize[0], bricksize[1], surveysize[2]]
    while bs[0] * bs[1] * bs[2] * samplesize * 2 < maxbytes:
        bs[1] *= 2
    bs[1] = min(bs[1], surveysize[1])
    while bs[0] * bs[1] * bs[2] * samplesize * 2 < maxbytes:
        bs[0] *= 2
    bs[0] = min(bs[0], surveysize[0])
    return bs

def _readall_1(surveysize, blocksize, dtype, readfn):
    """
    Iterates over the entire file and returns data in blocks.
    This is the internal part of readall_2, it handles blocking the
    read but not chunking the data to be returned to the caller.
    """
    data  = np.zeros(blocksize, dtype=dtype) if readfn else None
    done  = np.int64(0)
    total = np.product(surveysize)

    for ii in range(0, surveysize[0], blocksize[0]):
        for jj in range(0, surveysize[1], blocksize[1]):
            for kk in range(0, surveysize[2], blocksize[2]):
                start = np.array((ii, jj, kk), dtype=np.int64)
                count = np.minimum(start + blocksize, surveysize) - start
                view = data[:count[0],:count[1],:count[2]] if data is not None else None
                #print("Reading", start, count, view.shape)
                if readfn: readfn(start, view)
                done += np.product(count)
                yield start, count, view
    assert done == np.product(surveysize)

def _readall_2(surveysize, blocksize, chunksize, dtype, readfn, progress):
    """
    Iterates over the entire file and returns data in chunks.
    All numeric and array parameters use numpy types.
    """
    done  = np.int64(0)
    total = np.product(surveysize)
    # Give a chance to abort before we even start.
    if progress and not progress(done, total): return
    alldata = _readall_1(surveysize=surveysize,
                         blocksize=blocksize,
                         dtype=dtype,
                         readfn=readfn)
    for datastart, datasize, data in alldata:
        for ii in range(0, datasize[0], chunksize[0]):
            for jj in range(0, datasize[1], chunksize[1]):
                # After reading a block but before yielding.
                if progress and not progress(done, total): return
                for kk in range(0, datasize[2], chunksize[2]):
                    start = np.array((ii, jj, kk), dtype=np.int64)
                    end   = np.minimum(start + chunksize, datasize)
                    count = end - start
                    view = data[start[0]:end[0],start[1]:end[1],start[2]:end[2]] if data is not None else None
                    done += np.product(count)
                    yield datastart + start, count, view
        # After yielding, give a chance to abort before the next read.
        # Also makes sure the final done==total is sent.
        if progress and not progress(done, total): return
    assert done == np.product(surveysize)

def _readall_3(surveysize, bricksize, blocksize, chunksize, dtype, readfn, maxbytes, progress):
    """
    Handle default arguments, make sure all data has numpy types,
    and then pass control to _readall_2.
    """
    surveysize = np.array(surveysize, dtype=np.int64)
    bricksize  = np.array(bricksize, dtype=np.int64)
    maxbytes   = np.int64(maxbytes if maxbytes is not None else 128*1024*1024)
    blocksize  = np.array(blocksize if blocksize is not None else _choose_blocksize(
        surveysize, bricksize, np.dtype(dtype).itemsize, maxbytes),
        dtype=np.int64)
    chunksize  = np.array(chunksize if chunksize is not None else blocksize, dtype=np.int64)
    blocksize[blocksize==0] = surveysize[blocksize==0]
    chunksize[chunksize==0] = blocksize[chunksize==0]
    if False:
        fmt = lambda x: "{0} = {1} voxels, {2:.1f} MB".format(str(tuple(x)), np.product(x), np.product(x) * dtype.itemsize/(1024*1024))
        print("survey", fmt(surveysize), "of", np.dtype(dtype).name)
        print("brick ", fmt(bricksize))
        print("block ", fmt(blocksize))
        print("chunk ", fmt(chunksize))
    # Since _readall_2 contains a yield, the function won't actually be
    # invoked until somebody consumes its output.
    result = _readall_2(surveysize=surveysize,
                      blocksize=blocksize,
                      chunksize=chunksize,
                      dtype=dtype,
                      readfn=readfn,
                      progress=progress)
    return result

def readall(r, *, blocksize = None, chunksize=None, maxbytes=128*1024*1024, dtype=None, cropsize = None, cropoffset = None, lod = 0, sizeonly = False, progress=None):
    """
    Convenience function to iterate over an entire cube, trying to use
    an efficient block size on read and optionally returning a smaller
    chunk size to the caller. The returned data buffer belongs to this
    iterator and may be overwritten on the next call. Two different
    iterators do not share their buffers.

    Note: If blocksize is a multiple of chunksize and chunksize is a
    multiple of bricksize then the following holds: if a chunk smaller
    than the requested chunksize is received then this can only be due
    to reading at the end or bottom of the survey. The caller might rely
    on this. E.g. when computing low resolution bricks and writing them
    out in a way that avoids a read/modify/write.

    CAVEAT: This function mingh be overkill in some cases.

    parameters:
        r: ZgyReader       -- The open ZGY file
        blocksize: (i,j,k) -- How much to read (at most) in one call.
                              If omitted, use a reasonable default.
                              Dims set to 0 mean "as much as possible".
        chunksize: (i,j,k) -- How much to return (at most) at once.
                              Defaults to blocksize, i.e. return the
                              same blocks that are read. Will never
                              return more than blocksize, so if set to
                              more than 64 you probably want to set an
                              explicit blocksize as well.
        maxbytes: int64    -- Max bytes to read at once, defaults to 128 MB
                              Ignored if an explicit blocksize is specified.
        dtype: np.dtype    -- Data type of returned buffer. Defaults to
                              what is in the file. Use np.float32 to
                              convert and scale (if needed) to float.
        cropsize: (i,j,k)  -- Only read this amount of data.
                              Note, no sanity check of this parameter.
        cropoffset: (i,j,k)   Start reading at this point.
                              Note, no sanity check of this parameter.
        lod: int              Pass >0 for decimated data.
        sizeonly: bool     -- If false, return sizes but not data.
        progress: callable -- Invoked after each read from disk.
    """
    if sizeonly:
        readfn = None
    elif cropoffset is None or tuple(cropoffset) == (0, 0, 0):
        readfn = lambda pos, data: r.read(pos, data, lod=lod)
    else:
        readfn = lambda pos, data: r.read(
            (pos[0]+cropoffset[0],
             pos[1]+cropoffset[1],
             pos[2]+cropoffset[2]),
            data, lod=lod)

    impl_dtype = _map_SampleDataTypeToDataType(r.datatype)
    dtype      = dtype or np.dtype(_map_DataTypeToNumpyType(impl_dtype))
    size       = np.array(r.size, dtype=np.int64)
    size       = (size + ((1<<lod) - 1)) // (1<<lod)
    return _readall_3(surveysize=cropsize if cropsize is not None else size,
                      bricksize=r.bricksize,
                      blocksize=blocksize,
                      chunksize=chunksize,
                      dtype=dtype,
                      readfn=readfn,
                      maxbytes=maxbytes,
                      progress=progress)

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
