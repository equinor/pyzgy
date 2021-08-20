#!/usr/bin/env python3

"""
Enums not visible to the public API.

Some of the enums are used to describe parts of the file format,
giving a symbolic name to an integer stored in the file.
That kind of information should definitely be hidden from clients.
"""
##@package openzgy.impl.enum
#@brief Enums not visible to the public API.

from enum import Enum
import numpy as np

class RawDataType(Enum):
    """
    Sample data type as stored on the file.
    In the public API this maps to SampleDataType.
    Source: BrickedFileVersion.cpp, MetaDataValue.h
    This enum is used for all versions.
    Note that the existing public ZGY library
    only recognizes SignedInt8, SignedInt16, and Float32.
    """
    SignedInt8    = 0
    UnsignedInt8  = 1
    SignedInt16   = 2
    UnsignedInt16 = 3
    SignedInt32   = 4
    UnsignedInt32 = 5
    Float32       = 6
    IbmFloat32    = 7

def _map_DataTypeToStructFormatCode(dt):
    """Internal method to get the "struct" format code."""
    return {
        RawDataType.SignedInt8:    "<b",
        RawDataType.UnsignedInt8:  "<B",
        RawDataType.SignedInt16:   "<h",
        RawDataType.UnsignedInt16: "<H",
        RawDataType.SignedInt32:   "<i",
        RawDataType.UnsignedInt32: "<I",
        RawDataType.Float32:       "<f",
    }[dt]

def _map_DataTypeToNumpyType(dt):
    """
    Internal method to get the "numpy" format code.
    Caveat: This returns a type object inheriting numpy.number,
    not a numpy.dtype instance. These are often interchangeable
    but not always.
    """
    return {
        RawDataType.SignedInt8:    np.int8,
        RawDataType.UnsignedInt8:  np.uint8,
        RawDataType.SignedInt16:   np.int16,
        RawDataType.UnsignedInt16: np.uint16,
        RawDataType.SignedInt32:   np.int32,
        RawDataType.UnsignedInt32: np.uint32,
        RawDataType.Float32:       np.float32,
    }[dt]

class RawCoordType(Enum):
    """
    Coordinate type codes as stored in V1 files only.
    The values are stored on the file, so the numbers must not be changed.
    Source: BrickedFileVersion.cpp.
    There is no corresponding enum in the API layer.
    """
    Unknown        = 0
    Meters         = 1
    Feet           = 2
    ArcSec         = 3  # value = deg*3600 + min*60 + sec
    ArcDeg         = 4  # value = deg + min/60 + sec/3600
    ArcDegMinSec   = 5  # value = deg*10000 + min*100 + sec

class RawHorizontalDimension(Enum):
    """
    Horizontal dimension as seen in V2 files and later.
    In the public API this maps to UnitDimension.
    The values are stored in the file, so the numbers must not be changed.
    Source: PoststackSeis3dInfo.h, MetaDataValue.h, ReaderImp::getMetaData
    """
    Unknown = 0
    Length = 1
    ArcAngle = 2

class RawVerticalDimension(Enum):
    """
    Vertical dimension as seen in V2 files and later.
    In the public API this maps to UnitDimension.
    The values are stored in the file, so the numbers must not be changed.
    Source: PoststackSeis3dInfo.h, MetaDataValue.h, ReaderImp::getMetaData
    """
    Unknown = 0
    Depth = 1
    SeismicTWT = 2
    SeismicOWT = 3

class RawGridDefinition(Enum):
    """
    Method used to define the geometry. Only FourPoint is allowed for write,
    and only ThreePoint (treated as FourPoint) and FourPoint supported on read.
    The values are stored in the file, so the numbers must not be changed.
    There is no corresponding enum in the API layer.
    """
    Unknown    = 0
    Parametric = 1
    ThreePoint = 2
    FourPoint  = 3

class BrickStatus(Enum):
    """
    Brick status as used in the internal API only.
    """
    Missing = 0
    Constant = 1
    Normal = 2
    Compressed = 3

class UpdateMode(Enum):
    """
    WORK IN PROGRESS, potential configurable behavior.

    A ZGY file cannot be updated once created, but individual bricks might
    be written to more than once while the file is still open for create.

    Updating a brick might cause loss of quality if the update was made
    as part of a read/modify/write cycle. It might also cause space to be
    wasted in the file since ZGY does not try to recycle freed bricks.
    For this reason the application should explicitly indicate that it
    accepts the loss of quality and/or leakage.

    Kinds of leakage:

     -  Brick to be overwritten is in a closed segment. This is
        expected to be rare, and only relevant for cloud storage.

     -  Brick to be overwritten and/or new brick is compressed
        and the new data is smaller. Leaks the size difference,
        although for implementation reasons we might want to leak
        the entire old brick ("Pedantic" mode).

     -  Brick to be overwritten and/or new brick is compressed
        and the new data is larger. Leaks the old brick.

    The default is "Always" for uncompressed local files and "Constant"
    otherwise.

    It is fairly safe to set an uncompressed cloud file to "Always" but
    there are some scenarios where very small regions are written to a
    large file where this might cause much leakage. So the caller
    needs to confirm he knows what he is doing.

    Compressed files should only be set to "Always" in very special cases
    or in unit tests. The potential leakage is much larger, as is the
    problem of multiple compress and decompress cycles causing noise.
    """

    Never = 0       # Never allow updating. Can only write to "Missing" bricks.

    Constant = 1    # Can write to both "Missing" and "Constant" bricks. This
                    # permission is needed if the client wants to establish
		    # a default value for missing samples by setting the entire
		    # survey to this default. Followed by writing real data.
		    # "Never" and "Constant" require that the application
		    # writes brick aligned data. If a read/modify/write is
		    # indicated then this will raise an exception.
		    # The only drawback of "Constant" over "Never" is that
		    # "Constant" can cause a slight confusion: A particular
		    # read/modify/write might be allowed if the previous write
		    # just happened to have all constant values.

    #NoCompress = 2 # Not recommended because it gives confusing behavior.
                    # Update is only allowed when both the old and the new
                    # brick is uncompressed. The only leaks allowed are those
                    # caused by the target block being in a closed segment on
                    # the cloud. The problem is that the decision is made
                    # per brick. It is more consistent to decide up front
                    # whether any compression might happen. If so, use
                    # Constant or Never. If not, just use Always.

    #NoLeaks = 3    # Not recommended because it gives confusing behavior.
                    # As "Always" but if it turns out that a brick would be
                    # leaked, even in the rare "closed segment" case, the
                    # code will raise an exception.

    Always = 4      # Always allow updating. This may cause leaked data in
                    # some cases. Uncompressed local data will not leak.
                    # Uncompressed cloud data can only leak when the target
                    # block being in a closed segment on the cloud.
		    # This can be the default for uncompressed data.
		    # For compressed data it is usually a bad idea.
                    # Not only could there be a lot more leakage, but if
                    # the reason for the overwrite is a read/modify/write
                    # and the data had lossy compression then the
                    # compression noise will accumulate.

    Pedantic = 5    # As "Always", but when either the existing or the new
                    # brick is compressed then the old brick is leaked
                    # unconditionally. This gives more reproducible behavior
                    # but wastes more space.

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
