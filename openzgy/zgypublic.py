#!/usr/bin/env python3

"""
Consolidate old and new Python API.

This file is for testing only, when comparing the old and new apis.
The Python wrapper for the old ZGY-Public API is not quite identical
to the new OpenZGY api. This file wraps the old API to become more
similar to the new.

Application code should NOT use this package when migrating. Instead,
please modify the code to fit the slightly changed API. The documentation
for this package might help to show which changes are required.
"""
##@package openzgy.zgypublic
#@brief Consolidate old and new Python API.

import zgy
from .api import SampleDataType, UnitDimension

def _fixKeywordArgs(kwargs):
    """
    The new API uses real enums instead of stringly typed values.
    The names of each tag matches, so the conversion is trivial.
    Also handles "iocontext" and "sourcetype" although those two
    are not in either interface yet.
    """
    result = {}
    for key, value in kwargs.items():
        if key in ("zunitdim", "hunitdim", "datatype", "sourcetype"):
            result[key] = value.name
        elif key in ("iocontext", "verbose"):
            pass
        else:
            result[key] = value
    return result

def _fixResultData(kwdata):
    """
    Inverse of _fixKeywordArgs. Convert strings to enums.
    """
    result = {}
    for key, value in kwdata.items():
        if key in ("datatype", "sourcetype"):
            result[key] = SampleDataType[value]
        elif key in ("zunitdim", "hunitdim"):
            result[key] = UnitDimension[value]
        else:
            result[key] = value
    return result

# Note, using mostly Doxygen style annotation because for this file
# it isn't really useful to have help instde Python.

class ZgyReader(zgy.ZgyReader):
    ##@brief Ignore iocontext and verbose arguments.
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **_fixKeywordArgs(kwargs))

    ##@brief Ignore iocontext and verbose arguments.
    def read(self, *args, **kwargs):
        return super().read(*args, **_fixKeywordArgs(kwargs))

    ##@brief Old: strings, new: enums for 4 properties.
    #@details datatype, sourcetype, zunitdim, hunitdim need to be mapped.
    @property
    def meta(self):
        return _fixResultData(super().meta)

    ##@brief Old: string, new: openzgy.api.SampleDataType.
    @property
    def datatype(self):
        return SampleDataType[super().datatype]

    ##@brief Attribute doesn't exist in old api
    @property
    def raw_datarange(self):
        return super().datarange

    ##@brief Old: string, new: openzgy.api.UnitDimension.
    @property
    def zunitdim(self):
        return UnitDimension[super().zunitdim]

    ##@brief Old: string, new: openzgy.api.UnitDimension.
    @property
    def hunitdim(self):
        return UnitDimension[super().hunitdim]

    ##@brief Old: missing from api.
    #@details For the old API the bricksize will always be returned as
    #(64,64,64) on read and always set to that value on file create.
    @property
    def bricksize(self):
        return (64, 64, 64)

class ZgyWriter(zgy.ZgyWriter):
    ##@brief Ignore iocontext and verbose, and map 4 enums to string.
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **_fixKeywordArgs(kwargs))

    ##@brief Ignore iocontext and verbose arguments.
    def write(self, *args, **kwargs):
        return super().write(*args, **_fixKeywordArgs(kwargs))

    ##@brief Ignore and let _\_exit_\_ handle it instead.
    def finalize(*args, **kwargs):
        return None

    ##@brief Old: strings, new: enums for 4 properties.
    #@details datatype, sourcetype, zunitdim, hunitdim need to be mapped.
    @property
    def meta(self):
        return _fixResultData(super().meta)

    ##@brief Old: string, new: openzgy.api.SampleDataType.
    @property
    def datatype(self):
        return SampleDataType[super().datatype]

    ##@brief Attribute doesn't exist in old api
    @property
    def raw_datarange(self):
        return super().datarange

    ##@brief Old: string, new: openzgy.api.UnitDimension.
    @property
    def zunitdim(self):
        return UnitDimension[super().zunitdim]

    ##@brief Old: string, new: openzgy.api.UnitDimension.
    @property
    def hunitdim(self):
        return UnitDimension[super().hunitdim]

    ##@brief Old: missing from api.
    #@details For the old API the bricksize will always be returned as
    #(64,64,64) on read and always set to that value on file create.
    @property
    def bricksize(self):
        return (64, 64, 64)

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
