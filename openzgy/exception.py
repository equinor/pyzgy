#!/usr/bin/env python3

"""
Defines exceptions that may be raised by OpenZGY.

These classes are both visible to the OpenZGY public API and referenced
directly from the implementation classes. I apologize for the broken
encapsulation. Re-mapping the exceptions in the API layer didn't seem
worth the trouble.
"""
##@package openzgy.exception
#@brief Exceptions that may be raised by OpenZGY.

##@brief Base class for all exceptions thrown by %OpenZGY.
class ZgyError(Exception):
    """
    Base class for all exceptions thrown by %OpenZGY.
    """
    pass


##@brief Corrupted or unsupported ZGY file.
class ZgyFormatError(ZgyError):
    """
    Corrupted or unsupported ZGY file.

    In some cases a corrupted file might lead to a ZgyInternalError
    or ZgyEndOfFile being thrown instead of this one. Because it isn't
    always easy to figure out the root cause.
    """
    pass

##@brief The ZGY file became corrupted while writing to it.
class ZgyCorruptedFile(ZgyError):
    """
    The ZGY file became corrupted while writing to it.

    No further writes are allowed on this file because a previous write
    raised an exception and we don't know the file's state. Subsequent
    writes will also throw this exception.

    The safe approach is to assume that the error caused the file to
    become corrupted. It is recommended that the application closes and
    deletes the file.
    """
    pass

##@brief Exception that might be caused by the calling application.
class ZgyUserError(ZgyError):
    """
    Exception that might be caused by the calling application.

    Determining whether a problem is the fault of the calling application
    or the %OpenZGY library itself can be guesswork. Application code
    might choose to treat ZgyUserError and ZgyInternalError the same way.
    """
    pass

##@brief Exception that might be caused by a bug in %OpenZGY.
class ZgyInternalError(ZgyError):
    """
    Exception that might be caused by a bug in %OpenZGY.

    Determining whether a problem is the fault of the calling application
    or the %OpenZGY library itself can be guesswork. Application code
    might choose to treat ZgyUserError and ZgyInternalError the same way.

    A corrupt file might also be reported as ZgyInternalError instead of
    the more appropriate ZgyFormatError.
    """
    pass

##@brief Trying to read past EOF.
class ZgyEndOfFile(ZgyError):
    """
    Trying to read past EOF.

    This is always considered an error, and is often due to a corrupted
    ZGY file. So this error should probably be treated as a ZgyFormatError.
    """
    pass

##@brief Exception used internally to request a retry.
class ZgySegmentIsClosed(ZgyError):
    """
    Exception used internally to request a retry.
    A write to the cloud failed because the region that was attempted
    written had already been flushed. And the cloud back-end does not
    allow writing it again. The calling code, still inside the OpenZGY
    library, should be able to catch and recover from this problem.
    """
    pass

##@brief User aborted the operation.
class ZgyAborted(ZgyError):
    """
    User aborted the operation.

    If the user supplied a progress callback and this callback returned
    false then the operation in progress will and by throwing this
    exception. Which means that this is not an error; it is a consequence
    of the abort.
    """
    pass

##@brief Missing feature.
class ZgyMissingFeature(ZgyError):
    """
    Missing feature.

    Raised if some optional plug-in (e.g. some cloud back end or a
    compressor) was loaded or explicitly requested, so we know about
    it, but the plug-in is not operational for some reason.
    """
    pass

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
