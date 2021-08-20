#!/usr/bin/env python3

"""
Test misc.
"""

import os, time
import numpy as np
from .. import api

# If SD tests cannot run then in most cases treat this as a warning only.
# There is not much point in letting dozens of tests fail for the same
# reason. Once Seismic Store is fully supported, make one and only one
# test that fails if we cannot connect.
#
# Tests do not hard code credentials, so either check that the fallback
# environment has been set, or explicitly pick those up here.
try:
    import sdglue as sd
except Exception as ex:
    print("seismic store access via sdglue is not available:", ex)
    sd = None
if sd and not (os.getenv("OPENZGY_SDURL") and os.getenv("OPENZGY_SDAPIKEY")):
    print("seismic store access requires $OPENZGY_SDURL and $OPENZGY_SDAPIKEY")
    sd = None

def SDTestData(name):
    """
    Return the full path of a test file stored in Seismic Store.
    The default sd:// uri is hard coded but may be overridden by
    setting $OPENZGY_SDTESTDATA. The expected test data must have
    been uploaded to that location. The tests need read access and
    should preferably not have write access. If name is empty then
    just return the prefix without a trailing slash.
    """
    project = os.getenv("OPENZGY_SDTESTDATA", "sd://sntc/testdata")
    if project and project[-1] == "/": project = project[:-1]
    return project + "/" + name if name else project

def SDTestSink(name):
    """
    Return the full path to where in Seismic Store the specified test
    output should be stored. The default sd:// uri is hard coded but
    may be overridden by setting $OPENZGY_SDTESTSINK. The tests need
    read/write access. If name is empty then just return the prefix
    without a trailing slash.
    """
    project = os.getenv("OPENZGY_SDTESTSINK", "sd://sntc/testsink/d")
    if project and project[-1] == "/": project = project[:-1]
    return project + "/" + name if name else project

def SDCredentials(**kwargs):
    """
    Convenience to hard code credentials for testing. Returns a dict.
    Picking up sdurl/sdapikey from the environment is redundant since
    the library already does this as a fallback.
    """
    result = {
        "sdurl":    os.getenv("OPENZGY_SDURL", ""),
        "sdapikey": os.getenv("OPENZGY_SDAPIKEY", ""),
        "sdtoken":  os.getenv("OPENZGY_TOKEN", "FILE:carbon.slbapp.com")
    }
    for k, v in kwargs.items():
        result[k] = v
    return result

def HasSeismicStore():
    """
    Used to enable or disable unit tests based on what software is loaded.
    TODO-Low structure the tests better, to make it less likely to need this.
    """
    return not sd is None

def HasZFPCompression():
    """
    Used to enable or disable unit tests based on what software is loaded.
    TODO-Low structure the tests better, to make it less likely to need this.
    """
    return ("ZFP" in api.ZgyKnownCompressors() and
            "ZFP" in api.ZgyKnownDecompressors())

class TempFileAutoDelete:
    """
    Arrange for this explicitly named temporary file to be deleted when
    the instance goes out of scope. Works both for seismic store and locally.

    A missing file is considered an error. The test somehow did not manage
    to create the file. If the test wants to delete the file itself then
    it should call disarm() after the deletion.

    If an exception is already pending then a missing or undeletable file
    only causes a warning to be printed. The pending exception is presumably
    more important.

    There is a fairly harmless race condition between the creation of the
    instance and the actual creation of the file. If an exception is thrown
    in that interval then a message is printed about not being able to delete
    the file. You may alternatively pass armed=False to the constructor and
    then invoke arm() at the exact point you know the file exists. Good luck.
    If you get an exception when creating a file on the cloud, you might not
    know whether the file got created or not.

    The recommended naming convention is to use both the current time and
    a per-process random number as a prefix. This makes it simpler to
    clean up files that in spite of the aito delete got left behind.

    The class can also be used for files intended to be persistent, by
    ensuring the file gets deleted if any error happened while creating
    it. Call disarm() only when you are sure you want to keep it.
    At that point your file is no longer considered temporary.

    Init takes a "silent" parameter which is only intended for unit tests.
    """
    def __init__(self, name, iocontext = None, *, armed = True, silent = False):
        self._name = name
        self._iocontext = iocontext
        self._silent = silent
        self._armed = armed

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._armed and self._name:
            self.remove(self._name, self._iocontext,
                        raise_on_error = False if type else True,
                        silent = self._silent)
            self._armed = False
            #print("AUTO-DELETE", self._name)
        #else:
        #    print("AUTO-DELETE was disarmed.")

    def disarm(self):
        """
        Tell the instance to not try to delete the file. Either because we
        deleted it ourselves, or because we for some reason decided to keep it,
        or we want to guard agaist exceptions during create by temporarily
        disarming and then arming the file.
        """
        #print("DISARM", self._name)
        self._armed = False

    def arm(self):
        """
        Tell the instance to try to delete the file on exit.
        """
        #print("ARM", self._name)
        self._armed = True

    @property
    def name(self):
        return self._name

    @staticmethod
    def remove(name, iocontext = None, raise_on_error = True, silent = False):
        """
        Delete a file locally or in seismic store.
        Not using openzgy.api.ZgyUtils.delete() because that method
        will always suppress "file not found" errors. Please Don't do this
        unless you are ok with the explicit sdapi dependency.
        """
        credentials = ("", "", "")
        if iocontext:
            try:
                credentials = (iocontext["sdurl"],
                               iocontext["sdapikey"],
                               iocontext["sdtoken"])
            except (TypeError, KeyError):
                credentials = (iocontext.sdurl,
                               iocontext.sdapikey,
                               iocontext.sdtoken)
        if name:
            try:
                if name and name[:5] == "sd://":
                    if sd is not None:
                        with sd.SdUtil(credentials) as u:
                            u.delete(name)
                else:
                    os.remove(name)
            except Exception as ex:
                if not silent:
                    print("WARNING: Cannot AUTO-DELETE", name, str(type(ex)), str(ex))
                if raise_on_error:
                    raise

    @staticmethod
    def _randomname():
        try:
            with open("/dev/urandom", "rb") as random:
                n = np.frombuffer(random.read(4), dtype=np.uint32)[0]
        except FileNotFoundError: # Probably on a windows box.
            n =random.randint(0, 0xffffffff)
        return "{0:08x}-".format(n)

class LocalFileAutoDelete(TempFileAutoDelete):
    """
    As TempFileAutoDelete, but explicitly requesting a local file with
    a random prefix in from of the name.
    """
    _prefix = "tmp-{0:08x}-".format(int(time.time()))
    _prefix = os.path.join(os.getenv("TESTRUNDIR", '.'), _prefix)
    def __init__(self, suffix, *, silent = False):
        name = self._prefix + self._randomname() + suffix
        super().__init__(name, None, silent=silent)

class CloudFileAutoDelete(TempFileAutoDelete):
    """
    As TempFileAutoDelete, but explicitly requesting a file on the
    seismic store with a random prefix in from of the name.
    """
    _prefix = SDTestSink(None)
    _prefix += "/tmp-{0:08x}-".format(int(time.time()))
    def __init__(self, suffix, iocontext,*, silent = False):
        name = self._prefix + self._randomname() + (suffix or "tmp.tmp")
        super().__init__(name, iocontext, silent=silent)

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
