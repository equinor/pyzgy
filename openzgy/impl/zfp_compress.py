#!/usr/bin/env python3

##@package openzgy.impl.zfp_compress

import math
import numpy as np
import time
import sys

from ..impl import enum as impl_enum
from ..exception import *
from ..impl.compress import CompressFactoryImpl, CompressPlugin, CompressStats, CompressionResult

try:
    zfpy_ok = False
    from zfpy import compress_numpy as zfpy_compress_numpy
    from zfpy import decompress_numpy as zfpy_decompress_numpy
    zfpy_ok = True
except Exception as ex:
    print("Warning: ZFP compression is not available:", str(ex), file=sys.stderr)

class ZfpCompressPlugin(CompressPlugin):
    """
    Implement ZFP compression. See the CompressPlugin base type
    for details. Most methods are static or class methods.
    The exception is __init__, __call__, and dump().
    """
    @staticmethod
    def compress(data, snr = 30, stats = None):
        if zfpy_ok:
            return ZfpCompressPlugin._zfp_compress(data, snr, stats)
        else:
            return None

    @staticmethod
    def decompress(cdata, status, shape, file_dtype, user_dtype):
        if zfpy_ok:
            return ZfpCompressPlugin._zfp_decompress(cdata, status, shape)
        else:
            return None

    @staticmethod
    def factory(snr = 30):
        return None if snr <= 0 else ZfpCompressPlugin(snr)

    def __init__(self, snr = 30):
        self._snr = snr
        self._details = "ZFP[target_snr={0:.1f}]".format(snr)
        self._stats = CompressStats(self._details)

    def __call__(self, data):
        return self.compress(data, self._snr, self._stats)

    def __str__(self):
        return self._details

    def dump(self, msg=None, *, outfile=None, text=True, csv=False, reset=True):
        if not self._stats.empty():
            self._stats.dump(msg, outfile=outfile, text=text, csv=csv)
        if reset:
            self._stats = CompressStats(self._details)

    @staticmethod
    def _zfp_precision_from_snr(data, want_snr):
        # Not sure I want this test...
        if want_snr < 10 or want_snr > 70:
            return 0, 0, 5

        if not np.issubdtype(data.dtype, np.integer):
            if not np.all(np.isfinite(data)):
                return 0, 0, 5 # Only lossless and no-compression handles this.

        # ZFP precision has a valid range of 1 to 64 bit planes.
        # If compression is done simply by reducing the number of bits:
        # If signal uses 8 bits (-128..+127) the average magnitude of the
        # signal would be 64 and the acerage quantization noise 0.5.
        # So SNR with the current metric would be 20*log10(128) ~= 42 dB.
        # Assume ZFP is better than that if asked to use N bit planes:
        # 48 dB at 8 bits, or simply 6 dB per bit.
        # A heuristic taken from hca_filt.zgy gets a somewhat different
        # result, but is quite accurate for that data set.
        # TODO-Test: test on other data sets; test on int8 and int16 as well.
        #return (want_snr + 3) // 6  if want_snr >= 10 and want_snr <= 70 else 0
        precision = (int(want_snr) + 23) // 5
        snr_rounded = (precision * 5) - 20
        return precision, snr_rounded, 5

    @classmethod
    def _zfp_try_one_lossless(cls, idata, file_dtype, snr_wanted):
        """
        Lossless compression of the input. snr_wanted is only used for
        logging since the actual snr will be perfect.
        """
        ctime = None
        dtime = None
        ddata = idata
        starttime = time.perf_counter()
        cdata = zfpy_compress_numpy(idata.astype(np.float32, copy=False))
        if True: # For testing only, for logging compression and time used.
            ctime = time.perf_counter() - starttime
            ddata = zfpy_decompress_numpy(cdata).astype(idata.dtype)
            dtime = time.perf_counter() - starttime - ctime
            if not np.allclose(idata, ddata, equal_nan=True):
                raise InternalError("zfp: claims to be lossless but isn't.")
        isize = idata.size * file_dtype.itemsize
        csize = len(cdata)
        # Noise should be 0 and snr 99, but I go thru the motions
        # to get accurate statistics. "signal" gets logged.
        if np.all(np.isfinite(idata)):
            signal, noise = CompressStats._collect_snr(idata, ddata)
        else:
            signal, noise = (0, 0)
        snr_result = CompressStats._compute_snr(signal, noise)
        return CompressionResult(cdata       = cdata,
                                 csize       = csize,
                                 isize       = isize,
                                 signal      = signal,
                                 noise       = noise,
                                 ctime       = ctime,
                                 dtime       = dtime,
                                 snr_result  = snr_result,
                                 snr_wanted  = snr_wanted,
                                 snr_rounded = 99,
                                 snr_step    = 0)

    @classmethod
    def _zfp_try_one_precision(cls, idata, file_dtype, snr_wanted):
        # the next method checks for non-finite numbers.
        precision, snr_rounded, snr_step = cls._zfp_precision_from_snr(idata, snr_wanted)
        if precision:
            starttime = time.perf_counter()
            cdata = zfpy_compress_numpy(idata, precision=precision)
            ctime = time.perf_counter() - starttime
            ddata = zfpy_decompress_numpy(cdata)
            dtime = time.perf_counter() - starttime - ctime
            isize = idata.size * file_dtype.itemsize
            csize = len(cdata)
            signal, noise = CompressStats._collect_snr(idata, ddata)
            snr_result = CompressStats._compute_snr(signal, noise)
            return CompressionResult(cdata       = cdata,
                                     csize       = csize,
                                     isize       = isize,
                                     signal      = signal,
                                     noise       = noise,
                                     ctime       = ctime,
                                     dtime       = dtime,
                                     snr_result  = snr_result,
                                     snr_wanted  = snr_wanted,
                                     snr_rounded = snr_rounded,
                                     snr_step    = snr_step)
        else:
            return CompressionResult(*((None,)*11))

    @classmethod
    def _zfp_compress_precision(cls, idata, file_dtype, want_snr, stats):
        r = cls._zfp_try_one_precision(idata, file_dtype, want_snr)
        if r.cdata is None or r.csize > 0.9*r.isize:
            r = cls._zfp_try_one_lossless(idata, file_dtype, want_snr)
        if r.cdata is not None:
            # need to make sure the compressed size is not larger than
            # the input. Note that r.isize needs to be computed as the size
            # the brick has on disk. If idata has been converted from int8
            # to int32 or float32 it is still the size on disk that matters.
            # The test below should hardly ever fail for int16 or float data
            # but for int8 it is a real possibility.
            if r.csize <= 0.9 * r.isize:
                stats.add(signal=r.signal, noise=r.noise, snr=r.snr_result,
                          isize=r.isize, csize=r.csize,
                          ctime=r.ctime, dtime=r.dtime,
                          msg="forced")
                return r.cdata
        return None

    @classmethod
    def _zfp_compress(cls, data, want_snr = 30, stats = None):
        """
        Compression plug-in offering ZFP compression of data bricks.

        The function can be passed to _writeRegion and _writeAllLODs
        as-is but will in that case have a hard coded snr of 30.
        And no compression statistics will be recorded.
        To be able to change the snr use something like
            lambda x, snr=snr: _zfp_compress(x, snr)
        which will capture your local "snr" variable.

        The input is a 3d or (TODO-Low 2d) numpy array and the output is bytes.
        ZFP or whatever algorithm is used is assumed to handle big / little
        endian conversion itself. TODO-Worry this is not quite true for ZFP.
        See the documentation. A special compilation flag is needed
        on bug endian machines. Also I suspect the optional hedaer
        (which this code uses) might need byte swapping.
        """
        # If forced inside test_compress.run_file, make sure we are really just used to handle float data.
        #assert data.dtype == np.float32

        if want_snr < 0 or not zfpy_ok:
            return None # will end up uncompressed.

        # Note, the api currently enforces file_dtype == np.float32,
        # to avoid having the user shoot himself in the foot.

        file_dtype = data.dtype
        data = data.astype(np.float32, copy=False)

        cdata = cls._zfp_compress_precision(data, file_dtype, want_snr, stats)
        if cdata is not None: return cdata

        # Give up. Do not compress this brick.
        # Not sure whether I want to count this brick in the statistics.
        # Definitely don't include ctime and dtime; timing here is irrelevant.
        if stats:
            isize = data.size * data.itemsize
            signal, noise = CompressStats._collect_snr(data, data)
            stats.add(signal=signal, noise=noise, snr=99,
                      isize=isize, csize=isize, msg="noncompr")
        return None

    @classmethod
    def _zfp_decompress(cls, cdata, status, shape):
        """
        Decompress data produced by _zfp_compress.
        ZFP will return an ndarray and it is assumed that any byte
        swapping has already been taken care of. ZFP encodes size in
        its own header so we ignore what is passed by the caller. ZFP
        also encodes dtype, but doesn't recognize int8 or int16 so
        thise will show up as float or int32 and must be converted.

        We need to be told which data type the caller eventually wants,
        because the file might contain integral data encoded as float.
        If the caller wants float data it would add more noise if we
        temporarily convert it to int here.

        See CompressPlugin.decompress for the argument list etc.
        """
        if len(cdata) < 4 or cdata[:3] != bytes("zfp", "ASCII"):
            return None # Not ours.
        # Might be e.g. a memoryview if reading from cloud.
        if not cdata is bytes: cdata = bytes(cdata)
        ddata = zfpy_decompress_numpy(cdata)
        #print("Inflated", len(cdata), "->", ddata.size*ddata.itemsize)
        return ddata

# Add to list of known factories.
if zfpy_ok:
    CompressFactoryImpl.registerCompressor("ZFP", ZfpCompressPlugin.factory)
    CompressFactoryImpl.registerDecompressor("ZFP", ZfpCompressPlugin.decompress)

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
