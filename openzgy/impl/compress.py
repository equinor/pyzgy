#!/usr/bin/env python3

##@package openzgy.impl.compress

import math
import sys
import numpy as np
from collections import namedtuple

from ..impl import enum as impl_enum
from ..exception import *

CompressionResult = namedtuple("Compression", "cdata csize isize signal noise ctime dtime snr_result snr_wanted snr_rounded snr_step")
CompressionResult.__doc__ = """The result returned from _xxx_try_one() functions.
The first member, cdata, is the one that really matters.
The snr_xxx members may be useful for goal seek.
All members except cdata may be useful logging and testing.
"""
CompressionResult.cdata.__doc__ = "bytes-like compressed data"
CompressionResult.csize.__doc__ = "Size in bytes of the compressed data"
CompressionResult.isize.__doc__ = "Size in bytes of the original input data"
CompressionResult.signal.__doc__ = "Sum of all sample values in the input"
CompressionResult.noise.__doc__ = "Sum of all absolute errors in round trip"
CompressionResult.ctime.__doc__ = "Time in seconds used for compression"
CompressionResult.dtime.__doc__ = "Time in seconds used for decompression"
CompressionResult.snr_result.__doc__ = "The SNR that was achieved"
CompressionResult.snr_wanted.__doc__ = "The SNR that was requested by the user"
CompressionResult.snr_rounded.__doc__ = "The SNR that was requested from the algorithm"
CompressionResult.snr_result.__doc__ = "Distance to next or previous quality level"

class CompressPlugin:
    """
    Base class for OpenZGY compression plug-ins.
    If anybody wants to add additional compression algorithms it
    is recommended but not required to use this base class. See
    CompressFactoryImpl.register{Compressor,Decompressor} for how to
    use plain functors (C++) or callables (Python) instead.

    This class performs triple duty as it handles both compression
    and decompression static methods (need not have been together)
    and an instance of the class can be used a compressor functor
    if a lambda is too limiting. To invoke the methods:

        MyCompressPlugin.factory(...)(data)
        MyCompressPlugin.compress(data, ...) (NOT recommended)
        MyCompressPlugin.decompress(cdata,status,shape,file_dtype,user_dtype)

    The following will also work but should only be used for very simple
    compressors that have no parameters. In the first case MyCompressPlugin
    won't have the option to return None for certain parameters, and in the
    second case handling a variable arguent list becomes trickier.

    To register this class:
      CompressFactoryImpl.registerCompressor("My",MyCompressPlugin.factory)
      CompressFactoryImpl.registerDecompressor("My",MyCompressPlugin.decompress)

    To use the compression part from client code:
       compressor = ZgyCompressFactory("My", ...)
    """
    @staticmethod
    def compress(data, *args, **kwargs):
        """
        This is an abstract method.

        Compress a 3d or (TODO-Low 2d) numpy array, returning a bytes-like
        result. If called with a single "data" argument the compression
        will be done with default parameters and no extended logging.

        Additional arguments are specific to the compression type.

        The function can be used directly as the compression hook.
        But you probably want a lambda expression or a real instance
        of this class instead, to be able to specify parameters.

        The compression algorithm is used is assumed to handle big / little
        endian conversion itself. TODO-Worry this is not quite true for ZFP.
        See the documentation. A special compilation flag is needed
        on big endian machines. Also I suspect the optional hedaer
        (which this code uses) might need byte swapping.
        """
        raise ZgyInternalError("Attempt to invoke abstract method")

    @staticmethod
    def decompress(cdata, status, shape, file_dtype, user_dtype):
        """
        This is an abstract method.

        Decompress bytes or similar into a numpy.ndarray.

        Arguments:
          cdata      -- bytes or bytes-like compressed data,
                        possibly with trailing garbage.
          status     -- Currently always BrickStatus.Compressed,
                        in the future the status might be used to
                        distinguish between different compression
                        algorithms instead of relying on magic numbers.
          shape      -- Rank and size of the result in case this is
                        not encoded by the compression algorithm.
          file_dtype -- Original value type before compression,
                        in case the decompressor cannot figure it out.
                        This will exactly match the dtype of the
                        data buffer passed to the compressor.
          user_dtype -- Required value type of returned array.

        Passing an uncompressed brick to this function is an error.
        We don't have enough context to handle uncompressed bricks
        that might require byteswapping and fix for legacy quirks.
        Also cannot handle constant bricks, missing bricks, etc.

        The reason user_dtype is needed is to avoid additional
        quantization noise when the user requests integer compressed data
        to be read as float. the decompressor might need to convert
        float data to int, only to have it converted back to float later.

        Current assumptions made of all candidate algorithms:

            -  The compressed data stream may have trailing garbage;
               this will be silently ignored by the decompressor.

            -  The compressed data stream will never be longer than
               the uncompressed data. This needs to be enforced by
               the compressor. The compressor is allowed to give up
               and tell the caller to not compress this brick.

            -  The reason for the two assumptions above is an
               implementation detail; the reported size of a
               compressed brick is not completely reliable.
               This might change in the next version

            -  The compressed data stream must start with a magic
               number so the decompressor can figure out whether
               this is the correct algorithm to use.

        If the assumptions cannot be met, the compressor / decompressor
        for this particular type could be modified to add an extra header
        with the compressed size and a magic number. Or we might add a
        (size, algorithm number) header to every compressed block to
        relieve the specific compressor / decompressor from worrying
        about this. Or the brick status could be used to encode which
        algorithm was used, picked up from the MSB of the lup entry.
        Which would also require the compressor to return both the
        actual compressed data and the code to identify the decompressor.
        That is the main reason we are also passed the "status" arg.
        Caveat: If adding an extra header, keep in mind that this header
        must be included when checking that the compressed stream is not
        too big.
        """
        raise ZgyInternalError("Attempt to invoke abstract method")

    def __init__(self, *args, **kwargs):
        """
        Create an instance that remembers the arguments it was created with.
        When the instance is called as a function it will invoke compress()
        with those arguments. So you can use either of the following:

            compressor = CompressPlugin.compress # no arguments
            compressor = CompressPlugin(...)
            compressor = lambda x: CompressPlugin.compress(x, ...)

        Derived classes don't need to redefine __init__ and __call__.
        But they might want to in order to get argument checking.
        The __init__ in the base class accepts any arguments so an error
        won't be caught until the first time the compressor is invoked.
        """
        self._args = args
        self._kwargs = kwargs

    def __call__(self, data):
        """
        Invoke the compressor with arguments passed by the constructor.
        """
        return self.compress(data, *args, **kwargs)

    def dump(*args, **kwargs):
        """
        Output statistics to standard output, if possible.
        """
        pass

class CompressFactoryImpl:
    """
    Registry of known compress and decompress algorithms.
    Those two are completely separate but we might as well
    handle both in the same class.
    """
    _compress_registry = {}
    _decompress_registry = []

    @staticmethod
    def registerCompressor(name, fn):
        """
        Register a factory function that will be called to create
        a function that in turn can be used to compress a data block.
        Pass fn = None if you for some reason need to remove a registration.

        The registered function can have any signature; the signature
        needs to include whatever parameters the actual compressor wants.

        The function that is created by the factory must have the signature:
            raw:          bytes.
            brickstatus:  impl.enum.BrickStatus,
            bricksize:    tuple(int,int,int),
            file_dtype:   np.dtype,
            result_dtype: np.dtype

        The function's return value:
            np.ndarray with rank 3, shape bricksize, and dtype result_dtype.

        Example usage of the factory:
            old: with ZgyWriter(snr=snr)
            new: with ZgyWriter(compressor = ZgyCompressFactory("ZFP", snr=30),

        The example shows that this is a bit more inconvenient for the end
        user. But it allows for using different compression plug-ins with
        arbitrary parameters.

        Note that user code doesn't need to use ZgyCompressFactory() and
        its list of known compressors. Instead a compression function
        can be provided directly. But most likely the factory method will
        be simpler to maintain.

        fn() is allowed to return None, which will have the same effect
        as if the user did not specify any compression. E.g. there might be
        a convention that snr<0 means store uncompressed. Allowing the
        factory to return None in that case means the example above would
        still work. Otherwise the snr<0 test would be included in the
        client code. Which is messy.
        """
        CompressFactoryImpl._compress_registry[name] = fn

    @staticmethod
    def registerDecompressor(name, fn):
        """
        Register a factory function that is able to decompress one or more
        types of compressed data. The registered functions will be called
        in reverse order of registration until one of them indicates that
        it has decompressed the data. You cannot remove a registration
        but you can effectively disable it by registering another one
        that recognizes the same input data.
        The supplied name is only for information.

        The function that is created by the factory must have the signature:
            raw:          bytes.
            brickstatus:  impl.enum.BrickStatus,
            bricksize:    tuple(int,int,int),
            file_dtype:   np.dtype,
            result_dtype: np.dtype
        """
        CompressFactoryImpl._decompress_registry.insert(0, (name, fn))

    def knownCompressors():
        """
        Return the names of all compressors known to the system.
        This is primarily for logging, but might in principle be used
        in a GUI to present a list of compressors to choose from.
        The problem with that is how to handle the argument list.
        """
        return list([k for k, v in CompressFactoryImpl._compress_registry.items()])

    def knownDecompressors():
        """
        Return the names of all compressors known to the system.
        This is primarily for logging.
        """
        return list([k for k, v in CompressFactoryImpl._decompress_registry])

    @staticmethod
    def factory(name, *args, **kwargs):
        fn = CompressFactoryImpl._compress_registry.get(name, None)
        if not fn:
            known = ",".join(CompressFactoryImpl.knownDecompressors())
            raise ZgyMissingFeature('Compression algorithm "{0}" not recognized. Must be one of ({1}).'.format(name, known))
        return fn(*args, **kwargs)

    @staticmethod
    def decompress(cdata, status, shape, file_dtype, user_dtype):
        """
        Loop over all registered decompressors and try to find one that
        can handle this particular brick. Raises an error if none found.
        See CompressPlugin.decompress() for parameter descriptions.
        """
        result = None
        if status != impl_enum.BrickStatus.Compressed:
            raise ZgyInternalError("Tried to decompress uncompressed data.")
        for k, v in CompressFactoryImpl._decompress_registry:
            result = v(cdata, status, shape, file_dtype, user_dtype)
            if result is not None:
                break
        if result is None:
            raise ZgyFormatError("Compression algorithm not recognized.")
        elif tuple(result.shape) != tuple(shape):
            raise ZgyFormatError("Decompression returned unexpected data.")
        return result

class CompressStats:
    def __init__(self, details):
        self._details  = details # Parameters etc, just for logging.
        self._signal   = 0
        self._noise    = 0
        self._original = 0
        self._packed   = 0
        self._types    = dict()
        self._lossy    = 0
        self._perfect  = 0
        self._all_info = [] # (measured_snr, measured_compressed_size_%)
        self._timed_bytes = 0
        self._ctime = 0
        self._dtime = 0

    def add_data(self, idata, csize, ddata, *, ctime = None, dtime = None, msg = None):
        signal, noise = self._collect_snr(idata, ddata)
        isize = idata.size * idata.itemsize
        snr = self._compute_snr(signal, noise)
        self.add(signal=signal, noise=noise, snr=snr, isize=isize, csize=csize,
                 ctime=ctime, dtime=dtime, msg=msg)
        #print("signal {0:8.0f} noise {1:8.0f} snr {2:.2f}".format(signal, noise, snr))

    def add(self, signal, noise, snr, isize, csize, *, ctime = None, dtime = None, msg = None):
        self._all_info.append([snr, 100*csize/isize])
        if snr < 99:
            # s/n only logged for lossy compression.
            self._lossy    += 1
            self._signal   += signal
            self._noise    += noise
        else:
            self._perfect  += 1
        self._original += isize
        self._packed   += csize
        if ctime is not None and dtime is not None:
            self._timed_bytes += isize
            self._ctime += ctime
            self._dtime += dtime
        if msg == "compress" and snr >= 99: msg = "compres*"
        if msg:
            self._types[msg] = 1 + self._types.setdefault(msg, 0)
        if False and msg:
            print("@@ yoo-hoo {0} {1} -> {2} factor {3:.1f} snr {4:.1f}".format(
                msg, isize, csize, isize/csize if csize else np.nan, snr))

    @staticmethod
    def _collect_snr(idata, ddata):
        """
        This function along with _compute_snr defines the cost function
        used when trying to quantify noise. The input data should not
        contain NaN or Inf. We won't be applying lossy compression to
        that kind of data anyway.

        There is no perfect solution. We want to avoid having
        neighboring bricks showning significant differences in
        quality. This sounds easy enough, but "similar quality"
        depends very much on the cost function.
        """
        # Computationally inexpensive, but has problems with spikes.
        # A brick containing many spikes will appear to have a higher
        # average amplitude, which allows the compressor to leave
        # more noise in it. It would have been better to use the
        # average signal level of the entire survey instead of the
        # brick. But that information is not available yet.
        signal = np.sum(np.abs(idata), dtype=np.float64)
        noise  = np.sum(np.abs(idata - ddata), dtype=np.float64)
        # Expensive, and has problems with dead traces and bricks
        # that contain more than 50% water. The signal and noise
        # in thise cases would be measuring only the dead and/or
        # water samples.
        #signal = np.median(np.abs(idata)) * idata.size
        #noise  = np.median(np.abs(idata - ddata)) * idata.size
        return signal, noise

    @staticmethod
    def _compute_snr(s, n):
        return 99 if n == 0 else -10 if s/n < 0.315 else min(99, 6 * math.log2(s/n))

    def snr(self):
        return self._compute_snr(self._signal, self._noise)

    def dump(self, msg = None, *, outfile=None, text=True, csv=False):
        outfile = outfile or sys.stdout
        # Measured SNR for each brick, excluding lossless unless everything
        # is lossless. In which case we pretend there is just one brick.
        # The reason is that the median compression makes more sense
        # that way. If I had been plotting all percentiles instead of
        # just printing the median it would have made more sense to include
        # the lossless (SNR 99) bricks as well.
        all_snr = list([x[0] for x in self._all_info if x[0] < 99]) or [99.0]
        all_snr.sort()
        # Measured compression ratio for each brick, excluding uncompressed
        # unless everything is uncompressed. In which case we pretend there
        # is just one brick. Note that lossless can still be compressed.
        ratios = list([ 100.0/e[1] for e in self._all_info if e[1] != 1 and e[1] != 0 ]) or [1.0]
        ratios.sort()
        good = self._original and self._packed
        args = {
            "msg":          msg or "Processed",
            "details":      self._details,
            "orig_mb":      self._original / (1024*1024),
            "pack_mb":      self._packed,
            "factor":       self._original / self._packed if good else -1,
            "percent":      100 * self._packed / self._original if good else 1,
            "n_perfect":    self._perfect,
            "n_lossy":      self._lossy,
            "snr":          self.snr(),
            "snr_min":      all_snr[0],
            "snr_med":      all_snr[len(all_snr)//2],
            "snr_max":      all_snr[-1],
            "factor_min":   ratios[0],
            "factor_med":   ratios[len(ratios)//2],
            "factor_max":   ratios[-1],
            "percent_max":  100 / ratios[0],
            "percent_med":  100 / ratios[len(ratios)//2],
            "percent_min":  100 / ratios[-1],
            "ctime":        0 if not self._ctime else (self._timed_bytes / self._ctime) / (1024*1024),
            "dtime":        0 if not self._dtime else (self._timed_bytes / self._dtime) / (1024*1024),
            "_ctime": self._ctime,
            "_dtime": self._dtime,
            "_cplusdtime": self._ctime + self._dtime,
            "_timed_bytes": self._timed_bytes / (1024*1024),
        }
        if text:
            print("{msg} {orig_mb:.0f} MB, compression factor {factor:.1f} (median {factor_med:.1f} with {n_perfect} bricks lossless and {n_lossy} bricks min snr {snr_min:.1f} median {snr_med:.1f} overall {snr:.1f}) compress {ctime:.1f} MB/s decompress {dtime:.1f} MB/s".format(**args), file=outfile)
            for k in sorted(self._types):
                print("   ", k, self._types[k], file=outfile)
        if csv:
            if csv == "header":
                print(";msg;requested snr;median snr;snr;compressed size%;median size%;compress MB/s;decompress MB/s;compress elapsed;decompress elapsed;sum elapsed;Total data MB", file=outfile)
            else:
                print(";{msg};{details};{snr_med:.1f};{snr:.1f};{percent:.1f};{percent_med:.1f};{ctime:.1f};{dtime:.1f};{_ctime:.1f};{_dtime:.1f};{_cplusdtime:.1f};{_timed_bytes:.1f}".format(**args), file=outfile)

    def empty(self):
            return self._lossy + self._perfect == 0

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
