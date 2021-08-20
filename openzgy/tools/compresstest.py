#!/usr/bin/env python3

"""
Measure compression quality and compression factor.

Unlike test_zfp this does not try to measure performance.
The focus is to exercise higher levels of code (CompressPlugin)
and to be able to easily compare the effects of compression
parameters. Possibly by doing multiple compressions of each block.

Caveat: Since the code does both compress and decompress internally,
it might in principle "cheat" and get better results that way.
I have other tests that actually write out a compressed file and
reads it back in, and those seem to have similar results. So this
is probably ok.
"""

# Some things I would like to learn:
#
#   * Make a scatter plot of quality vs. compression and also the
#     compression- and decompression speed to evaluate which of ZFP
#     and the old ZGY compression is best.
#
#     Unsure whether I should use median or average, probably the latter.
#     Compare the old algorithm with ZFP. For ZFP I can also try
#     different strategies (precision vs. accuracy, int or float).
#     This currently requires tweaking the code in impl_zfp_compress;
#     maybe I'll add it as an argument to the compressor later.
#
#     Run several experiments, sort and load the results into a spreadsheet.
#     For each compression plug-in, plot the average or maybe the median
#     achieved snr agaist the achieved compression ratio.
#     Compare the plots for different algorithms to see which is better.
#     For a more accurate comparison it is possible to plot every
#     single brick's quality vs. compression. This requires some more code.
#     Possibly using matplotlib instead of relying on an external spreadsheet.
#     The reason I may want this extra detail is that neither the overall
#     (survey taken as a single brick) or average or median snr has a penalty
#     for having wildly varying quality between bricks. I could perhaps
#     measure e.g. the 20-percentile of the per block snr instead?
#     The most detailed information comes from plotting every single
#     data point.
#
#     May need to repeat this for several data sets, including several each
#     of float, int16, and int8 data. When preparing int cubes make sure
#     to clip outliers.
#
#     >>> Ongoing. In most cases Old ZGY compresses better but is slower.
#
#   * Measure the combined snr if a data set is first converted to int8
#     (or int16) and then compressed. Is the noise added in each step
#     independant?
#
#   * Accuracy of snr to zfp parameter heuristic.
#
#     To measure this I need to disable the goal seek algorithm.
#     Then run the tests and do a scatter plot of wanted vs. result snr.
#
#   * Which ZFP mode gives the best compression given an snr constraint?
#     And which is fastest for compress and fastest for decompress?
#
#       -- float data in "tolerance" mode [recommended in doc]
#       -- float data in "precision" mode
#       -- int data in "precision" mode [recommended in doc]
#       -- int data converted to float in "tolerance" mode
#       -- int data converted to float in "precision" mode
#
#      >>> Previous test results suggest the difference is not large.
#
#          The snr heuristic seems to work better in "precision" mode,
#          it might even be good enough to use this as our S/N metric
#          and skip the goal seek.
#
#          The two alternatives that convert int to float have an
#          additional caveat: If the app wants to read the data
#          as unscaled integers (to save memory) then there will be
#          an extra float -> int conversion that adds noise. So if
#          we choose any of those it might be better to lie when the
#          app asks for data type and claim it was float all along.
#
#   * Will ZFP give me better compression if I pass it 128^3 bricks?
#     >>> No. ZFP actually compresses each 4^3 block separately.
#
#   * Wish list: If the input is large, automatically crop out the
#     center and only analyze that. Or extract a random number of
#     brick-columns.

import numpy as np
import os, sys, math, time, argparse
from zgycompress import compress as old_compress_
from zgycompress import inflate as old_decompress_
from ..api import ZgyReader, ZgyWriter, ProgressWithDots, _map_SampleDataTypeToDataType
from ..impl.compress import CompressStats
from ..impl.zfp_compress import ZfpCompressPlugin
from ..impl import enum as impl_enum
from ..impl.bulk import ZgyInternalBulk
from ..iterator import readall
from ..test.utils import SDCredentials
from ..exception import *

global_args = None

def old_compress(data, sqnr):
    pad = np.array((64, 64, 64), dtype=np.int64) - np.array(data.shape)
    assert np.all(pad >= 0)
    if np.any(pad > 0):
        data = np.pad(data, ((0, pad[0]), (0, pad[1]), (0, pad[2])), 'edge')
    # The old compressor only handles float.
    data = data.astype(np.float32, copy=False)
    # Explicitly make the buffer fortran contiguous.
    # This is probably simpler than reversing the axis.
    data = np.copy(data, order="F")
    return old_compress_(data, sqnr)

def old_decompress(cdata):
    bricksize = (64, 64, 64) # Only works with this bricksize.
    tmp = old_decompress_(cdata, bricksize)
    return np.frombuffer(tmp, dtype=np.float32).reshape(bricksize, order="F")

class OldCompressPlugin:
    """
    Mock plugin, doesn't produce anything but does record the compression
    quality and compression rate for each brick.
    """
    @staticmethod
    def compress(idata, snr = 30, stats = None):
        # The old compressor only handles 64^3, so just log this as
        # being uncompressed. A brick size < 64^3 should work but we
        # end up padding every single brick and will probably get a
        # very bad compression rate.
        if np.any(np.array(idata.shape, dtype=np.int64) > 64):
            if stats:
                stats.add_data(idata, idata.size * idata.itemsize, idata)
                return
        starttime = time.perf_counter()
        cdata = old_compress(idata, snr)
        ctime = time.perf_counter() - starttime
        ddata = old_decompress(cdata)
        dtime = time.perf_counter() - starttime - ctime
        # Strip off any padding done on compress.
        # Note, this is a kludge. Normally we would have
        # the survey size available, so the old data need
        # not be present.
        s = idata.shape
        ddata = ddata[0:s[0], 0:s[1], 0:s[2]]
        if stats:
            stats.add_data(idata, len(cdata), ddata, ctime=ctime, dtime=dtime)

    def __init__(self, snr = 30):
        self._snr = snr
        self._details = "Old[target_snr={0:.1f}]".format(snr)
        self._stats = CompressStats(self._details)

    def __call__(self, data):
        return self.compress(data, self._snr, self._stats)

    def __str__(self):
        return self._details

    def dump(self, msg = None, *, outfile=None, text=True, csv=False, reset=True):
        self._stats.dump(msg, outfile=outfile, text=text, csv=csv)
        if reset:
            self._stats = CompressStats(details = self._details)

class CompressPluginPlain:
    """
    Mock plugin, doesn't produce anything but does record the compression
    quality for compression to 8 bit. Does not record timing statistics.
    """
    @staticmethod
    def compress(idata, wide = False, codingrange = (-1,1), stats = None):
        file_dtype = np.int16 if wide else np.int8
        if idata.dtype == np.float32:
            cdata = ZgyInternalBulk._scaleToStorage(idata, codingrange, file_dtype)
            ddata = ZgyInternalBulk._scaleToFloat(cdata, codingrange, file_dtype)
        elif idata.dtype == np.int8 or idata.dtype == file_dtype:
            cdata = idata.astype(file_dtype) # only needed for size
            ddata = idata # always perfect
        elif idata.dtype == np.int16 and file_dtype == np.int8:
            cdata = (idata // 256).astype(file_dtype)
            ddata = cdata.astype(idata.dtype) * 256
        else:
            assert False
        if stats:
            stats.add_data(idata, cdata.size * cdata.itemsize, ddata)
        global dejavu
    def __init__(self, codingrange = (-1,1), snr = 30):
        # TODO-Low, input arg "wide" instead of snr.
        self._codingrange = tuple(codingrange)
        self._wide = bool(snr <= 30)
        self._details = "int16" if self._wide else "int8"
        self._stats = CompressStats(self._details)

    def __call__(self, data):
        return self.compress(data, wide = self._wide, codingrange=self._codingrange, stats=self._stats)

    def __str__(self):
        return 'int16' if self._wide else 'int8'

    def dump(self, msg = None, *, outfile=None, text=True, csv=False, reset=True):
        self._stats.dump(msg, outfile=outfile, text=text, csv=csv)
        if reset:
            self._stats = CompressStats(details = self._details)

def find_center(r):
    """
    Crop partial bricks outside or below the survey since they
    would just confuse the statistics.
    If the file is large then crop out an area around the center.
    Try to read the full traces but limit the il and xl.
    """
    size = np.array(r.size, dtype=np.int64)
    cropsize = np.minimum(size, np.array([640,640,12800], dtype=np.int64))
    cropsize = (cropsize//64)*64
    cropoffset = ((size - cropsize)//128)*64
    if np.any(size - cropsize >= 64):
        print("Reading center", tuple(cropsize),
              "offset", tuple(cropoffset),
              "of survey size", tuple(r.size))
    return cropsize, cropoffset

def run_file(filename, logfilename):
    print("{0} -> {1}".format(filename, logfilename))
    with ZgyReader(filename, iocontext = SDCredentials()) as r:
        cropsize, cropoffset = find_center(r)
        compressors = []
        faclist = [ZfpCompressPlugin, OldCompressPlugin, CompressPluginPlain]
        snrlist = [ 20, 40, 30, 50, 10, 70, 60, ]
        snrlist.sort() # in batch mode, this is preferable.
        for factory in faclist:
            for snr in snrlist:
                compressors.append(factory(snr=snr))
                # Mega-kludge!
                if hasattr(compressors[-1], "_codingrange"):
                    compressors[-1]._codingrange = r.datarange

        processed = [0] * len(compressors)
        elapsed = [0] * len(compressors)
        progress = ProgressWithDots()
        chunksize = np.array((64, 64, 64), dtype=np.int64)
        # Set different from 64^3 to test different sizes in ZFP.
        # If selecting a block size > 64, the blocksize for reading
        # needs to be adjusted. Probably just set it to (N, N, 0).
        # Otherwise we might not get any full-chunksize bricks at all.
        # Similarly adjusted the cropping to avoid partial bricks.
        print("." * 51, file=sys.stderr, flush=True)
        # @@@ use dtype=np.float32 to convert everything to float first.
        for datastart, datasize, data in readall(r, chunksize=chunksize, progress=progress, maxbytes=128*1024*1024, cropsize=cropsize, cropoffset=cropoffset):
            if not np.all(datasize == chunksize):
                print("Oops: datasize", datasize, "chunksize", chunksize)
            assert np.all(datasize == chunksize) # Because file is cropped
            cix = 0
            for c in compressors:
                starttime = time.perf_counter()
                c(data)
                this_elapsed = time.perf_counter() - starttime
                elapsed[cix] += this_elapsed
                processed[cix] += data.size * data.itemsize
                cix += 1
                buffer_dtype = data.flat[0].dtype

        all_info = []
        with open(logfilename, "w") as csvfile:

            print(filename, file=csvfile)
            print("TABLE 1", file=csvfile)
            # This table measures a single time for compress and decompress
            # so it is not very useful except for validating table 2.
            # The overhead of goal seek is included both for old and new
            # algorithm.

            for cix in range(len(compressors)):
                if cix==0 or type(compressors[cix]) != type(compressors[cix-1]):
                    print("\n;Name;snr;size%;c+d MB/s;MB data;Elapsed", file=csvfile)
                s = compressors[cix]._stats # Gross violation of encapsulation.
                print(";{0};{1:.1f};{2:.1f};{3:.5g};{4:.5g};{5:.5g}".format(
                    str(compressors[cix]),
                    s.snr(),
                    100 * s._packed / max(1, s._original),
                    (processed[cix]/elapsed[cix])/(1024*1024),
                    processed[cix]/(1024*1024),
                    elapsed[cix]), file=csvfile)

            print("", file=csvfile)
            print("\nTABLE 2", file=csvfile)
            # This table measures compress- and decompress times separately.
            # TODO-Low: old algorithm includes goal seek overhead, while the
            # new one does not.

            for cix in range(len(compressors)):
                c = compressors[cix]
                if cix==0 or type(compressors[cix]) != type(compressors[cix-1]):
                    c.dump("", text=False, csv="header", reset=False, outfile=csvfile)
                all_info.append(c._stats._all_info)
                c.dump(str(c), text=False, csv=True, reset=False, outfile=csvfile)
                c.dump(str(c))

            if True:
                data = np.array(all_info, dtype=np.float32)
                data = data.reshape(len(faclist), len(snrlist), -1, 2)
                np.save(logfilename + ".npy", data)

            if False:
                s = data.shape
                data = data.reshape(s[0], s[1]*s[2], s[3])
                print(buffer_dtype, r.datatype.name, file=csvfile, sep="/")
                for c in faclist:
                    print(";SNR;{0} size%".format(str(c(snr=0))), file=csvfile, end="")
                print(file=csvfile)
                for pair in range(data.shape[1]):
                    for algo in range(data.shape[0]):
                        for value in range(data.shape[2]):
                            print(";{0:.1f}".format(data[algo,pair,value]), file=csvfile, end="")
                    print(file=csvfile)

if __name__ == "__main__":
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description='ZFP performance test.')
    parser.add_argument('filename', help='ZGY cube to operate on')
    parser.add_argument('--csv', default='TC.csv', help='Output easy to parse result')
    global_args = parser.parse_args()
    run_file(global_args.filename, global_args.csv)

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
