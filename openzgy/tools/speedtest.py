#!/usr/bin/env python3

"""
Measure compression- and decompression speed for data already in memory.
See zfp-speedtest.sh for usage.
Also reports on requested vs. achieved SNR. Does not store the output data.
To visually inspect the compression quality see zfp-comparison.sh.
"""

import numpy as np
import os, sys, math, time, argparse
from zfpy import compress_numpy as new_compress
from zfpy import decompress_numpy as new_decompress
from zgycompress import compress as old_compress_
from zgycompress import inflate as old_decompress_
from ..api import ZgyReader, ZgyWriter, SampleDataType, ProgressWithDots
from ..test.utils import SDCredentials

# TODO-Low consider using iterator.readall. But, is test_zfp still useful?

global_args = None

def old_compress(data, sqnr):
    assert data.shape == (64, 64, 64)
    if global_args.buggy:
        # We will pass our brick indexed il, xl, z which is wrong
        # since the old compressor wants z varying slowest.
        return old_compress_(data, sqnr)
    else:
        # Explicitly make the buffer fortran contiguous.
        # This is probably simpler than reversing the axis.
        data = np.copy(data, order="F")
        return old_compress_(data, sqnr)

def old_decompress(cdata):
    bricksize = (64, 64, 64) # KLUDGE
    tmp = old_decompress_(cdata, bricksize)
    if global_args.buggy:
        return np.frombuffer(tmp, dtype=np.float32).reshape(bricksize)
    else:
        return np.frombuffer(tmp, dtype=np.float32).reshape(bricksize, order="F")

def snr(s, n):
    return 99 if n == 0 else 20 * math.log10(s / n)

def calc_snr(data, ddata):
    signal = np.sum(np.abs(data), dtype=np.float64)
    noise  = np.sum(np.abs(data - ddata), dtype=np.float64)
    return snr(signal, noise)

def calc_snr_from_list(data, ddata):
    signal = 0
    noise = 0
    for ii in range(len(data)):
        signal += np.sum(np.abs(data[ii]), dtype=np.float64)
        noise  += np.sum(np.abs(data[ii] - ddata[ii]), dtype=np.float64)
    return snr(signal, noise)

def zfp_parameters(want_snr, datarange, dtype):
    """
    Express the Petrel style "requested signal/noise ratio"
    in terms of ZFP configuration parameters.
    """
    if datarange == 0 or not np.isfinite(datarange) or want_snr > 70:
        return dict()
    else:
        tolerance = datarange / math.pow(10, (want_snr-8)/20)
        return { "tolerance": tolerance }

def slurp_open_file(r, progress):
    def _roundup(x, step): return ((x + step - 1) // step) * step
    def _rounddown(x, step): return (x // step) * step
    bricksize = tuple(r.bricksize)
    # Since I am testing compression, don't confuse matters
    # by introducing partial bricks.
    blocksize = (bricksize[0],
                 _rounddown(r.size[1], r.bricksize[1]),
                 _rounddown(r.size[2], r.bricksize[2]))
    total = r.size[0] // blocksize[0]
    # No more than 1024 MVoxel to avoid running out of memory (and time).
    # TODO-Low, would be nice to crop both inline and crossline range
    # to have less likelyhood of dead traces.
    print("BS", np.product(blocksize) // (1024*1024), "MB, count", total)
    have_room = 1024*1024*1024 // np.product(blocksize)
    new_total = min(total, max(have_room, 1))
    first_il = ((total - new_total) // 2) * blocksize[0]
    print("Adjust il count from start 0 count {0} to start {1} count {2}".format(total, first_il, new_total))
    total = new_total
    done = 0
    blocks = []
    data = np.zeros(blocksize, dtype=np.float32)
    for ii in range(first_il, total*blocksize[0] + first_il, blocksize[0]):
        r.read((ii, 0, 0), data)
        for jj in range(0, _rounddown(r.size[1], bricksize[1]), bricksize[1]):
            for kk in range(0, _rounddown(r.size[2], bricksize[2]), bricksize[2]):
                blocks.append(np.copy(data[:,jj:jj+bricksize[1],kk:kk+bricksize[2]]))
        done += 1
        if progress: progress(done, total)
    return blocks

def compress_all(blocks, progress):
    cblocks = []
    done = 0
    total = len(blocks)
    for data in blocks:
        if global_args.old:
            cblocks.append(old_compress(data, global_args.sqnr))
        else:
            datarange = np.amax(data) - np.amin(data) if np.all(np.isfinite(data)) else np.nan
            zfp_kwargs = zfp_parameters(global_args.sqnr, datarange, data.dtype)
            cblocks.append(new_compress(data, **zfp_kwargs))
        done += 1
        if progress: progress(done, total)
    return cblocks

def uncompress_all(cblocks, progress):
    result = []
    done = 0
    total = len(cblocks)
    for cdata in cblocks:
        if global_args.old:
            result.append(old_decompress(cdata))
        else:
            result.append(new_decompress(cdata))
        done += 1
        if progress: progress(done, total)
    return result

def slurp(filename):
    starttime = time.time()
    with ZgyReader(filename, iocontext = SDCredentials()) as r:
        blocks = slurp_open_file(r, progress=ProgressWithDots())
        bricksize_in_bytes = int(np.product(r.bricksize) * 4)
    r_time = time.time() - starttime

    for b in blocks:
        assert b.size * b.itemsize == bricksize_in_bytes

    count1 = len(blocks)
    len1 = 4 * np.sum([len(e.flat) for e in blocks], dtype=np.int64)

    starttime = time.time()
    cblocks = compress_all(blocks, progress=ProgressWithDots())
    c_time = time.time() - starttime

    count2 = len(cblocks)
    len2 = np.sum([len(e) for e in cblocks], dtype=np.int64)
    #blocks = None

    starttime = time.time()
    for i in range(global_args.repeat):
        ublocks = uncompress_all(cblocks, progress=ProgressWithDots())
    u_time = (time.time() - starttime) // global_args.repeat

    count3 = len(ublocks)
    len3 = 4 * np.sum([len(e.flat) for e in ublocks], dtype=np.int64)

    assert count1 == count2
    assert count2 == count3
    assert len1 == len3
    for b in ublocks:
        assert 4*len(b.flat) == bricksize_in_bytes
    assert count1 * bricksize_in_bytes == len1

    MB = 1024*1024
    r_rate = (len1 / r_time) / MB if r_time else 0
    c_rate = (len1 / c_time) / MB if c_time else 0
    u_rate = (len1 / u_time) / MB if u_time else 0


    # Now measure the signal to noise ratio:
    snrlist = [calc_snr(blocks[ii], ublocks[ii]) for ii in range(count1)]
    snrlist = list(sorted(snrlist))
    total_snr = calc_snr_from_list(blocks, ublocks)

    info = {
        "name": filename,
        "brickcount": count1,
        "bricksize": bricksize_in_bytes / MB,
        "compressor": ("ZGY" if global_args.old else "ZFP"),
        "r_time": r_time,
        "c_time": c_time,
        "u_time": u_time,
        "r_rate": r_rate,
        "c_rate": c_rate,
        "u_rate": u_rate,
        "u_mb": len1 / MB,
        "c_mb": len2 / MB,
        "c_ratio": len1 / len2,
        "snr_ask": global_args.sqnr,
        "snr_min": snrlist[0],
        "snr_max": snrlist[-1],
        "snr_avg": np.sum(snrlist)/len(snrlist),
        "snr_median": snrlist[len(snrlist)//2],
        "snr_total": total_snr,
    }

    if global_args.csv:
        hdr_format = "!!;Name;Brick count;Brick size (MB);Compressor;Read (sec);Compress (sec);Inflate(sec);Read (MB/s);Compress (MB/s);Inflate(MB/s);Uncompressed (MB);Compressed (MB);Compression Ratio;Requested SNR;Min SNR;Max SNR;Avg SNR;Median SNR;Overall SNR" if global_args.csvheader else None
        data_format = "!!;{name};{brickcount};{bricksize:.0f};{compressor};{r_time:.2f};{c_time:.2f};{u_time:.2f};{r_rate:.2f};{c_rate:.2f};{u_rate:.2f};{u_mb:.0f};{c_mb:.3f};{c_ratio:.2f};{snr_ask:.0f};{snr_min:.2f};{snr_max:.2f};{snr_avg:.2f};{snr_median:.2f};{snr_total:.2f}"
    else:
        hdr_format = None
        data_format = """
Processed {brickcount} bricks of {bricksize} MB using {compressor} compressor
Elapsed (sec): read {r_time:6.2f} compress {c_time:6.2f} inflate {u_time:6.2f}
Rates  (MB/s): read {r_rate:6.2f} compress {c_rate:6.2f} inflate {u_rate:6.2f}
Compression ratio: {u_mb:.0f} MB / {c_mb:.3f} MB = {c_ratio:.2f}
Signal/Noise ratio: asked {snr_ask} min {snr_min:.2f} max {snr_max:.2f} avg {snr_avg:.2f} median {snr_median:.2f} overall {snr_total:.2f}"""

    if hdr_format: print(hdr_format)
    print(data_format.format(**info))

    if global_args.hist and global_args.hist[0]:
        print('Saving histogram as "{0}"'.format(global_args.hist[0]))
        import matplotlib.pyplot as plt
        #hist, bins = np.histogram(snrlist, bins=99, range=(0,99))
        plt.hist(snrlist, bins=99, range=(0,99))
        plt.savefig(global_args.hist[0])

if __name__ == "__main__":
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description='ZFP performance test.')
    parser.add_argument('filename', help='ZGY cube to operate on')
    parser.add_argument('--csv', action='store_true', help="Machine readable results")
    parser.add_argument('--csvheader', action='store_true', help="Machine readable results")
    parser.add_argument('--old', action='store_true', help="Use the old ZGY compressor instead of zfp")
    parser.add_argument('--sqnr', default=30, type=float, help='Target SQNR')
    parser.add_argument('--repeat', default=1, type=int, help='Run the decompress step this many times')
    parser.add_argument('--histogram', dest='hist', nargs=1, default="", type=str, help='Save histogram of actual SQNR values to png file')
    parser.add_argument('--buggy', action='store_true', help="Compressor expects C ordering")
    global_args = parser.parse_args()
    print(global_args)
    slurp(global_args.filename)

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
