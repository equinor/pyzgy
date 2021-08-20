#!/usr/bin/env python3

"""
Implement a simple compare command, primarily to check a round trip of
compress and decompress.
"""

import numpy as np
import os, sys, argparse
import matplotlib.pyplot as plt
from ..api import ZgyReader, ZgyWriter, SampleDataType, ProgressWithDots
from ..impl.enum import BrickStatus
from ..impl.compress import CompressStats as Stats
from ..test.utils import SDCredentials
from ..iterator import readall

def compare_stats(r1, r2):
    old = r1.statistics
    new = r2.statistics
    old_avg = old.sum / old.cnt
    old_rms = float(np.sqrt(old.ssq / old.cnt))
    new_avg = new.sum / new.cnt
    new_rms = float(np.sqrt(new.ssq / new.cnt))
    # Note, use "rms" as the reference also for avg, as old_avg may be ~0.
    delta_avg = (new_avg - old_avg) / old_rms
    delta_rms = (new_rms - old_rms) / old_rms
    # Also check the value range.
    delta_min = abs(old.min - new.min) / (old.max - old.min)
    delta_max = abs(old.max - new.max) / (old.max - old.min)
    delta_range = max(delta_min, delta_max)
    print("in: ", old)
    print("out:", new)
    print('"Difference in stats" "avg:" {0:7.4f}% "rms:" {1:7.4f}% "range:" {2:7.4f}% "count:" {3}'.format(
        100*delta_avg, 100*delta_rms, 100*delta_range, new.cnt - old.cnt))

def compare_open_files(r1, r2, w, progress, histogram, label):
    def _roundup(x, step): return ((x + step - 1) // step) * step
    filesize1 = r1._fd.xx_eof
    filesize2 = r2._fd.xx_eof
    stats = Stats("MyStats")
    bricksize = np.minimum(np.array(r1.bricksize, dtype=np.int64),
                           np.array(r2.bricksize, dtype=np.int64))
    bufsize = np.zeros(3, dtype=np.int64)
    delta_histogram_clip = 0.01 # @@ change me!
    delta_histogram = None
    delta_histogram = np.zeros(200, dtype=np.int64)
    input_rms = np.sqrt(r1.statistics.ssq / r1.statistics.cnt)
    delta_hist_range = (-input_rms * delta_histogram_clip,
                        +input_rms * delta_histogram_clip)
    for start, count, _ in readall(r1, sizeonly=True):
        bufsize = np.maximum(bufsize, count)
    data1 = np.zeros(bufsize, dtype=np.float32)
    data2 = np.zeros(bufsize, dtype=np.float32)
    for start, count, _ in readall(r1, sizeonly=True, progress=progress):
        view1 = data1[:count[0],:count[1],:count[2]]
        view2 = data2[:count[0],:count[1],:count[2]]
        r1.read(start, view1)
        r2.read(start, view2)
        if w:
            w.write(start, view2 - view1)
        ddelta = (view2 - view1)
        np.clip(ddelta, delta_hist_range[0], delta_hist_range[1], out=ddelta)
        dhist = np.histogram(ddelta, bins=len(delta_histogram), range=delta_hist_range)
        delta_histogram += dhist[0]
        # readall() can't help with the inner loop because we had to
        # read from two files, not just one.
        for ii in range(0, count[0], bricksize[0]):
            for jj in range(0, count[1], bricksize[1]):
                for kk in range(0, count[2], bricksize[2]):
                    beg = np.array((ii, jj, kk), dtype=np.int64)
                    end = np.minimum(beg + bricksize, count)
                    bpos = beg // r2.bricksize
                    bstatus, _, _, bsize = r2._accessor._getBrickFilePosition(*bpos, 0)
                    d1 = view1[beg[0]:end[0],beg[1]:end[1],beg[2]:end[2]]
                    d2 = view2[beg[0]:end[0],beg[1]:end[1],beg[2]:end[2]]
                    if bstatus in (BrickStatus.Normal, BrickStatus.Compressed):
                        stats.add_data(d1, bsize, d2, msg=str(bstatus))
    MEGA = float(1024*1024)
    print("Actual file size {0:.0f} MB / {1:.0f} MB = ratio {2:.1f} relative {3:.1f}%".format(
          filesize1 / MEGA, filesize2 / MEGA, filesize1 / filesize2, 100 * (filesize2 / filesize1)))
    compare_stats(r1, r2)
    stats.dump()
    if histogram:
        #print(stats._all_info)
        # For the histogram I want to also include the lossless bricks;
        # unlike what is done in CompressStats.dump().
        allsnr = list([x[0] for x in stats._all_info]) or [99.0]
        allsnr = np.clip(allsnr, 0, 99)
        allrat = list([100.0/e[1] for e in stats._all_info if e[1] != 0]) or [1]
        allrat = np.clip(allrat, 0.8001, 99.9999)
        allrat = 100.0 / allrat # Now in range 1% .. 125%
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 9), dpi=100)
        fig.subplots_adjust(hspace=0.5)

        ax1.hist(allsnr, bins=100, range=(0, 100))
        #ax1.title.set_text("Signal to Noise (dB) for each brick.")
        #ax1.text(0.5, 0.9, 'Signal to Noise (dB) for each brick.',
        #         transform=ax1.transAxes, ha="center")
        ax1.set_xlabel('Measured signal to noise (dB)')
        ax1.set_ylabel('% bricks with this SNR')

        ax2.hist(allrat, bins=100, range=(1, 125))
        #ax2.title.set_text("Compression ratio for each brick.")
        #ax2.text(0.5, 0.9, 'Compression ratio for each brick.',
        #         transform=ax2.transAxes, ha="center")
        ax2.set_xlabel('Size in % of input')
        ax2.set_ylabel('% bricks with this size')

        xbins = len(delta_histogram)
        xaxis = np.arange(xbins, dtype=np.float32) # 0..nbins
        xaxis /= xbins # 0..1
        xaxis = 2*xaxis - 1 # -1..+1
        xaxis *= delta_histogram_clip
        #print("@@@ sample count", np.sum(delta_histogram), "zeros", delta_histogram[xbins//2])
        ax3.plot(xaxis, delta_histogram * (100 / np.sum(delta_histogram)), color='orange')
        #ax3.title.set_text("Histogram of errors, relative to RMS of data.")
        #ax3.text(0.5, 0.9, 'Histogram of errors, relative to RMS of data.',
        #         transform=ax3.transAxes, ha="center")
        ax3.set_xlabel('Error relative to RMS of data ({0:.0f})'.format(input_rms))
        ax3.set_ylabel('% samples with this error')

        fig.suptitle(label)
        plt.savefig(histogram)

def compare(filename1, filename2, outfilename, histogram, label):
    with ZgyReader(filename1, iocontext = SDCredentials()) as r1:
        with ZgyReader(filename2, iocontext = SDCredentials()) as r2:
            if r1.size != r2.size:
                print("Survey size mismatch:", r1.size, r2.size)
            else:
                if outfilename:
                    with ZgyWriter(outfilename,
                                   templatename=filename1,
                                   datatype=SampleDataType.float,
                                   iocontext = SDCredentials()) as w:
                        compare_open_files(r1, r2, w, progress=ProgressWithDots(), histogram=histogram, label=label)
                        w.finalize(progress=ProgressWithDots())
                else:
                    compare_open_files(r1, r2, None, progress=ProgressWithDots(), histogram=histogram, label=label)

if __name__ == "__main__":
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description='Compare two presumed equal ZGY files. Report any noise that might be caused by compression and/or quantization.', epilog=None)
    parser.add_argument('input1', help='ZGY input cube, local or sd://')
    parser.add_argument('input2', help='ZGY input cube, local or sd://')
    parser.add_argument('--output', help='Optional difference ZGY output cube, local or sd://, overwritten if it already exists.')
    parser.add_argument('--histogram', help='Optional histograms of noise and compression saved to this png file. Local files only.')
    parser.add_argument('--label', help='Title for the histogram. Defaults to the file names.')
    args = parser.parse_args()
    #print(args)
    if not args.label:
        args.label = "File 1: {0}\nFile 2: {1}".format(args.input1, args.input2)
    compare(args.input1, args.input2, args.output, args.histogram, args.label)

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
