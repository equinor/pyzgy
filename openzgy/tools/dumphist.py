#!/usr/bin/env python3

"""
Compute a histogram (default 4096 bins) of the input file and plot it.
The intent is to spot float- or int16 data that has been improperly
processed. Possibly by having been stored as int8 at some point in
the processing.

Many data files have a huge number of zeros, in which case real
samples would hardly be visible at all. The code tries to fix this by
clipping the 5 highest entries. Caveat: If the data set contains
ridiculously large spikes then then all the real data will look like
~0 and will all end up in the same bin. Which is then clipped, leaving
only the garbage visible. This is not a bug per se but can be very
confusing. As an example, look at B-03A-87-LA which has real data
+/- 12 or so, and random samples in the +/- 1e16 range.
"""

import numpy as np
import os, sys, time, argparse
import matplotlib.pyplot as plt
from ..api import ZgyReader, ZgyWriter, SampleDataType, ProgressWithDots
from ..test.utils import SDCredentials
from ..iterator import readall

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

def run_open_file(r, datarange, *, progress = None, bincount = 256, crop = False):
    histogram = np.zeros(bincount, dtype=np.int64)
    (cropsize, cropoffset) = find_center(r) if crop else (None, None)
    for _, _, data in readall(r, dtype=np.float32, cropsize=cropsize, cropoffset=cropoffset, progress=progress):
        histogram += np.histogram(data, bins=bincount, range=datarange)[0]
    return histogram

def run(srcfilename, outfilename, datarange=None, bincount=256, crop=False):
    with ZgyReader(srcfilename, iocontext = SDCredentials()) as r:
        if not datarange:
            datarange = tuple(r.datarange)
            # Make symmetric
            if datarange[0] * datarange[1] < 0:
                lim = max(abs(datarange[0]), abs(datarange[1]))
                datarange = (-lim, lim)
        print("Data type", r.datatype,
              "range", datarange, "codingrange", r.datarange,
              "legacy", r._meta._ih._codingrange,
              "statistical", (r.statistics.min, r.statistics.max))
        if r.datatype == SampleDataType.int8:
            bincount=256
        histogram = run_open_file(r, datarange, bincount=bincount, crop=crop,
                                  progress=ProgressWithDots())
    assert len(histogram) == bincount
    plot_histogram("file: " + srcfilename, histogram)
    plt.savefig(outfilename)

def plot_histogram(title, histogram):
    bincount = len(histogram)
    nz = np.count_nonzero(histogram)
    zz = len(histogram) - nz
    #print(list(sorted(histogram)))
    #clipvalue = list(sorted(histogram))[-nz//10] # at 90-percentile
    clipvalue = list(sorted(histogram))[-5] # clip a fixed number of bins
    # Clipping so a large number of zero or close-to-zero won't cause trouble.
    if False:
        print("raw histogram   ", histogram)
        print("sorted histogram", np.array(sorted(histogram))[-6:])
        print("clip from", list(sorted(histogram))[-1],
              "to", clipvalue, "samples/bin")
    # But: If there are fewer than 5 buckets with data in them, don't bother.
    # This can really happen. If a data set has a single huge spike then only
    # the "spike" bin and the "close to zero" bin will be populated.
    # The histogram will be useless, but at least try to show what happened
    # instead of bombing out with a divide by zero.
    # Example data set: sd://sntc/millet/usgs/zgy/B-09-88-LA.zgy
    # which has a real value range +/- 100000 but spices at -9.1e15
    if clipvalue != 0:
        np.clip(histogram, 0, clipvalue, out=histogram)
        #print("clipped hist at ", clipvalue, ":", histogram)
    else:
        print("WARNING: Fewer than 5 buckets filled. Are there spikes?")

    topvalue = list(sorted(histogram))[-1]
    sumvalue = np.sum(histogram)

    print("{0:.1f}% of the {1} histogram bins are empty.".format(
        100.0*zz/len(histogram),
        len(histogram)))

    if sumvalue == 0:
        print("ERROR: histogram is empty.")
    else:
        print("{0:.1f}% of all samples fall in the same bin.".format(
            100.0*topvalue/sumvalue))

    # The plot only allows displaying 512 distinct values without the plot
    # itself starting to generate artifacts. So we must compress, or zoom,
    # or preferably both.
    factor = max(1, bincount // 512)

    zoomed = histogram[(factor-1)*bincount//(2*factor):(factor+1)*bincount//(2*factor)]
    compressed = np.array([np.sum(histogram[ii:ii+factor]) for ii in range(0, bincount, factor)], dtype=np.int64)

    #plt.figure(num=None, figsize=(20, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 7), dpi=100, facecolor="red", edgecolor="green")
    #ax1.set_facecolor('#dddddd')
    ax1.plot(np.arange(len(zoomed)), zoomed)
    ax2.plot(np.arange(len(compressed)), compressed)
    ax1.title.set_text("Zoomed in {0}x, showing the {1} bins at the center of a {2} bin histogram".format(factor, bincount//factor, bincount))
    ax2.title.set_text("Entire histogram, compressed to {0} bins".format(bincount//factor))
    #x = [111,122,155,192,11,123,120,]
    #y = [3,4,3,5,9,10,23]
    #plt.bar(x,y)
    fig.suptitle(title)

if __name__ == "__main__":
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description='Compute and show histogram')
    parser.add_argument('input', help='ZGY input cube, local or sd://')
    parser.add_argument('--output', default=None, help='PNG output file')
    parser.add_argument('--datarange', nargs=2, default=None, metavar=('MIN', 'MAX'), type=float, help='Histogram limits')
    parser.add_argument('--bins', nargs=1, default=[4096], type=int, help='Number of bins in histogram')
    parser.add_argument("--crop", action='store_true', help='Limit the number of traces scanned')
    args = parser.parse_args()
    args.datarange = tuple(args.datarange) if args.datarange is not None else None
    args.bins = args.bins[0]
    if not args.output:
        args.output = (args.input.rstrip('/').split('/')[-1]) + "-histogram.png"
    print(args)
    run(args.input, args.output, args.datarange, args.bins, args.crop)

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
