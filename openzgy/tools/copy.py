#!/usr/bin/env python3

"""
Implement a simple copy command that exercises the new code.
Optionally crop the input.
"""

import numpy as np
import os, sys, time, argparse
from ..api import ZgyReader, ZgyWriter, SampleDataType, ProgressWithDots, ZgyCompressFactory
from ..test.utils import SDCredentials
from ..iterator import readall

def suggest_range(value, dt):
    """
    Special case handling for inconsistent all-constant files.
    A coding range with min==max is not valid, so we must choose
    something else.
    """
    dt_lo, dt_hi = {SampleDataType.int8: (-128, +127),
                    SampleDataType.int16: (-32768, +32767)}[dt]
    if value == 0:
        # Choose the range -1..+1, but slightly wider at the low end
        # to make float-0 map exactly to int-0, zero centric.
        return (dt_lo / dt_hi, 1)
    elif value > 0:
        # Choose the range 0..2*value, but slightly less at the high end
        # to make float-value map exactly to int-0, zero centric.
        return (0, value * (1 - dt_hi / dt_lo))
    else:
        # Choose the range 2*value..0, but slightly wider at the low end
        # to make float-value map exactly to int-0, zero centric.
        return (value * (1 - dt_lo / dt_hi), 0)

def find_center(r, maxsize):
    """
    Crop partial bricks outside or below the survey since they
    would just confuse the statistics.
    If the file is large then crop out an area around the center.
    Try to read the full traces but limit the il and xl.
    """
    size = np.array(r.size, dtype=np.int64)
    cropsize = np.minimum(size, np.array(maxsize, dtype=np.int64))
    cropsize = (cropsize//64)*64
    cropoffset = ((size - cropsize)//128)*64
    if np.any(size - cropsize >= 64):
        print("Reading center", tuple(cropsize),
              "offset", tuple(cropoffset),
              "of survey size", tuple(r.size))
    return tuple(cropsize), tuple(cropoffset)

def copy_open_file(r, w, *, progress, offset=(0,0,0), noisefactor=0):
    noiselevel = abs(r.datarange[1] - r.datarange[0]) / noisefactor if noisefactor > 0 else None
    for datastart, datasize, data in readall(r, dtype=np.float32, cropsize=w.size, cropoffset=offset, progress=progress):
        if noiselevel:
            # Note, maybe multiply with a random factor as well,
            # if the goal is to test compression when white noise
            # is present but where the input used to be integral.
            data += (np.random.random_sample(data.shape) - 0.5) * noiselevel
        w.write(datastart, data)

def read_and_discard(r, cropsize, *, progress, offset=(0,0,0), noisefactor=0):
    """This is for performance testing only."""
    for datastart, datasize, data in readall(r, dtype=np.float32, cropsize=cropsize, cropoffset=offset, progress=progress):
        pass

def copy(srcfilename, dstfilename, crop_offset=None, crop_size=None, crop_center=None, forcetype='unchanged', datarange=None, noisefactor=0, snr = 0, progress1 = None, progress2 = None):
    starttime = time.time()
    with ZgyReader(srcfilename, iocontext = SDCredentials()) as r:
        if crop_center:
            if crop_offset and crop_offset != (0,0,0):
                print("WARNING: ignoring --offset because --center specified.")
            if not crop_size or crop_size == (0,0,0):
                crop_size = (640, 640, 12800)
            crop_size, crop_offset = find_center(r, crop_size)
        # Zero cropped size means no cropping, i.e. "to end of survey".
        # TODO-Low passing crop_size but not crop_offset could mean "center".
        crop_beg  = crop_offset or (0,0,0)
        crop_size = crop_size or (0,0,0)
        crop_beg = tuple([crop_beg[i] if crop_beg[i] >= 0 else r.size[i] + crop_beg[i] for i in range(3)])
        crop_size = tuple([crop_size[i] or r.size[i]-crop_beg[i] for i in range(3)])
        crop_end  = tuple([crop_beg[i] + crop_size[i] - 1 for i in range(3)])
        # Need to re-calculate the corners and the annotation
        index_corners = [ [ crop_beg[0], crop_beg[1] ],
                          [ crop_end[0], crop_beg[1] ],
                          [ crop_beg[0], crop_end[1] ],
                          [ crop_end[0], crop_end[1] ] ]
        try:
            world_corners = [ r.indexToWorld(tuple(x)) for x in index_corners ]
        except Exception:
            # World coordinates bad, e.g. all might have been zero.
            # Keeping the original corners can hardly make it worse.
            world_corners = [ [0, 0], [0, 0], [0, 0], [0, 0] ]
        annotstart = r.indexToAnnot((crop_beg[0], crop_beg[1]))
        zstart = r.zstart + crop_beg[2] * r.zinc
        # Optionally change the datatype (given in bytes per sample)
        datatype = {
            'unchanged': r.datatype,
            'int8':      SampleDataType.int8,
            'int16':     SampleDataType.int16,
            'float':     SampleDataType.float
        } [forcetype]

        singlevalue = None
        if not datarange:
            if r.datatype in (SampleDataType.int8, SampleDataType.int16):
                if r.raw_datarange[0] == r.raw_datarange[1]:
                    singlevalue = r.raw_datarange[0]
                    datarange = suggest_range(singlevalue, r.datatype)
                    # Note, if read datatype != write datatype the
                    # data range no longer is zero centric. On the
                    # other hand, the write datatype is then probably
                    # float which means it doesn't matter.

        print("Cropping: offset ", crop_beg, "size", crop_size, "of", r.size)
        #print("World corners now", world_corners, "was", r.corners)
        #print("Annot start now  ", annotstart, "was", r.annotstart)
        print("Data type", r.datatype, "->", datatype)
        with ZgyWriter(dstfilename,
                       compressor = ZgyCompressFactory("ZFP", snr = snr),
                       iocontext = SDCredentials(),
                       size = crop_size or r.size,
                       datatype = datatype if snr<=0 else SampleDataType.float,
                       datarange = datarange or r.datarange,
                       zunitdim = r.zunitdim,
                       zunitname = r.zunitname,
                       zunitfactor = r.zunitfactor,
                       hunitdim = r.hunitdim,
                       hunitname = r.hunitname,
                       hunitfactor = r.hunitfactor,
                       zstart = zstart,
                       zinc = r.zinc,
                       annotstart = r.annotstart,
                       annotinc = r.annotinc,
                       corners = world_corners) as w:
            opentime = time.time()
            if dstfilename != "/dev/null":
                if singlevalue is not None:
                    # The input file is known to contain just a single value.
                    # And the coding range is degenerate. How this is handled
                    # by the reader is not well defined. So, ignore the values
                    # that were read and just use the constant. Don't honor
                    # noiselevel int this case. That kludge doesn't really
                    # make sense for integral files anyway.
                    if True:
                        print("Writing a constant-value file, range",
                              datarange[0], "..", datarange[1],
                              "value", singlevalue)
                    w.writeconst((0,0,0), singlevalue, w.size, False)
                else:
                    copy_open_file(r, w, progress = progress1 or ProgressWithDots(), offset=crop_beg, noisefactor=noisefactor)
            else:
                read_and_discard(r, crop_size or r.size, progress = progress1 or ProgressWithDots(), offset=crop_beg)
            copytime = time.time()
            if dstfilename != "/dev/null":
                w.finalize(progress = progress2 or ProgressWithDots())
            finaltime = time.time()
    flushtime = time.time()
    if True:
        timing_report(w, flushtime - starttime)
    if True:
        print("Times: open {0:.2f} copy {1:.2f} final {2:.2f} flush {3:.2f}".format(
            opentime - starttime,
            copytime - opentime,
            finaltime - copytime,
            flushtime - finaltime))

def timing_report(writer, elapsed):
    bs = np.array(writer.bricksize, dtype=np.int64)
    size = np.array(writer.size, dtype=np.int64)
    paddedsize = ((size + bs - 1) // bs) * bs
    bandwidth = np.product(paddedsize) / elapsed # should I use size or padsize?
    bandwidth /= (1024*1024)
    print("Elapsed {0:7.2f} seconds, bandwidth {1:6.2f} MVoxel/s copying {2} {3} samples, exact {4:.0f} MVoxel, padded {5:.0f} MVoxel".format(
        elapsed, bandwidth, writer.datatype, tuple(size),
        np.product(size) / (1024*1024),
        np.product(paddedsize) / (1024*1024)))

def parseints(s):
    return tuple(map(int,s.split(",")))

def Main():
    # Can do this in a stand alone app, but not as part of a library.
    # The main problem is expressions like x = int(offset) + np.int32(size)
    # which works fine on Linux but fails occasionally on windows. The
    # reason is that offset gets changed to np.int32 on windows if it is
    # small enough; np.float64 otherwise. On Linux it is always changed
    # to np.int64. On windows the expression will overflow if and only
    # if offset is slightly less than np.int32.max.
    # Since I am adding extra tests anyway I'll cause exceptions to be
    # raised also on divide by error, underflow, etc.
    np.seterr(all='raise')

    parser = argparse.ArgumentParser(description='Copy a ZGY file.', epilog="""
    The output cube will have its data bricks sorted by lod, I, J, K
    for optimized reads from the cloud.""")
    parser.add_argument('input', help='ZGY input cube, local or sd://')
    parser.add_argument('output', help='ZGY output cube, local or sd://')
    parser.add_argument('--offset', nargs=1, default=["0,0,0"], type=str,
                        help='i,j,k Starting point in the source cube. Negative numbers count from the end.')
    parser.add_argument('--size', nargs=1, default=["0,0,0"], type=str,
                        help='i,j,k size of data to be copied. Zero means to end of cube.')
    parser.add_argument('--center', action='store_true', help="Ignore --offset, crop out center of cube.")
    parser.add_argument('--forcetype', default='unchanged', choices=['unchanged', 'int8', 'int16', 'float'], help='Produce a cube of this type instead of keeping the input type. If converting from float to int then the coding range must already be correct.')
    parser.add_argument('--datarange', nargs=2, default=None, metavar=('MIN', 'MAX'), type=float, help='Required when converting from float to integral types.')
    parser.add_argument('--noisefactor', nargs=1, default=[0], metavar='FACTOR', type=float, help='Amount of noise to add. E.g. 5000 means add 1/5000 of total data range.')
    parser.add_argument('--overwrite', action='store_true', help="Quietly overwrite the output file if it exists already.")
    parser.add_argument('--snr', default=0, type=int,
                        help='Pass 10..70 for lossy compression, 99 for lossless, 0 for uncompressed.')
    args = parser.parse_args()
    #print(args)
    args.offset = parseints(args.offset[0])
    args.size = parseints(args.size[0])
    args.datarange = tuple(args.datarange) if args.datarange is not None else None
    args.noisefactor = args.noisefactor[0]
    #print(args)
    if not args.input or not args.output:
        print("File names cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if not args.overwrite and os.path.exists(args.output):
        print('Output file "{0}" already exists.'.format(args.output), file=sys.stderr)
        print('If you really meant to overwrite it, specify --overwrite.', file=sys.stderr)
        sys.exit(1)
    copy(args.input, args.output, args.offset, args.size, args.center, args.forcetype, args.datarange, args.noisefactor, args.snr)

if __name__ == "__main__":
    Main()

# Copyright 2017-2021, Schlumberger
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
