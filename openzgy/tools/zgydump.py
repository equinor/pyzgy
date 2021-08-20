#!/usr/bin/env python3

import numpy as np
import os, sys
import argparse
from ..api import ZgyReader, SampleDataType
from ..test.utils import SDCredentials

_brief_info = """
File name                      = '{name}'
File size (bytes)              = {r._fd.xx_eof:,d}
File format and version        = {r.datatype.name} ZGY version {r._accessor._metadata._fh._version}
Current data Version           = {r.verid}
Brick size I,J,K               = {r.bricksize}
Number of bricks I,J,K         = {r.brickcount[0]}
Number of LODs                 = {r.nlods}
Coding range min/max           = {r.datarange[0]:.6g} {r.datarange[1]:.6g} (raw: {r.raw_datarange[0]:.6g} {r.raw_datarange[1]:.6g}) {nsamples:,d}
Statistical min/max/count      = {r.statistics.min:.6g} {r.statistics.max:.6g} {r.statistics.cnt:,d}
Histogram range min/max/count  = {r.histogram.min:.6g} {r.histogram.max:.6g} {r.histogram.cnt:,d}
Inline start/increment/count   = {r.annotstart[0]} {r.annotinc[0]} {r.size[0]}
Xline  start/increment/count   = {r.annotstart[1]} {r.annotinc[1]} {r.size[1]}
Sample start/increment/count   = {r.zstart} {r.zinc} {r.size[2]}
Horizontal projection system   = {r._accessor._metadata._ih._hprjsys}
Horizontal dim/factor/name     = {r.hunitdim.name} {r.hunitfactor} '{r.hunitname}'
Vertical dim/factor/name       = {r.zunitdim.name} {r.zunitfactor} '{r.zunitname}'
Ordered Corner Points Legend   = [  <i>,   <j>] {{ <inline>,   <xline>}} (  <easting>,  <northing>)
Ordered Corner Point 1         = [{r.indexcorners[0][0]:5d}, {r.indexcorners[0][1]:5d}] {{{r.annotcorners[0][0]:9g}, {r.annotcorners[0][1]:9g}}} ({r.corners[0][0]:11.2f}, {r.corners[0][1]:11.2f})
Ordered Corner Point 2         = [{r.indexcorners[1][0]:5d}, {r.indexcorners[1][1]:5d}] {{{r.annotcorners[1][0]:9g}, {r.annotcorners[1][1]:9g}}} ({r.corners[1][0]:11.2f}, {r.corners[1][1]:11.2f})
Ordered Corner Point 3         = [{r.indexcorners[2][0]:5d}, {r.indexcorners[2][1]:5d}] {{{r.annotcorners[2][0]:9g}, {r.annotcorners[2][1]:9g}}} ({r.corners[2][0]:11.2f}, {r.corners[2][1]:11.2f})
Ordered Corner Point 4         = [{r.indexcorners[3][0]:5d}, {r.indexcorners[3][1]:5d}] {{{r.annotcorners[3][0]:9g}, {r.annotcorners[3][1]:9g}}} ({r.corners[3][0]:11.2f}, {r.corners[3][1]:11.2f})
"""

_hist_info = "Histogram bin {0:3d}              = {1:11d}"

def all_brick(reader):
    for lod in range(reader.nlods):
        for ii in range(reader.brickcount[lod][0]):
            for jj in range(reader.brickcount[lod][1]):
                for kk in range(reader.brickcount[lod][2]):
                    info = reader._accessor._getBrickFilePosition(ii,jj,kk,lod)
                    yield ((lod, ii, jj, kk),) + info

def all_alpha(reader):
    for lod in range(reader.nlods):
        for ii in range(reader.brickcount[lod][0]):
            for jj in range(reader.brickcount[lod][1]):
                info = reader._accessor._getAlphaFilePosition(ii,jj,lod)
                yield ((lod, ii, jj),) + info + (0,)

def summary_brick_offsets(reader):
    alpha = dict()
    brick = dict()
    for info in all_alpha(reader):
        alpha[info[1].name] = alpha.get(info[1].name, 0) + 1
    for info in all_brick(reader):
        brick[info[1].name] = brick.get(info[1].name, 0) + 1
    print("{0:30s} = {1}".format("Alpha status", str(alpha)))
    print("{0:30s} = {1}".format("Brick status", str(brick)))

def summary_normal_size(reader, *, header = True):
    """
    Useful for performance measurements.
    1) The number of allocated LOD0 bricks, used to compute bandwidth after
       timing the read of the entire file. Specified as bricks and MB.
    2) Size of one brick_column, allocated or not, given as above.
    3) Brick size in bytes.
    Note that size in MB is computed by integer division.
    If you need them exact, do the (float) division yourself.
    """
    x = [e for e in all_brick(reader) if e[0][0] == 0 and e[1].name == "Normal"]
    normal = len(x)
    bytespersample = {SampleDataType.int8:  1,
                      SampleDataType.int16: 2,
                      SampleDataType.float: 4}[reader.datatype]
    bytesperbrick = np.product(reader.bricksize) * bytespersample
    colsize = (reader.size[2] + reader.bricksize[2] - 1) // reader.bricksize[2]
    bytespercolumn = bytesperbrick * colsize
    fmt = "{0:30s} = LOD0: {1} {2} MB column {3} {4} MB brick {5}"
    if not header: fmt = "{1} {2} {3} {4} {5}"
    print(fmt.format(
        "Normal LOD0 bricks & col size", normal,
        (normal * bytesperbrick) // (1024*1024),
        colsize,
        (colsize * bytesperbrick) // (1024*1024),
        bytesperbrick))

def dump_brick_offsets(reader, sort):
    print("BRICK offsets:")
    table = all_brick(reader)
    if sort:
        table = sorted(table, key=lambda x: x[2] or 0)
    for (pos, brickstatus, fileoffset, constvalue, bricksize) in table:
        addr = "[{0}][{1}][{2}][{3}]".format(pos[0], pos[1], pos[2], pos[3])
        if fileoffset is not None:
            print("{addr:20s} = {fileoffset:16x} {brickstatus.name} Size {bricksize:8x}".format(
                addr=addr,
                brickstatus=brickstatus, fileoffset=fileoffset,
                constvalue=constvalue, bricksize=bricksize))
        else:
            print("{addr:20s} = {fileoffset:16s} {brickstatus.name} {constvalue}".format(
                addr=addr,
                brickstatus=brickstatus, fileoffset="",
                constvalue=constvalue, bricksize=""))

def dump_alpha_offsets(reader, sort):
    print("ALPHA offsets:")
    table = all_alpha(reader)
    if sort:
        table = sorted(table, key=lambda x: x[2] or 0)
    for (pos, brickstatus, fileoffset, constvalue, bricksize) in table:
        addr = "[{0}][{1}][{2}]".format(pos[0], pos[1], pos[2])
        if fileoffset is not None:
            print("{addr:20s} = {fileoffset:16x} {brickstatus.name}".format(
                addr=addr,
                brickstatus=brickstatus, fileoffset=fileoffset,
                constvalue=constvalue))
        else:
            print("{addr:20s} = {fileoffset:16s} {brickstatus.name} {constvalue}".format(
                addr=addr,
                brickstatus=brickstatus, fileoffset="",
                constvalue=constvalue))

def dump_combined_offsets(reader, sort):
    print("BRICK and ALPHA offsets sorted by address:")
    table = list(all_brick(reader)) + list(all_alpha(reader))
    if sort:
        table.sort(key=lambda x: x[2] or 0)
    for (pos, brickstatus, fileoffset, constvalue, bricksize) in table:
        if len(pos) == 4:
            addr = "brick [{0}][{1}][{2}][{3}]".format(pos[0], pos[1], pos[2], pos[3])
        elif len(pos) == 3:
            addr = "alpha [{0}][{1}][{2}]".format(pos[0], pos[1], pos[2])
        if fileoffset is not None:
            print("{addr:26s} = {fileoffset:16x} {brickstatus.name} Size {bricksize:8x}".format(
                addr=addr,
                brickstatus=brickstatus, fileoffset=fileoffset,
                constvalue=constvalue, bricksize=bricksize))
        else:
            print("{addr:26s} = {fileoffset:16s} {brickstatus.name} {constvalue}".format(
                addr=addr,
                brickstatus=brickstatus, fileoffset="",
                constvalue=constvalue, bricksize=""))

def run(filename, options):
    with ZgyReader(filename, iocontext = SDCredentials()) as reader:
        if options.only_lod0_info:
            summary_normal_size(reader, header=False)
            return
        args = dict(name=filename,
                    nsamples=np.product(reader.size),
                    r=reader)
        #print(_brief_info.format(**args))
        for line in _brief_info.split('\n'):
            if line:
                try:
                    print(line.format(**args))
                except Exception as ex:
                    print(line.split('=')[0] + "= N/A " + str(ex))
        summary_brick_offsets(reader)
        summary_normal_size(reader)
        if options.histogram:
            hh = reader.histogram.bin
            for ii in range(len(hh)):
                print(_hist_info.format(ii, hh[ii]))
        if options.sorted_offsets:
            dump_combined_offsets(reader, True)
        elif options.offsets:
            dump_brick_offsets(reader, False)
            dump_alpha_offsets(reader, False)

# Features from the old C++ zgydump
#    -b --brief BriefInfo()
#    (default)  FullInfo()
#       In new code, --histogram must be requested explicitly.
#       There is no --brief since the two outputs then end up
#       rather similar.
#    -p --performance PerfInfo()
#       normal/empty/const counts now printed unconditionally.
#       Can also consider merging test_zgydump and test_isoptimal.
#    -o --offset Offsets()
#       Better than the original, handles const(value) and compressed(size).
#    -s --slice  Slice()
#       Covered by test_show. Might as well keep this as a separate exe.
#    -a --alpha Alpha()
#       Not implemented, as the alpha tiles are deprecated.

def Main():
    np.seterr(all='raise')

    parser = argparse.ArgumentParser(description='Show ZGY file details')
    parser.add_argument('files', nargs="+", help='ZGY files, local or sd://')
    parser.add_argument('--histogram', action='store_true', help="Show the 256-bin histogram")
    parser.add_argument('--offsets', action='store_true', help="Show offset of each brick")
    parser.add_argument('--sorted-offsets', action='store_true', help="Sort by file offset")
    parser.add_argument('--only-lod0-info', action='store_true', help="Only lod0 stats etc.")
    args = parser.parse_args()
    for filename in args.files:
        run(filename, args)

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
