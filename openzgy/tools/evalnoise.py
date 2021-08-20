#!/usr/bin/env python3

#print('Running' if __name__ == '__main__' else 'Importing', __file__)

import numpy as np
import sys
import os
import argparse
from PIL import Image
import tkinter as tk

from ..api import ZgyReader
from ..test.utils import SDCredentials
from ..tools.viewzgy import savePNG, showFileInTk

def run(filename, *, lods = [0], direction = 0, datarange=None,
        outname = None, iocontext=None):
    """
    Read one trace or slices in the specified direction, save and display.
    """
    allslices = []
    with ZgyReader(filename, iocontext=SDCredentials()) as reader:
        for lod in lods:
            step = 1<<lod
            start = [0, 0, 0]
            size = np.array(reader.size, dtype=np.int64) // (1 << lod)
            start[direction] = size[direction] // 2
            size[direction] = 1
            if direction == 2:
                size[0] = min(size[0], 1024)
                size[1] = min(size[1], 1024)
            section = np.zeros(size, dtype=np.float32)
            reader.read(start, section, lod)
            allslices.append(np.squeeze(section))
    s = allslices[0].shape
    w = np.sum([allslices[i].shape[0] for i in range(3)])
    combined = np.zeros((w, s[1]), dtype=allslices[0].dtype)
    combined[0:s[0],:] = allslices[0]
    ss = allslices[1].shape
    combined[s[0]:s[0]+ss[0], 0:ss[1]] = allslices[1]
    sss = allslices[2].shape
    combined[s[0]+ss[0]:s[0]+ss[0]+sss[0], 0:sss[1]] = allslices[2]
    combined = combined[::-1,::]
    savePNG(combined, outname, datarange=datarange)
    showFileInTk(outname, title=os.path.basename(outname))

def Main(args, *, gain="1", prefix=""):
    """
    Show 3 ZGY files, with the 3 first LODs.
    The files are assumed to be uncompressed original, file after compression,
    and the difference between those two. The value range of the diff is a
    certain percentage of the value range in the first inout. This makes it
    possible to compare the quality from different settings.
    """
    with ZgyReader(args[0], iocontext=None) as reader:
        datarange0 = reader.datarange
        datarange1 = ((reader.datarange[0]/gain, reader.datarange[1]/gain))
    run(args[0], lods=range(3), direction=0, datarange = datarange0,
            outname = prefix + "-showorig.png")
    run(args[1], lods=range(3), direction=0, datarange = datarange0,
            outname = prefix + "-showcomp.png")
    run(args[2], lods=range(3), direction=0, datarange = datarange1,
            outname = prefix + "-showdiff.png")

if __name__ == "__main__":
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description='Evaluate compression')
    parser.add_argument('orig', help='Original ZGY input file')
    parser.add_argument('comp', help='Data after compression')
    parser.add_argument('diff', help='Compression noise')
    parser.add_argument('--gain', nargs=1, default=[50], type=int,
                        help='Scaling of diff cube relative to orig range.')
    parser.add_argument('--prefix', nargs=1, default=["check"], type=str,
                        help='prefix for output file name.')
    args = parser.parse_args()
    #print(args)
    Main([args.orig, args.comp, args.diff], gain=args.gain[0], prefix=args.prefix[0])
    sys.exit(0)

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
