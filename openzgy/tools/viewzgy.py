#!/usr/bin/env python3

import numpy as np
import sys
import os
from PIL import Image, ImageTk
import tkinter as tk

seismic_default = [
    [161, 255, 255], [160, 253, 255], [159, 252, 254], [158, 250, 254],
    [157, 249, 253], [156, 247, 253], [155, 246, 253], [154, 244, 252],
    [153, 242, 252], [152, 241, 251], [151, 239, 251], [150, 237, 251],
    [148, 236, 250], [147, 234, 250], [146, 232, 249], [145, 230, 249],
    [144, 229, 248], [143, 227, 248], [142, 225, 247], [140, 223, 247],
    [139, 221, 246], [138, 219, 246], [137, 217, 246], [136, 215, 245],
    [134, 213, 245], [133, 211, 244], [132, 209, 243], [130, 207, 243],
    [129, 205, 242], [128, 203, 242], [126, 200, 241], [125, 198, 241],
    [123, 196, 240], [122, 194, 240], [120, 191, 239], [119, 189, 238],
    [117, 186, 238], [116, 184, 237], [114, 181, 237], [113, 179, 236],
    [111, 177, 235], [110, 174, 235], [108, 171, 234], [106, 169, 233],
    [104, 166, 233], [103, 163, 232], [101, 160, 231], [99, 157, 231],
    [97, 155, 230],  [96, 152, 229],  [94, 149, 228],  [92, 146, 228],
    [90, 143, 227],  [88, 139, 226],  [86, 136, 225],  [84, 133, 225],
    [82, 130, 224],  [80, 126, 223],  [77, 123, 222],  [75, 119, 221],
    [73, 116, 220],  [71, 112, 219],  [68, 109, 218],  [66, 105, 217],
    [64, 101, 217],  [61, 97, 216],   [59, 93, 215],   [56, 89, 214],
    [54, 85, 213],   [51, 81, 211],   [48, 76, 210],   [45, 72, 209],
    [43, 68, 208],   [40, 63, 207],   [37, 59, 206],   [34, 54, 205],
    [31, 49, 203],   [28, 44, 202],   [24, 39, 201],   [21, 34, 200],
    [18, 29, 198],   [14, 23, 197],   [11, 17, 196],   [8, 12, 194],
    [4, 6, 193],     [0, 0, 191],     [5, 5, 184],     [9, 9, 178],
    [13, 13, 171],   [18, 18, 164],   [22, 22, 158],   [27, 27, 151],
    [32, 32, 144],   [36, 36, 138],   [40, 40, 131],   [45, 45, 124],
    [49, 49, 117],   [54, 54, 110],   [58, 58, 104],   [63, 63, 97],
    [67, 67, 90],    [72, 72, 83],    [77, 77, 77],    [81, 81, 81],
    [86, 86, 86],    [92, 92, 92],    [96, 96, 96],    [101, 101, 101],
    [107, 107, 107], [111, 111, 111], [116, 116, 116], [122, 122, 122],
    [126, 126, 126], [131, 131, 131], [137, 137, 137], [141, 141, 141],
    [146, 146, 146], [152, 152, 152], [156, 156, 156], [162, 162, 162],
    [167, 167, 167], [172, 172, 172], [177, 177, 177], [182, 182, 182],
    [187, 187, 187], [192, 192, 192], [197, 197, 197], [202, 202, 202],
    [202, 201, 200], [198, 196, 192], [193, 190, 184], [189, 185, 176],
    [185, 180, 168], [181, 175, 160], [177, 170, 152], [172, 164, 144],
    [168, 159, 136], [164, 153, 128], [160, 148, 120], [156, 143, 112],
    [151, 137, 104], [147, 132, 96],  [143, 127, 88],  [139, 122, 80],
    [135, 116, 72],  [130, 111, 64],  [126, 106, 56],  [122, 101, 48],
    [118, 95, 40],   [114, 90, 32],   [109, 85, 24],   [105, 79, 16],
    [101, 74, 8],    [97, 69, 0],     [103, 65, 0],    [108, 61, 0],
    [114, 56, 0],    [119, 53, 0],    [125, 48, 0],    [130, 44, 0],
    [136, 40, 0],    [141, 36, 0],    [147, 32, 0],    [152, 28, 0],
    [158, 24, 0],    [164, 20, 0],    [169, 16, 0],    [175, 12, 0],
    [180, 8, 0],     [186, 4, 0],     [191, 0, 0],     [193, 6, 0],
    [194, 12, 0],    [196, 17, 0],    [197, 23, 0],    [198, 29, 0],
    [200, 34, 0],    [201, 39, 0],    [202, 44, 0],    [203, 49, 0],
    [205, 54, 0],    [206, 59, 0],    [207, 63, 0],    [208, 68, 0],
    [209, 72, 0],    [210, 76, 0],    [211, 81, 0],    [213, 85, 0],
    [214, 89, 0],    [215, 93, 0],    [216, 97, 0],    [217, 101, 0],
    [217, 105, 0],   [218, 109, 0],   [219, 112, 0],   [220, 116, 0],
    [221, 120, 0],   [222, 123, 0],   [223, 126, 0],   [224, 130, 0],
    [225, 133, 0],   [225, 136, 0],   [226, 140, 0],   [227, 143, 0],
    [228, 146, 0],   [228, 149, 0],   [229, 152, 0],   [230, 155, 0],
    [231, 158, 0],   [231, 160, 0],   [232, 163, 0],   [233, 166, 0],
    [233, 169, 0],   [234, 171, 0],   [235, 174, 0],   [235, 177, 0],
    [236, 179, 0],   [237, 182, 0],   [237, 184, 0],   [238, 187, 0],
    [238, 189, 0],   [239, 191, 0],   [240, 194, 0],   [240, 196, 0],
    [241, 198, 0],   [241, 200, 0],   [242, 203, 0],   [242, 205, 0],
    [243, 207, 0],   [244, 209, 0],   [244, 211, 0],   [245, 213, 0],
    [245, 215, 0],   [246, 217, 0],   [246, 219, 0],   [247, 221, 0],
    [247, 223, 0],   [247, 225, 0],   [248, 227, 0],   [248, 229, 0],
    [249, 230, 0],   [249, 232, 0],   [250, 234, 0],   [250, 236, 0],
    [251, 237, 0],   [251, 239, 0],   [251, 241, 0],   [252, 242, 0],
    [252, 244, 0],   [253, 246, 0],   [253, 247, 0],   [253, 249, 0],
    [254, 250, 0],   [254, 252, 0],   [255, 254, 0],   [255, 255, 0],
    ]

seismic_default = np.array(seismic_default, dtype=np.uint8)

def savePNG(data, outfile, *, title="Seismic", datarange=None):
    def normalize(a, *, datarange = None):
        a = a.astype(np.float32)
        dead = np.isnan(a)
        amin, amax = datarange or (np.nanmin(a), np.nanmax(a))
        # Zero should be at the center
        if amin * amax < 0:
            x = max(abs(amin), abs(amax))
            amin, amax = (-x, x)
        # NaN and Inf show as smallest number
        a[dead] = amin
        if amin == amax:
            a *= 0
        else:
            # Avoid underflow, because app might have np.seterr(all='raise')
            a = a.astype(np.float64)
            a = (a - amin) / (amax - amin)
        a = (a * 255).astype(np.uint8)
        return a, dead

    if not outfile:
        raise ValueError("outfile must be specified")
    #elif outfile[:-4] != ".png":
    #    raise ValueError("outfile must end in .png:", outfile[:-4])

    data = np.squeeze(data)
    data = np.transpose(data)
    data = np.flip(data, 1)
    data, dead = normalize(data, datarange=datarange)
    tmp = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    r = tmp[...,0]
    g = tmp[...,1]
    b = tmp[...,2]
    ind = seismic_default[data]
    r += ind[...,0] # data
    g += ind[...,1] # data
    b += ind[...,2] # data
    r[dead] = 255
    g[dead] = 255
    b[dead] = 0
    im = Image.fromarray(tmp, mode="RGB")
    im.save(outfile, format="PNG")

def showFileInTk(filename, title):
    window = tk.Tk()
    #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = ImageTk.PhotoImage(Image.open(filename))
    #This creates the main window of an application
    window.title(title)
    window.geometry("{0}x{1}".format(img.width(), img.height()))
    window.configure(background='grey')
    #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tk.Label(window, image = img)
    #The Pack geometry manager packs widgets in rows or columns.
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    #Start the GUI
    window.mainloop()

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
