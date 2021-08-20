#!/usr/bin/env python3

"""
Deal with spatial location of the cube.

The grid definition in gpiline, gpxline, gpx, and gpy defines
an affine transform from inline, crossline to world X, Y.
Older ZGY files might use several ways to specify the transform.

Since an affine transform is used it is possible to define a
coordinate system where the X and Y axes are not precisely
perpendicular to each other. It is up to the application reading the
files to decide whether to accept non-orthogonal coordinate systems.
When writing files you should think long and hard about whether you
really need to allow the user to create a non orthogonal ZGY file.
If you do, you might regret it after a few decades of supporting that
feature in all your applications that read ZGY.


"FourPoint" is what new ZGY files should use on write. This provides
the coordinates of the four corners of the survey both in ordinal
(i, j), annotation (inline, crossline), and world (X, Y) coordinates.
The order of these points is:

   First inline / first crossline,
   last inline / first crossline,
   first inline / last crossline,
   last inline / last crossline.

The API will always return the lattice as "FourPoint", even if the ZGY
file uses a different way of specifying geometry.

The ordinal cornere are not stored. They are implied to be

   (          0,           0)
   (size[0] - 1,           0)
   (          0, size[1] - 1)
   (size[0] - 1, size[1] - 1)

The annotation corners are required to be stored, even though this is
redundant with "orig" and "inc":

   (orig[0],                    orig[1])
   (orig[0]+inc[0]*(size[0]-1), orig[1])
   (orig[0],                    orig[1]+inc[1]*(size[1]-1))
   (orig[0]+inc[0]*(size[0]-1), orig[1]+inc[1]*(size[1]-1))

The world X, Y coordinates are required to be stored for all four
corners, in spite of the last corner being redundant. As long as the
transform is required to be affine, three points are sufficient.

"ThreePoint" is more general. Instead of storing the four corners,
three arbitrary points are specified with both inline, crossline and
world X, Y coordinates. The last point in the arrays is unused. As
long as the points are not colinear or duplicated, this is enough to
define the affine transform.

The "Parametric" type is not used. The original intent was that with
"Parametric", the "gazim" field would contain Inline, crossline
azimuths (a.k.a. "base angle" and "side angle", respectively) as
360-degree compass direction (0 = N, 90 = E, 180 = S, 270 = W). The
"gbinsz" field would contain Inline, crossline bin size as METERS
always (i.e. NOT dependent on hdim or hunit). To my knowledge, no ZGY
files have ever been written using these fields. So, readers should
treat both "Unknown" and "Parametric" as errors.

New files should always be written using "FourPoint" to assure that
all ZGY readers handle them correctly. When reading, always assume
"ThreePoint" on the file but convert to "FourPoint" before providing
presenting the information to the user.

This means the reader should trust that the first three points in
(gpiline, gpxline) line up with the first three points in (gpx, gpy).
The reader should not trust that those points are in fact the corners.
And the reader should ignore the last point completely. ZGY files in
the wild might have been written explicitly in "ThreePoint" mode, or
they might accidentally have been written as "FourPoint" with the
corners in the wrong order. Reading in "ThreePoint" mode handles all
those cases.
"""
##@package openzgy.impl.transform
#@brief Deal with spatial location of the cube.

def generalTransform(ax0, ay0, ax1, ay1, ax2, ay2,
                     bx0, by0, bx1, by1, bx2, by2,
                     data):
    """
    Coordinate conversion based on 3 arbitrary control points
    in both the source and the target system, i.e. no need
    to calculate the transformation matrix first.

    This is my favorite code snippet for coordinate conversion.
    Almost no ambiguity about what it does and how to use it.

    The data array is converted in place. The code tries to ensure
    that the output is floating point even though the input might
    not be. This is to prevent surprises if converting the result
    to a numpy type (where types are more strict) and forgetting
    to give an explicit type.

    See iltf2d.cpp.cpp.

    TODO-Test, since this Python code was not ported from there
    (I used lattice.py in interpretation-sandbox instead),
    some extra testing is needed to verify the code.
    """

    # Make everything relative to p0
    ax1 -= ax0; ay1 -= ay0;
    ax2 -= ax0; ay2 -= ay0;
    bx1 -= bx0; by1 -= by0;
    bx2 -= bx0; by2 -= by0;

    det = ax1*ay2 - ax2*ay1; # The determinant

    if abs(det) < 1.0e-6:
        # If this is used to interpret coords from a ZGY file
        # then caller should catch the exception and either
        # just substitute a default or raise ZgyFormatError
        raise RuntimeError("Colinear or coincident points.")

    for pos in range(len(data)):
        xq = data[pos][0] - ax0;
        yq = data[pos][1] - ay0;
        s = (xq*ay2 - ax2*yq)/det;
        t = (ax1*yq - xq*ay1)/det;
        data[pos][0] = float(bx0 + s*bx1 + t*bx2);
        data[pos][1] = float(by0 + s*by1 + t*by2);

def acpToOcp(orig, inc, size, il, xl, wx, wy):
    """
    Convert 3 arbitrary control points containing annotation- and world coords
    into 4 ordered corner points according to the Petrel Ordered Corner Points
    (OCP) definition, which corresponds to these bulk data indices:
        (          0,           0)
        (size[0] - 1,           0)
        (          0, size[1] - 1)
        (size[0] - 1, size[1] - 1)
    See PetrelOrientationHandling

    This is used to convert from a ThreePoint to a FourPoint definition.
    If the definition is already FourPoint then calling this function with
    the 3 first points should return the same result.

    See OrderedCornerPoints.cpp.
    TODO-Test, since this Python code was not ported from there
    (I used lattice.py in interpretation-sandbox instead),
    some extra testing is needed to verify the code.
    """
    last = [orig[0] + inc[0] * (size[0]-1), orig[1] + inc[1] * (size[1]-1)]
    corners = [[orig[0], orig[1]],
               [last[0], orig[1]],
               [orig[0], last[1]],
               [last[0], last[1]]]
    generalTransform(il[0], xl[0], il[1], xl[1], il[2], xl[2],
                     wx[0], wy[0], wx[1], wy[1], wx[2], wy[2],
                     corners)
    return corners

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
