import numpy as np
from openzgy.api import ZgyReader

class SeismicReader:
    def __init__(self, filename):
        self.filename = filename
        self.filehandle = ZgyReader(self.filename)

        self.n_ilines, self.n_xlines, self.n_samples = self.filehandle.size
        self.tracecount = self.n_xlines * self.n_ilines

        self.ilines = np.arange(int(self.filehandle.annotstart[0]),
                                int(self.filehandle.annotstart[0]+self.n_ilines*self.filehandle.annotinc[0]),
                                int(self.filehandle.annotinc[0]))

        self.xlines = np.arange(int(self.filehandle.annotstart[1]),
                                int(self.filehandle.annotstart[1]+self.n_xlines*self.filehandle.annotinc[1]),
                                int(self.filehandle.annotinc[1]))

        self.samples = np.arange(self.filehandle.zstart,
                                 self.filehandle.zstart+self.n_samples*self.filehandle.zinc,
                                 self.filehandle.zinc)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.filehandle.close()

    @staticmethod
    def coord_to_index(coord, coords, include_stop=False):
        try:
            index = np.where(coords == coord)[0][0]
        except:
            if include_stop and (coord == coords[-1] + (coords[-1] - coords[-2])):
                return len(coords)
            raise IndexError("Coordinate {} not in axis".format(coord))
        return index


    def read_inline_number(self, il_no):
        """Reads one inline from ZGY file

        Parameters
        ----------
        il_no : int
            The inline number

        Returns
        -------
        inline : numpy.ndarray of float32, shape: (n_xlines, n_samples)
            The specified inline, decompressed
        """
        return self.read_inline(self.coord_to_index(il_no, self.ilines))

    def read_inline(self, il_idx):
        """Reads one inline from ZGY file

        Parameters
        ----------
        il_id : int
            The ordinal number of the inline in the file

        Returns
        -------
        inline : numpy.ndarray of float32, shape: (n_xlines, n_samples)
            The specified inline, decompressed
        """
        buf = np.zeros((1, self.n_xlines, self.n_samples), dtype=np.float32)
        self.filehandle.read((il_idx, 0, 0), buf)
        return buf.reshape((self.n_xlines, self.n_samples))


    def read_crossline_number(self, xl_no):
        """Reads one crossline from ZGY file

        Parameters
        ----------
        xl_no : int
            The crossline number

        Returns
        -------
        crossline : numpy.ndarray of float32, shape: (n_ilines, n_samples)
            The specified crossline, decompressed
        """
        return self.read_crossline(self.coord_to_index(xl_no, self.xlines))

    def read_crossline(self, xl_idx):
        """Reads one crossline from ZGY file

        Parameters
        ----------
        xl_id : int
            The ordinal number of the crossline in the file

        Returns
        -------
        crossline : numpy.ndarray of float32, shape: (n_ilines, n_samples)
            The specified crossline, decompressed
        """
        buf = np.zeros((self.n_ilines, 1, self.n_samples), dtype=np.float32)
        self.filehandle.read((0, xl_idx, 0), buf)
        return buf.reshape((self.n_ilines, self.n_samples))


    def read_zslice_coord(self, samp_no):
        """Reads one zslice from ZGY file (time or depth, depending on file contents)

        Parameters
        ----------
        zslice_no : int
            The sample time/depth to return a zslice from

        Returns
        -------
        zslice : numpy.ndarray of float32, shape: (n_ilines, n_xlines)
            The specified zslice (time or depth, depending on file contents), decompressed
        """
        return self.read_zslice(self.coord_to_index(samp_no, self.samples))

    def read_zslice(self, z_idx):
        """Reads one zslice from ZGY file (time or depth, depending on file contents)

        Parameters
        ----------
        zslice_id : int
            The ordinal number of the zslice in the file

        Returns
        -------
        zslice : numpy.ndarray of float32, shape: (n_ilines, n_xlines)
            The specified zslice (time or depth, depending on file contents), decompressed
        """
        buf = np.zeros((self.n_ilines, self.n_xlines, 1), dtype=np.float32)
        self.filehandle.read((0, 0, z_idx), buf)
        return buf.reshape((self.n_ilines, self.n_xlines))


    def read_subvolume(self, min_il, max_il, min_xl, max_xl, min_z, max_z):
        """Reads a sub-volume from ZGY file

        Parameters
        ----------
        min_il : int
            The index of the first inline to get from the cube. Use 0 to for the first inline in the cube
        max_il : int
            The index of the last inline to get, non inclusive. To get one inline, use max_il = min_il + 1

        min_xl : int
            The index of the first crossline to get from the cube. Use 0 for the first crossline in the cube
        max_xl : int
            The index of the last crossline to get, non inclusive. To get one crossline, use max_xl = min_xl + 1

        min_z : int
            The index of the first time sample to get from the cube. Use 0 for the first time sample in the cube
        max_z : int
            The index of the last time sample to get, non inclusive. To get one time sample, use max_z = min_z + 1

        access_padding : bool, optional
            Functions which manage voxels used for padding themselves may relax bounds-checking to padded dimensions

        Returns
        -------
        subvolume : numpy.ndarray of float32, shape (max_il - min_il, max_xl - min_xl, max_z - min_z)
            The specified subvolume, decompressed
        """
        buf = np.zeros((max_il-min_il, max_xl-min_xl, max_z-min_z), dtype=np.float32)
        self.filehandle.read((min_il, min_xl, min_z), buf)
        return buf


    def read_volume(self):
        """Reads the whole volume from ZGY file

        Returns
        -------
        volume : numpy.ndarray of float32, shape (n_ilines, n_xline, n_samples)
            The whole volume, decompressed
        """
        return self.read_subvolume(0, self.n_ilines,
                                   0, self.n_xlines,
                                   0, self.n_samples)


    def get_trace(self, index):
        """Reads one trace from ZGY file

        Parameters
        ----------
        index : int
            The ordinal number of the trace in the file

        Returns
        -------
        trace : numpy.ndarray of float32, shape (n_samples)
            A single trace, decompressed
        """
        if not 0 <= index < self.n_ilines * self.n_xlines:
            raise IndexError("Index {} is out of range, total traces is {}".format(index, self.n_ilines * self.n_xlines))

        il, xl = index // self.n_xlines, index % self.n_xlines
        buf = np.zeros((1, 1, self.n_samples), dtype=np.float32)
        self.filehandle.read((il, xl, 0), buf)
        return buf

# Copyright 2021, Equinor
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
