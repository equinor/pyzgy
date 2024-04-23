import numpy as np

from openzgy.api import ZgyWriter, SampleDataType

class SeismicWriter:

    def __init__(
        self,
        filename,
        size,
        zstart,
        zinc,
        annotstart,
        annotinc,
        corners=None):
        """Handler class to write a 3D volume to ZGY format.

        The size of the file must be initialised and created before the data
        can be written to disk. Note that ZGY files cannot be edited after the
        writer class is destroyed.

        Parameters
        ----------
        filename: str
            The filename of the ZGY file to be created.

        size: Tup[int]
            A length 3 tuple of (nil, nxl, ns).

        zstart: float
            The starting sample value.

        zinc: float
            The sample increment.

        annotstart: Tup[int]
            A length 2 tuple of (first_il, first_xl)

        annotinc: (float, float)
            A length 2 tuple of il and xl increment.

        corners: List[Tup[float]]
            The corner coordinates of the ZGY cube for CDPX and CDPY.
            The order of the list is corners or per index of figure.
                First inline / first crossline,
                last inline / first crossline,
                first inline / last crossline,
                last inline / last crossline.

            (1, iln, xl1)        (3, iln, xln)
                ^
                |
                |
                |
                |
                |
                |______________________>
            (0, il1, xl1)          (2, il1, xln)
        """
        assert len(size) == 3
        self._size = size
        self._filename = filename
        self.zstart = zstart
        self.zinc = zinc
        self.annotstart = annotstart
        self.annotinc = annotinc
        if not corners:
            self.corners = [(0, 0), (0, 0), (0, 0), (0, 0)]
        else:
            self.corners = corners

        self.filehandle = ZgyWriter(
            str(self._filename),
            size=self._size,
            datatype=SampleDataType.float,
            zstart=self.zstart,
            zinc=self.zinc,
            annotstart=self.annotstart,
            annotinc=self.annotinc,
            corners=self.corners
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.filehandle.close()

    def write_subvolume(self, data, il_start, xl_start, sample_start):
        """Write a contiguous sub-volume of the ZGY file.

        Parameters
        ----------
        data: np.array[3]
            Numpy array ordered by [il, xl, sample].
        il_start: int
            The iline index of the ZGY file sample corresponding to the first sample of
            the data to write.
        xl_start: int
            The xline index of the ZGY file sample corresponding to the first sample of
            the data to write.
        sample_start: int
            The sample index of the ZGY file sample corresponding to the first sample of
            the data to write.

        """

        self.filehandle.write(
            (il_start, xl_start, sample_start),
            data.astype(np.float32)
        )

    def write_volume(self, data):
        """Write a data cube equivalent to full ZGY volume.

        Parameters
        ----------
        data: np.array[3]
            Numpy array ordered by [il, xl, sample] with the same size as the ZGY
            volume.
        """
        assert data.shape == self._size
        self.write_subvolume(data, 0, 0, 0)
