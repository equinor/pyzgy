import pytest
import pathlib
import numpy as np

from pyzgy.accessors import SeismicReader, SeismicWriter

def test_SeismicWriter(zgy_sgy_file_pairs, temp_dir):
    ZGY_FILE, _ = zgy_sgy_file_pairs

    with SeismicReader(ZGY_FILE) as reader:
        data = reader.read_volume()
        zstart = reader.samples[0]
        zinc = reader.zinc
        annotstart = reader.annotstart
        annotinc = reader.annotinc
        corners = reader.corners

    ZGY_FILE_OUT = pathlib.Path(ZGY_FILE).name

    with SeismicWriter(temp_dir/ZGY_FILE_OUT,
                       data.shape,
                       zstart,
                       zinc,
                       annotstart,
                       annotinc,
                       corners=corners
                       ) as writer:
        writer.write_volume(data)
        # write a sub-volume to the same open file
        writer.write_subvolume(data[2:, 2:, 10:30], 2, 2, 10)

    # read back and check
    with SeismicReader(str(temp_dir/ZGY_FILE_OUT)) as reader:
        assert np.allclose(data, reader.read_volume(), rtol=1e-5)
        assert zstart == reader.samples[0]
        assert zinc == reader.zinc
        assert annotstart == reader.annotstart
        assert annotinc == reader.annotinc
        assert corners == reader.corners
