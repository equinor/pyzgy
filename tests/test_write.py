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


def test_SeismicWriter_num_threads(temp_dir):
    from openzgy.api import ZgyReader
    rng = np.random.default_rng(1)
    data = rng.standard_normal((520, 300, 250), dtype=np.float32)
    for n in (1, 4):
        with SeismicWriter(temp_dir/f"threads{n}.zgy", data.shape, 0.0, 1.0,
                           (0, 0), (1.0, 1.0), num_threads=n) as writer:
            writer.write_volume(data)
    with ZgyReader(str(temp_dir/"threads1.zgy")) as r1, \
         ZgyReader(str(temp_dir/"threads4.zgy")) as r4:
        assert r1.nlods == r4.nlods == 5
        assert r1.statistics == r4.statistics
        for lod in range(r1.nlods):
            shape = [(s + (1 << lod) - 1) >> lod for s in r1.size]
            a = np.zeros(shape, np.float32)
            b = np.zeros(shape, np.float32)
            r1.read((0, 0, 0), a, lod=lod)
            r4.read((0, 0, 0), b, lod=lod)
            assert np.array_equal(a, b)
