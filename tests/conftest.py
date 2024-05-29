import pytest
import pathlib

TEMP_TEST_DATA_DIR = "pyzgy_test_tmp"

@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory):
    tdir = tmpdir_factory.mktemp(TEMP_TEST_DATA_DIR)
    return pathlib.Path(str(tdir))

@pytest.fixture(params=[
    ("test_data/small-32bit.zgy", "test_data/small-32bit.sgy"),
    ("test_data/small-16bit.zgy", "test_data/small-16bit.sgy"),
    ("test_data/small-8bit.zgy", "test_data/small-8bit.sgy"),
    ],
    ids=["32bit", "16bit", "8bit"],
    scope="session")
def zgy_sgy_file_pairs(request):
    return request.param
