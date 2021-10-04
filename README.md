# pyzgy

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/equinor/pyzgy/actions/workflows/python-app.yml/badge.svg)](https://github.com/equinor/pyzgy/actions/workflows/python-app.yml)
[![PyPi Version](https://img.shields.io/pypi/v/pyzgy.svg)](https://pypi.org/project/pyzgy/)

Convenience wrapper around Schlumberger's OpenZGY Python package which enables 
reading of ZGY files with a syntax familiar to users of segyio.

---

### Installation

Requires [**openzgy** package from Schlumberger](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/seismic/open-zgy/-/tree/master/python), which is (for now) bundled here under Apache v2.0 license

- Wheels from [PyPI](https://pypi.org/project/pyzgy/): `pip install pyzgy`
- Source from [Github](https://github.com/equinor/pyzgy): `git clone https://github.com/equinor/pyzgy.git`

---

### Usage

#### Use segyio-like interface to read ZGY files ####
```python
import pyzgy
with pyzgy.open("in.vds")) as zgyfile:
    il_slice = zgyfile.iline[zgyfile.ilines[LINE_IDX]]
    xl_slice = zgyfile.xline[LINE_NUMBER]
    zslice = zgyfile.depth_slice[SLICE_IDX]
    trace = zgyfile.trace[TRACE_IDX]
    trace_header = zgyfile.header[TRACE_IDX]
    text_file_header = zgyfile.text[0]
```

#### Read a ZGY file with underlying functions ####
```python
from pyzgy.accessors import SeismicReader
with SeismicReader("in.zgy") as reader:
    inline_slice = reader.read_inline_number(LINE_NUMBER)
    crossline_slice = reader.read_crossline(LINE_IDX)
    z_slice = reader.read_zslice_coord(SLICE_COORD)
    sub_vol = reader.read_subvolume(min_il=min_il, max_il=max_il,
                                    min_xl=min_xl, max_xl=max_xl,
                                    min_z=min_z, max_z=max_z)
```
