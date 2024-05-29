# pyzgy

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/equinor/pyzgy/actions/workflows/python-app.yml/badge.svg)](https://github.com/equinor/pyzgy/actions/workflows/python-app.yml)
[![SCM Compliance](https://scm-compliance-api.radix.equinor.com/repos/equinor/pyzgy/badge)](https://scm-compliance-api.radix.equinor.com/repos/equinor/pyzgy/badge)
[![PyPi Version](https://img.shields.io/pypi/v/pyzgy.svg)](https://pypi.org/project/pyzgy/)

Convenience wrapper around Schlumberger's OpenZGY Python package which enables 
reading of ZGY files with a syntax familiar to users of segyio.

The package also includes native support for loading and writing of ZGY data using Xarray.
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
with pyzgy.open("in.zgy")) as zgyfile:
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
    sub_vol = reader.read_subvolume(
        min_il=min_il, max_il=max_il,
        min_xl=min_xl, max_xl=max_xl,
        min_z=min_z, max_z=max_z
    )
```

#### Write a ZGY file with underlying function ####
```python
import numpy as np
from pyzgy.accessors import SeismicWriter

# create a dummy 10x10x10 cube
data = np.zeros((10, 10, 10))

with SeismicWriter("out.zgy"
    data.shape,
    0.0, # the first sample
    4.0, # the sample increment
    (100, 100), # the first iline and xline labels
    (1, 2), # the iline and xline increments
    ) as writer:
        writer.write_volume(data)
```

#### Native access and writing with Xarray ####
The Xarray Backend engine provides lazy loading support for the volume only. Opening large datasets should be possible, with sub-volume browsing using the native `xarray.Dataset.sel` method.

```python
import xarray as xr

# read a zgy file
zgy = xr.open_dataset("int.zgy")
inline_slice = zgy.sel(iline=LINE_NUMBER)
crossline_slice = zgy.sel(xline=XLINE_NUMBER)
z_slice = zgy.sel(sample=SAMPLE_VALUE)

sub_vol = zgy.sel(
    iline=range(LINE_START,LINE_END),
    xline=range(XLINE_START,XLINE_END),
    sample=range(SAMPLE_START,SAMPLE_END)
)

# write out to zgy file
zgy.pyzgy.to_zgy("out.zgy")

```