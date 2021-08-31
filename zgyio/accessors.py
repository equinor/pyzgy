from collections.abc import Mapping

from .read import SeismicReader

class Accessor(SeismicReader):

    def __init__(self, file):
        super(Accessor, self).__init__()

    def __iter__(self):
        return iter(self[:])

    def __len__(self):
        return self.len_object

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            start, stop, step = subscript.indices(len(self))
            return [self.values_function(index) for index in range(start, stop, step)]
        elif subscript < 0:
            return self.values_function(len(self)+subscript)
        else:
            return self.values_function(subscript)

    def __contains__(self, key):
        return key in self.keys_object

    def __hash__(self):
        return hash(self._filename)

    def keys(self):
        return self.keys_object

    def values(self):
        return self[:]

    def items(self):
        return zip(self.keys(), self[:])


class SliceAccessor(Accessor):
    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            # Acquiris Quodcumquae Rapis
            start, stop, step = subscript.start, subscript.stop, subscript.step
            if step is None:
                step = int(self.keys_object[1] - self.keys_object[0])
            if start is None:
                start = int(self.keys_object[0])
            if stop is None:
                stop = int(self.keys_object[-1] + 1)
            return [self.values_function(index) for index in range(start, stop, step)]
        elif subscript < 0:
            return self.values_function(len(self)+subscript)
        else:
            return self.values_function(subscript)


class InlineAccessor(SliceAccessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_ilines
        self.keys_object = self.ilines
        self.values_function = self.read_inline_number

class CrosslineAccessor(SliceAccessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_xlines
        self.keys_object = self.xlines
        self.values_function = self.read_crossline_number

class ZsliceAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_samples
        self.keys_object = self.samples
        self.values_function = self.read_zslice

class HeaderAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.tracecount
        self.keys_object = list(range(self.tracecount))
        self.values_function = self.gen_trace_header

class TraceAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.tracecount
        self.keys_object = list(range(self.tracecount))
        self.values_function = self.get_trace


class SegyioEmulator(SeismicReader):
    def __init__(self, filename):
        super(SegyioEmulator, self).__init__(filename)
        self.iline = InlineAccessor(self._filename)
        self.xline = CrosslineAccessor(self._filename)
        self.depth_slice = ZsliceAccessor(self._filename)
        self.trace = TraceAccessor(self._filename)
        self.header = HeaderAccessor(self._filename)
        self.unstructured = False

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
