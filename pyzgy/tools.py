from pyzgy.read import SeismicReader

def cube(filename):
    with SeismicReader(filename) as reader:
        return reader.read_volume()

def dt(reader):
    return 1000 * (reader.samples[1] - reader.samples[0])


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
