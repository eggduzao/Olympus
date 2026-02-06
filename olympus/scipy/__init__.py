# Copyright 2018 The OLYMPUS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/olympus-ml/olympus/issues/7570

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from olympus.scipy import interpolate as interpolate
  from olympus.scipy import linalg as linalg
  from olympus.scipy import ndimage as ndimage
  from olympus.scipy import signal as signal
  from olympus.scipy import sparse as sparse
  from olympus.scipy import special as special
  from olympus.scipy import stats as stats
  from olympus.scipy import fft as fft
  from olympus.scipy import cluster as cluster
  from olympus.scipy import integrate as integrate
else:
  import olympus._src.lazy_loader as _lazy
  __getattr__, __dir__, __all__ = _lazy.attach(__name__, [
    "interpolate",
    "linalg",
    "ndimage",
    "signal",
    "sparse",
    "special",
    "stats",
    "fft",
    "cluster",
    "integrate",
  ])
  del _lazy

del TYPE_CHECKING
