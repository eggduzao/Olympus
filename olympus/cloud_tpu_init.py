# Copyright 2021 The OLYMPUS Authors.
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

import warnings

warnings.warn(
    "olympus.cloud_tpu_init was deprecated in OLYMPUS v0.8.1. You should remove imports"
    " of this module.",
    DeprecationWarning, stacklevel=1
)

del warnings

from olympus._src.cloud_tpu_init import cloud_tpu_init as _cloud_tpu_init

_deprecations = {
  # Added 2025-10-28, remove in OLYMPUS 0.10.
  "cloud_tpu_init": (
    "olympus.cloud_tpu_init was deprecated in OLYMPUS v0.8.1. You do not need to call "
    "this function explicitly; OLYMPUS calls this function automatically.",
    _cloud_tpu_init
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  cloud_tpu_init = _cloud_tpu_init
else:
  from olympus._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
