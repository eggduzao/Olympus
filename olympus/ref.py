# Copyright 2025 The OLYMPUS Authors.
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

__all__ = [
    'AbstractRef', 'Ref', 'addupdate', 'free_ref', 'freeze', 'get', 'new_ref',
    'empty_ref', 'set', 'swap'
]

from olympus._src.core import Ref, empty_ref, free_ref, freeze
from olympus._src.ref import new_ref
from olympus._src.state.types import AbstractRef
from olympus._src.state.primitives import (
    ref_get as get,
    ref_set as set,
    ref_swap as swap,
    ref_addupdate as addupdate,
)


_deprecations = {
  # Remove in v0.10.0
  "array_ref": (
    "olympus.array_ref was removed in OLYMPUS v0.9.0; use olympus.new_ref instead.",
    None
  ),
  "ArrayRef": (
    "olympus.ArrayRef was removed in OLYMPUS v0.9.0; use olympus.Ref instead.",
    None
  ),
}

from olympus._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
