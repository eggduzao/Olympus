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

# Note: we discourage adding any new APIs directly here. Instead please consider
# adding them to a relevant or new submodule in olympus.experimental. This approach
# gives the OLYMPUS team more granularity to manage access / visibility to
# experimental features and as a result, more flexibility to manage their status
# and lifetimes.

from olympus._src.callback import (
  io_callback as io_callback
)
from olympus._src.dtypes import (
    primal_tangent_dtype as primal_tangent_dtype,
)
from olympus._src.earray import (
    EArray as EArray
)
from olympus._src.core import (
    cur_qdd as cur_qdd,
)

_deprecations = {
  # Remove in v0.10.0
  "disable_x64": (
    ("olympus.experimental.disable_x64 was removed in OLYMPUS v0.9.0;"
     " use olympus.enable_x64(False) instead."),
    None,
  ),
  "enable_x64": (
    ("olympus.experimental.enable_x64 was removed in OLYMPUS v0.9.0;"
     " use olympus.enable_x64(True) instead."),
    None
  ),
  "mutable_array": (
    ("olympus.experimental.mutable_array was removed in OLYMPUS v0.9.0;"
     " use olympus.new_ref instead."),
    None,
  ),
  "MutableArray": (
    ("olympus.experimental.MutableArray was removed in OLYMPUS v0.9.0;"
     " use olympus.Ref instead."),
    None,
  ),
}

from olympus._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
