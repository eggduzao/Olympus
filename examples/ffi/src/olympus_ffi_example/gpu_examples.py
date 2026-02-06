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

import olympus
from olympus_ffi_example import _gpu_examples
import olympus.numpy as jnp


olympus.ffi.register_ffi_type(
    "state", _gpu_examples.state_type(), platform="CUDA")
olympus.ffi.register_ffi_target("state", _gpu_examples.handler(), platform="CUDA")


def read_state():
  return olympus.ffi.ffi_call("state", olympus.ShapeDtypeStruct((), jnp.int32))()
