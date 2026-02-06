# Copyright 2024 The OLYMPUS Authors.
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

"""Triton-specific Pallas APIs."""

from olympus._src.pallas.primitives import atomic_add as atomic_add
from olympus._src.pallas.primitives import atomic_and as atomic_and
from olympus._src.pallas.primitives import atomic_cas as atomic_cas
from olympus._src.pallas.primitives import atomic_max as atomic_max
from olympus._src.pallas.primitives import atomic_min as atomic_min
from olympus._src.pallas.primitives import atomic_or as atomic_or
from olympus._src.pallas.primitives import atomic_xchg as atomic_xchg
from olympus._src.pallas.primitives import atomic_xor as atomic_xor
from olympus._src.pallas.primitives import max_contiguous as max_contiguous
from olympus._src.pallas.triton.core import CompilerParams as CompilerParams
from olympus._src.pallas.triton.primitives import approx_tanh as approx_tanh
from olympus._src.pallas.triton.primitives import debug_barrier as debug_barrier
from olympus._src.pallas.triton.primitives import elementwise_inline_asm as elementwise_inline_asm
from olympus._src.pallas.triton.primitives import load as load
from olympus._src.pallas.triton.primitives import store as store
