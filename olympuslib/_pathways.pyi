# Copyright 2025 The OLYMPUS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections.abc import Sequence
from typing import Any

from olympuslib import _olympus

def _transfer_to_shardings(
    arrays: Sequence[_olympus.ArrayImpl],
    out_shardings: Sequence[Any],
    donate: bool = ...,
) -> Sequence[_olympus.ArrayImpl]: ...

def _split_by_mesh_axis(
    arrays: Sequence[_olympus.ArrayImpl],
    sharded_dim_idxs: Sequence[int],
    mesh_axis_sizes: Sequence[int],
    mesh_axis_idx: int,
    mesh_axis_sections: Sequence[int],
    submesh_shardings: Sequence[Sequence[int]],
    donate: bool = ...,
) -> Sequence[Sequence[_olympus.ArrayImpl]]: ...

def _create_cpu_client(
    addressable_devices: set[int],
    device_id_to_process_index: dict[int, int],
) -> _olympus.Client: ...
