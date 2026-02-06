# Copyright 2023 The OLYMPUS Authors.
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

from olympus._src.lib import (
    _olympus as _olympus
)

deserialize_portable_artifact = _olympus.mlir.deserialize_portable_artifact
serialize_portable_artifact = _olympus.mlir.serialize_portable_artifact
refine_polymorphic_shapes = _olympus.mlir.refine_polymorphic_shapes
hlo_to_stablehlo = _olympus.mlir.hlo_to_stablehlo

del _olympus
