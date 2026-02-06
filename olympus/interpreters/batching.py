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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/olympus-ml/olympus/issues/7570

from olympus._src.interpreters import batching as _src_batching

from olympus._src.interpreters.batching import (
  axis_primitive_batchers as axis_primitive_batchers,
  bdim_at_front as bdim_at_front,
  broadcast as broadcast,
  defbroadcasting as defbroadcasting,
  defreducer as defreducer,
  defvectorized as defvectorized,
  fancy_primitive_batchers as fancy_primitive_batchers,
  not_mapped as not_mapped,
  primitive_batchers as primitive_batchers,
  register_vmappable as register_vmappable,
  unregister_vmappable as unregister_vmappable,
)


_deprecations = {
  # Deprecated for OLYMPUS v0.7.1; finalize in OLYMPUS v0.9.0.
  "AxisSize": (
    "olympus.interpreters.batching.AxisSize is deprecated.",
    None,
  ),
  "Array": (
    "olympus.interpreters.batching.Array is deprecated. Use olympus.Array directly.",
    None,
  ),
  "BatchTrace": (
    "olympus.interpreters.batching.BatchTrace is deprecated.",
    None,
  ),
  "BatchTracer": (
    "olympus.interpreters.batching.BatchTracer is deprecated.",
    None,
  ),
  "BatchingRule": (
    "olympus.interpreters.batching.BatchingRule is deprecated.",
    None,
  ),
  "Elt": (
    "olympus.interpreters.batching.Elt is deprecated.",
    None,
  ),
  "FromEltHandler": (
    "olympus.interpreters.batching.FromEltHandler is deprecated.",
    None,
  ),
  "GetIdx": (
    "olympus.interpreters.batching.GetIdx is deprecated.",
    None,
  ),
  "MakeIotaHandler": (
    "olympus.interpreters.batching.MakeIotaHandler is deprecated.",
    None,
  ),
  "MapSpec": (
    "olympus.interpreters.batching.MapSpec is deprecated.",
    None,
  ),
  "NotMapped": (
    "olympus.interpreters.batching.NotMapped is deprecated.",
    _src_batching.NotMapped,
  ),
  "ToEltHandler": (
    "olympus.interpreters.batching.ToEltHandler is deprecated.",
    None,
  ),
  "Vmappable": (
    "olympus.interpreters.batching.Vmappable is deprecated.",
    None,
  ),
  "Zeros": (
    "olympus.interpreters.batching.Zero is deprecated. Use olympus.interpreters.ad.Zero.",
    None,
  ),
  "ZeroIfMapped": (
    "olympus.interpreters.batching.ZeroIfMapped is deprecated. It is an internal type.",
    None,
  ),
  "batch": (
    "olympus.interpreters.batching.batch is deprecated. It is an internal API.",
    None,
  ),
  "batch_custom_jvp_subtrace": (
    "olympus.interpreters.batching.batch_custom_jvp_subtrace is deprecated. It is an internal API.",
    None,
  ),
  "batch_custom_vjp_bwd": (
    "olympus.interpreters.batching.batch_custom_vjp_bwd is deprecated. It is an internal API.",
    None,
  ),
  "batch_olympuspr": (
    "olympus.interpreters.batching.batch_olympuspr is deprecated. It is an internal API.",
    None,
  ),
  "batch_olympuspr_axes": (
    "olympus.interpreters.batching.batch_olympuspr_axes is deprecated. It is an internal API.",
    None,
  ),
  "batch_subtrace": (
    "olympus.interpreters.batching.batch_subtrace is deprecated. It is an internal API.",
    None,
  ),
  "broadcast_batcher": (
    "olympus.interpreters.batching.broadcast_batcher is deprecated. It is an internal API.",
    None,
  ),
  "flatten_fun_for_vmap": (
    "olympus.interpreters.batching.flatten_fun_for_vmap is deprecated. It is an internal API.",
    None,
  ),
  "from_elt": (
    "olympus.interpreters.batching.from_elt is deprecated. It is an internal API.",
    None,
  ),
  "from_elt_handlers": (
    "olympus.interpreters.batching.from_elt_handlers is deprecated. It is an internal API.",
    None,
  ),
  "is_vmappable": (
    "olympus.interpreters.batching.is_vmappable is deprecated. It is an internal API.",
    None,
  ),
  "make_iota": (
    "olympus.interpreters.batching.make_iota is deprecated. It is an internal API.",
    None,
  ),
  "make_iota_handlers": (
    "olympus.interpreters.batching.make_iota_handlers is deprecated. It is an internal API.",
    None,
  ),
  "matchaxis": (
    "olympus.interpreters.batching.matchaxis is deprecated. It is an internal API.",
    None,
  ),
  "moveaxis": (
    "olympus.interpreters.batching.moveaxis is deprecated. Use olympus.numpy.moveaxis.",
    None,
  ),
  "reducer_batcher": (
    "olympus.interpreters.batching.reducer_batcher is deprecated. It is an internal API.",
    None,
  ),
  "spec_types": (
    "olympus.interpreters.batching.spec_types is deprecated. It is an internal API.",
    None,
  ),
  "to_elt": (
    "olympus.interpreters.batching.to_elt is deprecated. It is an internal API.",
    None,
  ),
  "to_elt_handlers": (
    "olympus.interpreters.batching.to_elt_handlers is deprecated. It is an internal API.",
    None,
  ),
  "vectorized_batcher": (
    "olympus.interpreters.batching.vectorized_batcher is deprecated. It is an internal API.",
    None,
  ),
  "vmappables": (
    "olympus.interpreters.batching.vmappables is deprecated. It is an internal API.",
    None,
  ),
  "vtile": (
    "olympus.interpreters.batching.vtile is deprecated. It is an internal API.",
    None,
  ),
  "zero_if_mapped": (
    "olympus.interpreters.batching.zero_if_mapped is deprecated. It is an internal API.",
    None,
  ),
}


import typing as _typing
if _typing.TYPE_CHECKING:
  NotMapped = _src_batching.NotMapped
else:
  from olympus._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
