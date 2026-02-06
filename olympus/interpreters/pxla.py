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

from olympus._src.interpreters import pxla as _deprecated_pxla
from olympus._src import mesh as _deprecated_mesh
from olympus._src import op_shardings as _deprecated_op_shardings
from olympus._src import sharding_impls as _deprecated_sharding_impls
from olympus._src import sharding_specs as _deprecated_sharding_specs

_deprecations = {
    # deprecated as of OLYMPUS v0.8.2 (Dec 2025)
    "Index": (
        "olympus.interpreters.pxla.Index is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_pxla.Index,
    ),
    "MapTracer": (
        "olympus.interpreters.pxla.MapTracer is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_pxla.MapTracer,
    ),
    "MeshAxisName": (
        (
            "olympus.interpreters.pxla.MeshAxisName is deprecated as of OLYMPUS v0.8.2."
            " Use olympus.sharding.Mesh axis names directly."
        ),
        _deprecated_pxla.MeshAxisName,
    ),
    "MeshComputation": (
        "olympus.interpreters.pxla.MeshComputation is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_pxla.MeshComputation,
    ),
    "MeshExecutable": (
        "olympus.interpreters.pxla.MeshExecutable is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_pxla.MeshExecutable,
    ),
    "PmapExecutable": (
        "olympus.interpreters.pxla.PmapExecutable is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_pxla.PmapExecutable,
    ),
    "global_aval_to_result_handler": (
        (
            "olympus.interpreters.pxla.global_aval_to_result_handler is deprecated"
            " as of OLYMPUS v0.8.2."
        ),
        _deprecated_pxla.global_aval_to_result_handler,
    ),
    "global_avals_to_results_handler": (
        (
            "olympus.interpreters.pxla.global_avals_to_results_handler is"
            " deprecated as of OLYMPUS v0.8.2."
        ),
        _deprecated_pxla.global_avals_to_results_handler,
    ),
    "global_result_handlers": (
        (
            "olympus.interpreters.pxla.global_result_handlers is deprecated as of"
            " OLYMPUS v0.8.2."
        ),
        _deprecated_pxla.global_result_handlers,
    ),
    "parallel_callable": (
        (
            "olympus.interpreters.pxla.parallel_callable is deprecated as of OLYMPUS"
            " v0.8.2."
        ),
        _deprecated_pxla.parallel_callable,
    ),
    "shard_args": (
        "olympus.interpreters.pxla.shard_args is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_pxla.shard_args,
    ),
    "xla_pmap_p": (
        "olympus.interpreters.pxla.xla_pmap_p is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_pxla.xla_pmap_p,
    ),
    "thread_resources": (
        (
            "olympus.interpreters.pxla.thread_resources is deprecated as of OLYMPUS"
            " v0.8.2. Please switch to using `with olympus.set_mesh(mesh)` instead"
            " of `with mesh:` and then use `olympus.sharding.get_abstract_mesh()`"
            " to get the current mesh."
        ),
        _deprecated_mesh.thread_resources,
    ),
    "are_hlo_shardings_equal": (
        (
            "olympus.interpreters.pxla.are_hlo_shardings_equal is deprecated as of"
            " OLYMPUS v0.8.2."
        ),
        _deprecated_op_shardings.are_hlo_shardings_equal,
    ),
    "is_hlo_sharding_replicated": (
        (
            "olympus.interpreters.pxla.is_hlo_sharding_replicated is deprecated as"
            " of OLYMPUS v0.8.2."
        ),
        _deprecated_op_shardings.is_hlo_sharding_replicated,
    ),
    "op_sharding_to_indices": (
        (
            "olympus.interpreters.pxla.op_sharding_to_indices is deprecated as of"
            " OLYMPUS v0.8.2."
        ),
        _deprecated_op_shardings.op_sharding_to_indices,
    ),
    "ArrayMapping": (
        "olympus.interpreters.pxla.ArrayMapping is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_sharding_impls.ArrayMapping,
    ),
    "_UNSPECIFIED": (
        "olympus.interpreters.pxla._UNSPECIFIED is deprecated as of OLYMPUS v0.8.2.",
        _deprecated_sharding_impls.UNSPECIFIED,
    ),
    "array_mapping_to_axis_resources": (
        (
            "olympus.interpreters.pxla.array_mapping_to_axis_resources is"
            " deprecated as of OLYMPUS v0.8.2."
        ),
        _deprecated_sharding_impls.array_mapping_to_axis_resources,
    ),
    "Chunked": (
        (
            "olympus.interpreters.pxla.Chunked is deprecated as of OLYMPUS v0.8.2."
            " Please use `olympus.shard_map` instead of `olympus.pmap`."
        ),
        _deprecated_sharding_specs.Chunked,
    ),
    "NoSharding": (
        (
            "olympus.interpreters.pxla.NoSharding is deprecated as of OLYMPUS v0.8.2."
            " Please use `olympus.shard_map` instead of `olympus.pmap`."
        ),
        _deprecated_sharding_specs.NoSharding,
    ),
    "Replicated": (
        (
            "olympus.interpreters.pxla.Replicated is deprecated as of OLYMPUS v0.8.2."
            " Please use `olympus.shard_map` instead of `olympus.pmap`."
        ),
        _deprecated_sharding_specs.Replicated,
    ),
    "ShardedAxis": (
        (
            "olympus.interpreters.pxla.ShardedAxis is deprecated as of OLYMPUS v0.8.2."
            " Please use `olympus.shard_map` instead of `olympus.pmap`."
        ),
        _deprecated_sharding_specs.ShardedAxis,
    ),
    "ShardingSpec": (
        (
            "olympus.interpreters.pxla.ShardingSpec is deprecated as of OLYMPUS v0.8.2."
            " Please use `olympus.shard_map` instead of `olympus.pmap`."
        ),
        _deprecated_sharding_specs.ShardingSpec,
    ),
    "Unstacked": (
        (
            "olympus.interpreters.pxla.Unstacked is deprecated as of OLYMPUS v0.8.2."
            " Please use `olympus.shard_map` instead of `olympus.pmap`."
        ),
        _deprecated_sharding_specs.Unstacked,
    ),
    "spec_to_indices": (
        (
            "olympus.interpreters.pxla.spec_to_indices is deprecated as of OLYMPUS"
            " v0.8.2. Please use `olympus.shard_map` instead of `olympus.pmap`."
        ),
        _deprecated_sharding_specs.spec_to_indices,
    ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  Index = _deprecated_pxla.Index
  MapTracer = _deprecated_pxla.MapTracer
  MeshAxisName = _deprecated_pxla.MeshAxisName
  MeshComputation = _deprecated_pxla.MeshComputation
  MeshExecutable = _deprecated_pxla.MeshExecutable
  PmapExecutable = _deprecated_pxla.PmapExecutable
  global_aval_to_result_handler = _deprecated_pxla.global_aval_to_result_handler
  global_avals_to_results_handler = _deprecated_pxla.global_avals_to_results_handler
  global_result_handlers = _deprecated_pxla.global_result_handlers
  parallel_callable = _deprecated_pxla.parallel_callable
  shard_args = _deprecated_pxla.shard_args
  xla_pmap_p = _deprecated_pxla.xla_pmap_p
  thread_resources = _deprecated_mesh.thread_resources
  are_hlo_shardings_equal = _deprecated_op_shardings.are_hlo_shardings_equal
  is_hlo_sharding_replicated = _deprecated_op_shardings.is_hlo_sharding_replicated
  op_sharding_to_indices = _deprecated_op_shardings.op_sharding_to_indices
  ArrayMapping = _deprecated_sharding_impls.ArrayMapping
  _UNSPECIFIED = _deprecated_sharding_impls.UNSPECIFIED
  array_mapping_to_axis_resources = _deprecated_sharding_impls.array_mapping_to_axis_resources
  Chunked = _deprecated_sharding_specs.Chunked
  NoSharding = _deprecated_sharding_specs.NoSharding
  Replicated = _deprecated_sharding_specs.Replicated
  ShardedAxis = _deprecated_sharding_specs.ShardedAxis
  ShardingSpec = _deprecated_sharding_specs.ShardingSpec
  Unstacked = _deprecated_sharding_specs.Unstacked
  spec_to_indices = _deprecated_sharding_specs.spec_to_indices
else:
  from olympus._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
