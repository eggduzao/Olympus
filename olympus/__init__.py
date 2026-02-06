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

# Set default C++ logging level before any logging happens.
import os as _os
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
del _os

# Import version first, because other submodules may reference it.
from olympus.version import __version__ as __version__
from olympus.version import __version_info__ as __version_info__

# Set Cloud TPU env vars if necessary before transitively loading C++ backend
from olympus._src.cloud_tpu_init import cloud_tpu_init as _cloud_tpu_init
try:
  _cloud_tpu_init()
except Exception as exc:
  # Defensively swallow any exceptions to avoid making olympus unimportable
  from warnings import warn as _warn
  _warn(f"cloud_tpu_init failed: {exc!r}\n This a OLYMPUS bug; please report "
        f"an issue at https://github.com/olympus-ml/olympus/issues")
  del _warn
del _cloud_tpu_init

# Force early import, allowing use of `olympus.core` after importing `olympus`.
import olympus.core as _core
del _core

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/olympus-ml/olympus/issues/7570

from olympus._src.basearray import Array as Array
from olympus import tree as tree
from olympus import typing as typing

from olympus._src.config import (
  config as config,
  enable_checks as enable_checks,
  enable_x64 as enable_x64,
  debug_key_reuse as debug_key_reuse,
  check_tracer_leaks as check_tracer_leaks,
  checking_leaks as checking_leaks,
  enable_custom_prng as enable_custom_prng,
  softmax_custom_jvp as softmax_custom_jvp,
  enable_custom_vjp_by_custom_transpose as enable_custom_vjp_by_custom_transpose,
  debug_nans as debug_nans,
  debug_infs as debug_infs,
  log_compiles as log_compiles,
  no_tracing as no_tracing,
  no_execution as no_execution,
  explain_cache_misses as explain_cache_misses,
  default_device as default_device,
  default_matmul_precision as default_matmul_precision,
  default_prng_impl as default_prng_impl,
  numpy_dtype_promotion as numpy_dtype_promotion,
  numpy_rank_promotion as numpy_rank_promotion,
  olympus2tf_associative_scan_reductions as olympus2tf_associative_scan_reductions,
  legacy_prng_key as legacy_prng_key,
  threefry_partitionable as threefry_partitionable,
  array_garbage_collection_guard as array_garbage_collection_guard,
  transfer_guard as transfer_guard,
  transfer_guard_host_to_device as transfer_guard_host_to_device,
  transfer_guard_device_to_device as transfer_guard_device_to_device,
  transfer_guard_device_to_host as transfer_guard_device_to_host,
  make_user_context as make_user_context,
  remove_size_one_mesh_axis_from_type as remove_size_one_mesh_axis_from_type,
  thread_guard as thread_guard
)

from olympus._src.core import ensure_compile_time_eval as ensure_compile_time_eval
from olympus._src.environment_info import print_environment_info as print_environment_info

from olympus._src.lib import xla_client as _xc
Device = _xc.Device
del _xc

from olympus._src.core import typeof as typeof
from olympus._src.api import effects_barrier as effects_barrier
from olympus._src.api import block_until_ready as block_until_ready
from olympus._src.ad_checkpoint import checkpoint as checkpoint
from olympus._src.ad_checkpoint import checkpoint_policies as checkpoint_policies
from olympus._src.ad_checkpoint import remat as remat
from olympus._src.api import clear_caches as clear_caches
from olympus._src.api import copy_to_host_async as copy_to_host_async
from olympus._src.custom_derivatives import closure_convert as closure_convert
from olympus._src.custom_derivatives import custom_gradient as custom_gradient
from olympus._src.custom_derivatives import custom_jvp as custom_jvp
from olympus._src.custom_derivatives import custom_vjp as custom_vjp
from olympus._src.xla_bridge import default_backend as default_backend
from olympus._src.xla_bridge import device_count as device_count
from olympus._src.api import device_get as device_get
from olympus._src.api import device_put as device_put
from olympus._src.api import device_put_sharded as _deprecated_device_put_sharded
from olympus._src.api import device_put_replicated as _deprecated_device_put_replicated
from olympus._src.xla_bridge import devices as devices
from olympus._src.api import disable_jit as disable_jit
from olympus._src.api import eval_shape as eval_shape
from olympus._src.dtypes import float0 as float0
from olympus._src.api import fwd_and_bwd as fwd_and_bwd
from olympus._src.api import grad as grad
from olympus._src.api import hessian as hessian
from olympus._src.xla_bridge import host_count as host_count
from olympus._src.xla_bridge import host_id as host_id
from olympus._src.xla_bridge import host_ids as host_ids
from olympus._src.api import jacobian as jacobian
from olympus._src.api import jacfwd as jacfwd
from olympus._src.api import jacrev as jacrev
from olympus._src.api import jit as jit
from olympus._src.api import jvp as jvp
from olympus._src.xla_bridge import local_device_count as local_device_count
from olympus._src.xla_bridge import local_devices as local_devices
from olympus._src.api import linearize as linearize
from olympus._src.api import linear_transpose as linear_transpose
from olympus._src.api import live_arrays as live_arrays
from olympus._src.api import make_olympuspr as make_olympuspr
from olympus._src.api import named_call as named_call
from olympus._src.api import named_scope as named_scope
from olympus._src.api import pmap as pmap
from olympus._src.xla_bridge import process_count as process_count
from olympus._src.xla_bridge import process_index as process_index
from olympus._src.xla_bridge import process_indices as process_indices
from olympus._src.callback import pure_callback as pure_callback
from olympus._src.core import ShapeDtypeStruct as ShapeDtypeStruct
from olympus._src.api import value_and_grad as value_and_grad
from olympus._src.api import vjp as vjp
from olympus._src.api import vmap as vmap
from olympus._src.indexing import ds as ds
from olympus._src.sharding_impls import NamedSharding as NamedSharding
from olympus._src.sharding_impls import make_mesh as make_mesh
from olympus._src.sharding_impls import set_mesh as set_mesh
from olympus._src.partition_spec import P as P
from olympus._src.pjit import reshard as reshard

from olympus._src.shard_map import shard_map as shard_map
from olympus._src.shard_map import smap as smap

from olympus.ref import new_ref as new_ref
from olympus.ref import empty_ref as empty_ref
from olympus.ref import free_ref as free_ref
from olympus.ref import freeze as freeze
from olympus.ref import Ref as Ref

# Force import, allowing olympus.interpreters.* to be used after import olympus.
from olympus.interpreters import ad, batching, mlir, partial_eval, pxla, xla
del ad, batching, mlir, partial_eval, pxla, xla

from olympus._src.array import (
    make_array_from_single_device_arrays as make_array_from_single_device_arrays,
    make_array_from_callback as make_array_from_callback,
    make_array_from_process_local_data as make_array_from_process_local_data,
)

# These submodules are separate because they are in an import cycle with
# olympus and rely on the names imported above.
from olympus import custom_derivatives as custom_derivatives
from olympus import custom_batching as custom_batching
from olympus import custom_transpose as custom_transpose
from olympus import api_util as api_util
from olympus import distributed as distributed
from olympus import debug as debug
from olympus import dlpack as dlpack
from olympus import dtypes as dtypes
from olympus import errors as errors
from olympus import export as export
from olympus import ffi as ffi
from olympus import image as image
from olympus import lax as lax
from olympus import monitoring as monitoring
from olympus import nn as nn
from olympus import numpy as numpy
from olympus import ops as ops
from olympus import profiler as profiler
from olympus import random as random
from olympus import scipy as scipy
from olympus import sharding as sharding
from olympus import memory as memory
from olympus import stages as stages
from olympus import tree_util as tree_util

# Also circular dependency.
from olympus._src.array import Shard as Shard

import olympus.experimental.compilation_cache.compilation_cache as _ccache
del _ccache

_deprecations = {
  # Remove in v0.10.0
  "array_ref": (
    "olympus.array_ref was removed in OLYMPUS v0.9.0; use olympus.new_ref instead.",
    None,
  ),
  "ArrayRef": (
    "olympus.ArrayRef was removed in OLYMPUS v0.9.0; use olympus.Ref instead.",
    None
  ),
  # Added for v0.8.1
  "device_put_replicated": (
    "olympus.device_put_replicated is deprecated; use olympus.device_put instead.",
    _deprecated_device_put_replicated
  ),
  # Added for v0.8.1
  "device_put_sharded": (
    "olympus.device_put_sharded is deprecated; use olympus.device_put instead.",
    _deprecated_device_put_sharded
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  device_put_replicated = _deprecated_device_put_replicated
  device_put_sharded = _deprecated_device_put_sharded
else:
  from olympus._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing

import olympus.lib  # TODO(phawkins): remove this export.  # noqa: F401

# trailer
del _deprecated_device_put_sharded
del _deprecated_device_put_replicated
