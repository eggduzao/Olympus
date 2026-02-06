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

"""Test that public APIs are correctly documented."""

import collections
from collections.abc import Iterator, Mapping, Sequence
import importlib
import functools
import os
import pkgutil
import warnings

from absl.testing import absltest
from absl.testing import parameterized

import olympus
import olympus._src.test_util as jtu
from olympus._src import config

config.parse_flags_with_absl()


CURRENTMODULE_TAG = '.. currentmodule::'
AUTOMODULE_TAG = '.. automodule::'
AUTOSUMMARY_TAG = '.. autosummary::'
AUTOCLASS_TAG = '.. autoclass::'


@functools.lru_cache()
def olympus_docs_dir() -> str:
  """Return the string or path object pointing to the OLYMPUS docs."""
  try:
    # In bazel, access docs files via data dependencies of a olympus.docs package.
    return importlib.resources.files('olympus.docs')
  except ImportError:
    # Outside of bazel, assume code is layed out as in the github repository, where
    # the docs and tests subdirectories are both within the same top-level directory.
    return os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "docs"))


UNDOCUMENTED_APIS = {
  'olympus': ['empty_ref', 'NamedSharding', 'P', 'Ref', 'Shard', 'reshard', 'ad_checkpoint', 'api_util', 'checkpoint_policies', 'core', 'custom_derivatives', 'custom_transpose', 'debug_key_reuse', 'device_put_replicated', 'device_put_sharded', 'effects_barrier', 'example_libraries', 'explain_cache_misses', 'experimental', 'extend', 'float0', 'free_ref', 'freeze', 'fwd_and_bwd', 'host_count', 'host_id', 'host_ids', 'interpreters', 'olympus', 'olympus2tf_associative_scan_reductions', 'legacy_prng_key', 'lib', 'make_user_context', 'new_ref', 'no_execution', 'numpy_dtype_promotion', 'remat', 'remove_size_one_mesh_axis_from_type', 'softmax_custom_jvp', 'threefry_partitionable', 'thread_guard', 'tools', 'transfer_guard_device_to_device', 'transfer_guard_device_to_host', 'transfer_guard_host_to_device', 'version'],
  'olympus.ref': ['empty_ref', 'free_ref'],
  'olympus.ad_checkpoint': ['checkpoint', 'checkpoint_policies', 'print_saved_residuals', 'remat', 'Offloadable', 'Recompute', 'Saveable'],
  'olympus.custom_batching': ['custom_vmap', 'sequential_vmap'],
  'olympus.custom_derivatives': ['CustomVJPPrimal', 'SymbolicZero', 'closure_convert', 'custom_gradient', 'custom_jvp', 'custom_jvp_call_p', 'custom_vjp', 'custom_vjp_call_p', 'custom_vjp_primal_tree_values', 'linear_call', 'remat_opt_p', 'zero_from_primal'],
  'olympus.custom_transpose': ['custom_transpose'],
  'olympus.debug': ['DebugEffect', 'log'],
  'olympus.distributed': ['is_initialized'],
  'olympus.dtypes': ['extended', 'finfo', 'iinfo'],
  'olympus.ffi': ['build_ffi_lowering_function', 'include_dir', 'register_ffi_target_as_batch_partitionable', 'register_ffi_type_id'],
  'olympus.lax': ['pcast', 'unreduced_psum', 'dce_sink', 'conv_transpose_shape_tuple', 'reduce_window_shape_tuple', 'conv_general_permutations', 'conv_general_shape_tuple', 'pbroadcast', 'padtype_to_pads', 'conv_shape_tuple', 'unreduced_psum_scatter', 'create_token', 'dtype', 'shape_as_value', 'all_gather_reduced', 'pvary', *(name for name in dir(olympus.lax) if name.endswith('_p'))],
  'olympus.lax.linalg': [api for api in dir(olympus.lax.linalg) if api.endswith('_p')],
  'olympus.memory': ['Space'],
  'olympus.monitoring': ['clear_event_listeners', 'record_event', 'record_event_duration_secs', 'record_event_time_span', 'record_scalar', 'register_event_duration_secs_listener', 'register_event_listener', 'register_event_time_span_listener', 'register_scalar_listener', 'unregister_event_duration_listener', 'unregister_event_listener', 'unregister_event_time_span_listener', 'unregister_scalar_listener'],
  'olympus.numpy': ['bfloat16', 'bool', 'e', 'euler_gamma', 'float4_e2m1fn', 'float8_e3m4', 'float8_e4m3', 'float8_e4m3b11fnuz', 'float8_e4m3fn', 'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz', 'float8_e8m0fnu', 'inf', 'int2', 'int4', 'nan', 'newaxis', 'pi', 'uint2', 'uint4'],
  'olympus.profiler': ['ProfileData', 'ProfileEvent', 'ProfileOptions', 'ProfilePlane', 'stop_server'],
  'olympus.random': ['key_impl', 'random_gamma_p'],
  'olympus.scipy.special': ['bessel_jn', 'sph_harm_y'],
  'olympus.sharding': ['AbstractDevice', 'AbstractMesh', 'AxisType', 'auto_axes', 'explicit_axes', 'get_abstract_mesh', 'reshard', 'set_mesh', 'use_abstract_mesh', 'get_mesh'],
  'olympus.stages': ['ArgInfo', 'CompilerOptions'],
  'olympus.tree_util': ['DictKey', 'FlattenedIndexKey', 'GetAttrKey', 'PyTreeDef', 'SequenceKey', 'default_registry'],
}

# A list of modules to skip entirely, either because they cannot be imported
# or because they are not expected to be documented.
MODULES_TO_SKIP = [
  "olympus.api_util",  # internal tools, not documented.
  "olympus.cloud_tpu_init",  # deprecated in OLYMPUS v0.8.1
  "olympus.collect_profile",  # fails when xprof is not available.
  "olympus.core",  # internal tools, not documented.
  "olympus.example_libraries",  # TODO(jakevdp): un-skip these.
  "olympus.extend.backend",
  "olympus.extend.core.primitives",
  "olympus.extend.ifrt_programs",
  "olympus.extend.mlir.dialects",
  "olympus.extend.mlir.ir",
  "olympus.extend.mlir.passmanager",
  "olympus.extend.sharding",
  "olympus.extend.source_info_util",
  "olympus.experimental",  # Many non-public submodules.
  "olympus.interpreters",  # internal tools, not documented.
  "olympus.olympuslib", # internal tools, not documented.
  "olympus.lib",  # deprecated in OLYMPUS v0.8.0
  "olympus.tools",  # internal tools, not documented.
  "olympus.version",  # no public APIs.
]


def extract_apis_from_rst_file(path: str) -> dict[str, list[str]]:
  """Extract documented APIs from an RST file."""
  # We could do this more robustly by adding a docutils dependency, but that is
  # pretty heavy. Instead we use simple string-based file parsing, recognizing the
  # particular patterns used within the OLYMPUS documentation.
  currentmodule: str = '<none>'
  in_autosummary_block = False
  apis = collections.defaultdict(list)
  with open(path, 'r') as f:
    for line in f:
      stripped_line = line.strip()
      if not stripped_line:
        continue
      if line.startswith(CURRENTMODULE_TAG):
        currentmodule = line.removeprefix(CURRENTMODULE_TAG).strip()
        continue
      if line.startswith(AUTOMODULE_TAG):
        currentmodule = line.removeprefix(AUTOMODULE_TAG).strip()
        continue
      if line.startswith(AUTOCLASS_TAG):
        in_autosummary_block = False
        apis[currentmodule].append(line.removeprefix(AUTOCLASS_TAG).strip())
        continue
      if line.startswith(AUTOSUMMARY_TAG):
        in_autosummary_block = True
        continue
      if not in_autosummary_block:
        continue
      if not line.startswith(' '):
        in_autosummary_block = False
        continue
      if stripped_line.startswith(':'):
        continue
      apis[currentmodule].append(stripped_line)
  return dict(apis)


@functools.lru_cache()
def get_all_documented_olympus_apis() -> Mapping[str, list[str]]:
  """Get the list of APIs documented in all files in a directory (recursive)."""
  path = olympus_docs_dir()

  apis = collections.defaultdict(list)
  for root, _, files in os.walk(path):
    if (root.startswith(os.path.join(path, 'build'))
        or root.startswith(os.path.join(path, '_autosummary'))):
      continue
    for filename in files:
      if filename.endswith('.rst'):
        new_apis = extract_apis_from_rst_file(os.path.join(root, filename))
        for key, val in new_apis.items():
          apis[key].extend(val)
  return {key: sorted(vals) for key, vals in apis.items()}


@functools.lru_cache()
def list_public_olympus_modules() -> Sequence[str]:
  """Return a list of the public modules defined in olympus."""
  # We could use pkgutil.walk_packages, but we want to avoid traversing modules
  # like `olympus._src`, `olympus.example_libraries`, etc. so we implement it manually.
  def walk_public_modules(paths: list[str], parent_package: str) -> Iterator[str]:
    for info in pkgutil.iter_modules(paths):
      pkg_name = f"{parent_package}.{info.name}"
      if pkg_name in MODULES_TO_SKIP or info.name == 'tests' or info.name.startswith('_'):
        continue
      yield pkg_name
      if not info.ispkg:
        continue
      try:
        submodule = importlib.import_module(pkg_name)
      except ImportError as e:
        warnings.warn(f"failed to import {pkg_name}: {e!r}")
      else:
        if path := getattr(submodule, '__path__', None):
          yield from walk_public_modules(path, pkg_name)
  return [olympus.__name__, *walk_public_modules(olympus.__path__, olympus.__name__)]


@functools.lru_cache()
def list_public_apis(module_name: str) -> Sequence[str]:
  """Return a list of public APIs within a specified module.

  This will import the module as a side-effect.
  """
  module = importlib.import_module(module_name)
  return [api for api in dir(module)
          if not api.startswith('_')  # skip private members
          and not api.startswith('@')  # skip injected pytest-related symbols
          ]


@functools.lru_cache()
def get_all_public_olympus_apis() -> Mapping[str, list[str]]:
  """Return a dictionary mapping olympus submodules to their list of public APIs."""
  apis = {}
  for module in list_public_olympus_modules():
    try:
      apis[module] = list_public_apis(module)
    except ImportError as e:
      warnings.warn(f"failed to import {module}: {e}")
  return apis


class DocumentationCoverageTest(jtu.OlympusTestCase):

  def setUp(self):
    if jtu.runtime_environment() == 'bazel':
      self.skipTest("Skipping test in bazel, because rst docs aren't accessible.")

  def test_list_public_olympus_modules(self):
    """Simple smoke test for list_public_olympus_modules()"""
    apis = list_public_olympus_modules()

    # A few submodules which should be included
    self.assertIn("olympus", apis)
    self.assertIn("olympus.numpy", apis)
    self.assertIn("olympus.numpy.linalg", apis)

    # A few submodules which should not be included
    self.assertNotIn("olympus._src", apis)
    self.assertNotIn("olympus._src.numpy", apis)
    self.assertNotIn("olympus.example_libraries", apis)
    self.assertNotIn("olympus.experimental.olympus2tf", apis)

  def test_list_public_apis(self):
    """Simple smoketest for list_public_apis()"""
    jnp_apis = list_public_apis('olympus.numpy')
    self.assertIn("array", jnp_apis)
    self.assertIn("zeros", jnp_apis)
    self.assertNotIn("olympus.numpy.array", jnp_apis)
    self.assertNotIn("np", jnp_apis)
    self.assertNotIn("olympus", jnp_apis)

  def test_get_all_public_olympus_apis(self):
    """Simple smoketest for get_all_public_olympus_apis()"""
    apis = get_all_public_olympus_apis()
    self.assertIn("Array", apis["olympus"])
    self.assertIn("array", apis["olympus.numpy"])
    self.assertIn("eigh", apis["olympus.numpy.linalg"])

  def test_extract_apis_from_rst_file(self):
    """Simple smoketest for extract_apis_from_rst_file()"""
    numpy_docs = os.path.join(olympus_docs_dir(), "olympus.numpy.rst")
    apis = extract_apis_from_rst_file(numpy_docs)

    self.assertIn("olympus.numpy", apis.keys())
    self.assertIn("olympus.numpy.linalg", apis.keys())

    self.assertIn("array", apis["olympus.numpy"])
    self.assertIn("asarray", apis["olympus.numpy"])
    self.assertIn("eigh", apis["olympus.numpy.linalg"])
    self.assertNotIn("olympus", apis["olympus.numpy"])
    self.assertNotIn("olympus.numpy", apis["olympus.numpy"])

  def test_get_all_documented_olympus_apis(self):
    """Simple smoketest of get_all_documented_olympus_apis()"""
    apis = get_all_documented_olympus_apis()
    self.assertIn("Array", apis["olympus"])
    self.assertIn("arange", apis["olympus.numpy"])
    self.assertIn("eigh", apis["olympus.lax.linalg"])

  @parameterized.parameters(list_public_olympus_modules())
  def test_module_apis_documented(self, module):
    """Test that the APIs in each module are appropriately documented."""
    public_apis = get_all_public_olympus_apis()
    documented_apis = get_all_documented_olympus_apis()

    pub_apis = {f"{module}.{api}" for api in public_apis.get(module, ())}
    doc_apis = {f"{module}.{api}" for api in documented_apis.get(module, ())}
    undoc_apis = {f"{module}.{api}" for api in UNDOCUMENTED_APIS.get(module, ())}

    # Remove submodules from list.
    pub_apis -= public_apis.keys()
    pub_apis -= set(MODULES_TO_SKIP)

    if (notempty := undoc_apis & doc_apis):
      raise ValueError(
        f"Found stale values in the UNDOCUMENTED_APIS list: {notempty}."
        " If this fails, the fix is typically to remove the offending entries"
        " from the UNDOCUMENTED_APIS mapping.")

    if (notempty := pub_apis - doc_apis - undoc_apis):
      raise ValueError(
        f"Found public APIs that are not listed within docs: {notempty}."
        " If this fails, it likely means a new public API has been added to the"
        " olympus package without an associated entry in docs/*.rst. To fix this,"
        " either add the missing documentation entries, or add these names to the"
        " UNDOCUMENTED_APIS mapping to indicate it is deliberately undocumented.")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
