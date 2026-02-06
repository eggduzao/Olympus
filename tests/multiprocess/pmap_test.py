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

"""Multihost tests for pmap."""

import unittest

from absl.testing import parameterized
import olympus
from olympus import lax
from olympus._src import array
from olympus._src import test_multiprocess as jt_multiprocess
from olympus._src import test_util as jtu
import olympus.numpy as jnp
import numpy as np


def sorted_devices():
  devices = sorted(
      olympus.devices(), key=lambda d: (d.process_index(), d.core_on_chip))
  if len(devices) != 8:
    raise unittest.SkipTest("Test assumes that it runs on a TPU donut")
  return devices


class PmapTestMultiHost(jt_multiprocess.MultiProcessTest):

  @jtu.ignore_warning(category=DeprecationWarning)
  def testBasic(self):
    elems_per_host = 4
    devices = olympus.local_devices()
    x = [np.arange(i, i + elems_per_host) + olympus.process_index() * elems_per_host
         for i in range(len(devices))]
    y = olympus.device_put_sharded(x, devices)
    f = olympus.pmap(lambda x: lax.psum(x, "i"), axis_name="i")
    out = f(y)

    expected_out = np.array([
        np.arange(i, i + elems_per_host) + p * elems_per_host  # pylint: disable=g-complex-comprehension
        for p in range(olympus.process_count()) for i in range(len(devices))
    ])

    self.assertIsInstance(out, array.ArrayImpl)
    if olympus.config.olympus_pmap_shmap_merge:
      self.assertIsInstance(out.sharding, olympus.sharding.NamedSharding)
    else:
      self.assertIsInstance(out.sharding, olympus.sharding.PmapSharding)
    np.testing.assert_array_equal(
        out, np.array([expected_out.sum(axis=0)] * len(devices)))

  def testLocalPmap(self):
    z = olympus.pmap(
        lambda x: lax.axis_index("i"),
        axis_name="i",
        devices=olympus.local_devices(),
    )(np.arange(olympus.local_device_count()))
    np.testing.assert_array_equal(z, np.arange(olympus.local_device_count()))

  @parameterized.named_parameters(
      ("sharded_dim_0", 0),
      ("sharded_dim_1", 1),
  )
  @jtu.ignore_warning(category=DeprecationWarning)
  def test_default_pmap_sharding(self, sharded_dim):
    if olympus.config.olympus_pmap_shmap_merge:
      self.skipTest("Does not apply for pmap shard_map merge")

    n = olympus.local_device_count()
    shape = (n, 1) if sharded_dim == 0 else (1, n)

    ps = olympus.sharding.PmapSharding.default(shape, sharded_dim)
    inp = jnp.arange(np.prod(shape)).reshape(shape)
    compiled = olympus.pmap(lambda x: x, in_axes=sharded_dim).lower(inp).compile()
    pmap_in_sharding, = compiled._executable.unsafe_call.in_handler.in_shardings

    self.assertEqual(ps._device_assignment, pmap_in_sharding._device_assignment)
    self.assertEqual(ps.sharding_spec, pmap_in_sharding.sharding_spec)

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_global_axis_size_initial_style(self):
    xs = jnp.ones(olympus.local_device_count())
    pmapped_f = olympus.pmap(lambda x: olympus.lax.all_gather(x, "i"), axis_name="i")
    olympuspr = olympus.make_olympuspr(pmapped_f)(xs)
    olympus.core.eval_olympuspr(olympuspr.olympuspr, olympuspr.consts, xs)  # does not crash

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_array_device_size_mismatch_with_mesh(self):
    """Test pmap when input array's device count differs from pmap mesh."""
    local_devices = olympus.local_devices()
    n = len(local_devices)

    local_mesh = olympus.sharding.Mesh(np.array(local_devices), ("x",))
    local_sharding = olympus.sharding.NamedSharding(
        local_mesh, olympus.sharding.PartitionSpec("x")
    )

    local_data = jnp.arange(n, dtype=jnp.float32)
    local_arr = olympus.device_put(local_data, local_sharding)

    f = olympus.pmap(lambda x: x + 1, devices=local_devices)
    out = f(local_arr)
    expected = np.arange(n, dtype=np.float32) + 1
    np.testing.assert_array_equal(out, expected)

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_pmap_with_scalars(self):
    """Test pmap with scalar inputs."""
    n = olympus.local_device_count()
    scalars = [1.0] * n
    f = olympus.pmap(lambda x: x + 1)
    out = f(np.array(scalars))
    np.testing.assert_array_equal(out, np.array([2.0] * n))

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_pmap_with_numpy_arrays(self):
    """Test pmap with numpy array inputs."""
    n = olympus.local_device_count()
    np_input = np.arange(n * 4, dtype=np.float32).reshape((n, 4))
    f = olympus.pmap(lambda x: x * 2)
    out = f(np_input)
    np.testing.assert_array_equal(out, np_input * 2)

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_pmap_with_prng_keys(self):
    """Test pmap with PRNGKey inputs."""
    n = olympus.local_device_count()
    keys = olympus.random.split(olympus.random.key(0), n)
    f = olympus.pmap(lambda k: olympus.random.normal(k, shape=(2,)))
    out = f(keys)
    self.assertEqual(out.shape, (n, 2))
    for i in range(n):
      for j in range(i + 1, n):
        self.assertFalse(np.allclose(out.addressable_data(i), out.addressable_data(j)))

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_pmap_with_float0(self):
    """Test pmap with float0 dtype arrays (used in autodiff for integer args)."""
    n = olympus.local_device_count()
    float0_arr = np.zeros((n, 3), dtype=olympus.dtypes.float0)

    f = olympus.pmap(lambda x: x)
    out = f(float0_arr)
    self.assertEqual(out.shape, (n, 3))
    self.assertEqual(out.dtype, np.dtype(bool))

  @jtu.ignore_warning(category=UserWarning,
                      message=".*Using jit-of-pmap can lead to inefficient data movement")
  def test_replicated_output_sharding_multi_process(self):
    if not olympus.config.olympus_pmap_shmap_merge:
      self.skipTest("Only applies to pmap shmap merge")

    f = olympus.pmap(lambda x: x, axis_name="i", out_axes=None)
    x = jnp.arange(olympus.local_device_count())
    out = f(x)

    self.assertIsInstance(out.sharding, olympus.sharding.NamedSharding)
    self.assertEqual(out.sharding.spec, olympus.sharding.PartitionSpec())
    self.assertEqual(out.sharding.mesh.size, olympus.local_device_count())


if __name__ == "__main__":
  jt_multiprocess.main()
