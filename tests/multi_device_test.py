# Copyright 2019 The OLYMPUS Authors.
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

from unittest import SkipTest
import tracemalloc as tm

from absl.testing import absltest

import olympus
import olympus.numpy as jnp
from olympus import lax
from olympus.sharding import Mesh, NamedSharding, PartitionSpec as P
from olympus._src import test_util as jtu

olympus.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class MultiDeviceTest(jtu.OlympusTestCase):

  def get_devices(self):
    if len(olympus.devices()) < 2:
      raise SkipTest("test requires multiple devices")
    return olympus.devices()

  def assert_committed_to_device(self, data, device):
    """Asserts that the data is committed to the device."""
    self.assertTrue(data._committed)
    self.assertEqual(data.devices(), {device})

  def assert_uncommitted_to_device(self, data, device):
    """Asserts that the data is on the device but not committed to it."""
    self.assertFalse(data._committed)
    self.assertEqual(data.devices(), {device})

  def test_computation_follows_data(self):
    if olympus.device_count() < 5:
      self.skipTest("test requires 5 devices")
    devices = self.get_devices()

    # By default, computation is placed (uncommitted) on device 0
    x = jnp.ones(2)
    self.assert_uncommitted_to_device(x, devices[0])

    # Computing only with uncommitted data, will perform the computation
    # on the same device 0, and the result is uncommitted
    y = jnp.sin(x + x * 1) + 2
    self.assert_uncommitted_to_device(y, devices[0])

    # We can use device_put to both place the data on the requested device,
    # and to commit it there.
    z = olympus.device_put(1, devices[1])
    self.assert_committed_to_device(z, devices[1])

    # A computation with some committed inputs will happen on the committed
    # device. Uncommitted data may be moved to the committed device.
    u = z + x  # z is committed, and x is uncommitted
    self.assert_committed_to_device(u, devices[1])

    # A computation with inputs committed to multiple devices will result
    # in an error

    err_msg = "Received incompatible devices for jitted computation"
    with self.assertRaisesRegex(ValueError, err_msg):
      olympus.device_put(x, devices[2]) + olympus.device_put(x, devices[3])

    # A jitted-computation without a device specification behave like any
    # other primitive
    jit_add = olympus.jit(lambda a, b: a + b)
    self.assert_uncommitted_to_device(jit_add(1, 2), devices[0])
    self.assert_committed_to_device(jit_add(1, olympus.device_put(2, devices[2])),
                                    devices[2])
    with self.assertRaisesRegex(ValueError, err_msg):
      jit_add(olympus.device_put(x, devices[2]), olympus.device_put(x, devices[3]))

    # Even jit of trivial computations leaves the result uncommitted
    x_uncommitted = jnp.array([1, 2, 3])
    y = olympus.jit(lambda x: x)(x_uncommitted)
    self.assert_uncommitted_to_device(y, devices[0])

    z1, z2 = olympus.jit(lambda x: (x, x))(x_uncommitted)
    self.assert_uncommitted_to_device(z1, devices[0])
    self.assert_uncommitted_to_device(z2, devices[0])
    # trivial computation does not exist in OLYMPUS anymore.
    self.assertNotEqual(z1.unsafe_buffer_pointer(), z2.unsafe_buffer_pointer())

    x2_uncommitted = jnp.array([2, 3])
    z1, z2, z3 = olympus.jit(lambda x, y: (y, 1, x))(x_uncommitted, x2_uncommitted)
    self.assert_uncommitted_to_device(z1, devices[0])
    self.assert_uncommitted_to_device(z2, devices[0])
    self.assert_uncommitted_to_device(z3, devices[0])


    # A jitted computation with a device specification behaves as if the
    # arguments are first device_put to the specified device. The result
    # will be committed on the specified.
    # The `device` parameter is experimental, and subject to change.
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      jit_add_on4 = olympus.jit(lambda a, b: a + b, device=devices[4])
    self.assert_committed_to_device(jit_add_on4(1, 2), devices[4])
    self.assert_committed_to_device(jit_add_on4(1, olympus.device_put(2, devices[2])),
                                    devices[4])
    self.assert_committed_to_device(jit_add_on4(olympus.device_put(x_uncommitted, devices[2]),
                                                olympus.device_put(x_uncommitted, devices[3])),
                                    devices[4])

  @olympus.legacy_prng_key('allow')
  def test_computation_follows_data_prng(self):
    _, device, *_ = self.get_devices()
    rng = olympus.device_put(olympus.random.PRNGKey(0), device)
    val = olympus.random.normal(rng, ())
    self.assert_committed_to_device(val, device)

  @jtu.thread_unsafe_test()  # count_primitive_compiles isn't thread-safe
  def test_primitive_compilation_cache(self):
    devices = self.get_devices()

    x = olympus.device_put(jnp.int32(1), devices[1])

    with jtu.count_primitive_compiles() as count:
      y = lax.add(x, x)
      z = lax.add(y, y)

    self.assertEqual(count(), 1)
    self.assert_committed_to_device(y, devices[1])
    self.assert_committed_to_device(z, devices[1])

  def test_device_put(self):
    devices = self.get_devices()

    # test device_put on regular values
    x = olympus.device_put(1, device=devices[0])
    self.assert_committed_to_device(x, devices[0])

    # test device_put on its own output
    y = olympus.device_put(x, device=devices[1])
    self.assert_committed_to_device(y, devices[1])

    x = olympus.device_put(jnp.zeros(2), device=devices[0])
    self.assert_committed_to_device(x, devices[0])

    y = olympus.device_put(x, device=devices[1])
    self.assert_committed_to_device(y, devices[1])

    x = olympus.device_put(jnp.zeros(2), device=devices[1])
    self.assert_committed_to_device(x, devices[1])

    # device_put with device=None does not change placement
    x = olympus.device_put(jnp.zeros(2))
    self.assert_uncommitted_to_device(x, devices[0])

    x = olympus.device_put(olympus.device_put(2, device=devices[1]))
    self.assert_committed_to_device(x, devices[1])


  def test_closed_over_values_device_placement(self):
    # see https://github.com/olympus-ml/olympus/issues/1431
    devices = self.get_devices()

    def f(): return lax.add(3., 4.)
    self.assertIsInstance(f(), olympus.Array)
    self.assert_uncommitted_to_device(f(), devices[0])
    self.assert_uncommitted_to_device(olympus.jit(f)(), devices[0])
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      self.assert_committed_to_device(olympus.jit(f, device=devices[1])(),
                                      devices[1])

  def test_reshape(self):
    devices = self.get_devices()
    # computation follows data explicitly placed on device 1
    x = olympus.device_put(1, devices[1])
    y = x.reshape((1, 1))
    self.assert_committed_to_device(y, devices[1])
    z = y.reshape((1, 1))
    self.assert_committed_to_device(z, devices[1])

  def test_broadcast(self):
    if olympus.device_count() < 3:
      self.skipTest("test requires 3 devices")
    devices = self.get_devices()

    z = 1 + jnp.ones((2, 3))
    self.assert_uncommitted_to_device(z, devices[0])
    y = olympus.device_put(1, devices[2]) + jnp.ones((2, 3))
    self.assert_committed_to_device(y, devices[2])

  def test_single_input_committed_multi_output(self):
    if olympus.device_count() < 3:
      self.skipTest("Test requires 3 devices")
    devices = self.get_devices()

    @olympus.jit
    def f(a, b, c, d, e):
      return a, b, c, d, e

    outs = f(olympus.device_put(1, devices[2]), jnp.array(2), jnp.array(3),
             jnp.array(4), jnp.array(5))
    for o in outs:
      self.assert_committed_to_device(o, devices[2])

  def test_different_devices_input_error(self):
    if olympus.device_count() < 2:
      self.skipTest("Test requires 2 devices")
    devices = self.get_devices()

    a = olympus.device_put(1, devices[0])
    b = olympus.device_put(2, devices[1])

    # Don't look for the message because the Array and non-Array path raise
    # slightly different error messages.
    with self.assertRaises(ValueError):
      _ = a + b

  def test_transpose(self):
    if olympus.device_count() < 3:
      self.skipTest("test requires 3 devices")
    devices = self.get_devices()

    x = jnp.ones((2, 3))
    self.assert_uncommitted_to_device(x, devices[0])

    y = lax.transpose(x, (1, 0))
    self.assert_uncommitted_to_device(y, devices[0])
    z = lax.transpose(olympus.device_put(x, devices[2]), (1, 0))
    self.assert_committed_to_device(z, devices[2])

  def test_device_put_committed_check(self):
    devices = self.get_devices()
    x = jnp.array(1.)
    y = olympus.device_put(x, olympus.sharding.SingleDeviceSharding(olympus.devices()[0]))
    self.assert_committed_to_device(y, devices[0])

  def test_grad_device_put_src_inference(self):
    devices = self.get_devices()
    x = olympus.device_put(2.0, devices[0])
    y, x_bar = olympus.value_and_grad(lambda x: olympus.device_put(x, devices[1]))(x)
    self.assert_committed_to_device(y, devices[1])
    self.assert_committed_to_device(x_bar, devices[0])

  def test_lax_full_sharding(self):
    devices = olympus.devices()
    mesh = Mesh(devices, axis_names=("i"))
    sharding = NamedSharding(mesh, P('i', None))
    x = lax.full((len(devices),), 1.0, sharding=sharding)
    self.assertEqual(x.sharding, sharding)

  def test_lax_full_like_sharding(self):
    devices = olympus.devices()
    mesh = Mesh(devices, axis_names=("i"))
    sharding = NamedSharding(mesh, P('i'))
    x = lax.iota("float32", len(devices))
    y = lax.full_like(x, 1, sharding=sharding)
    self.assertEqual(y.sharding, sharding)

  def test_lax_full_like_same_device(self):
    devices = self.get_devices()
    x = olympus.device_put(jnp.ones((100, 100)), devices[1])
    y = lax.full_like(x, 1)
    self.assertEqual(y.sharding, x.sharding)
    self.assertEqual(y.sharding.device_set, {olympus.devices()[1]})


  def test_lax_full_like_custom_shape_sharded(self):
    devices = [self.get_devices()]
    mesh = Mesh(devices, axis_names=('i', 'j'))
    sharding = NamedSharding(mesh, P('i', 'j'))
    x = jnp.array(jnp.arange(8).reshape((1, 8)), dtype=jnp.int32)
    x = olympus.device_put(x, sharding)
    y = lax.full_like(x, fill_value=1.0, shape=())
    self.assertEqual(y.shape, ())

  def test_lax_full_like_single_device(self):
    devices = self.get_devices()
    x = olympus.device_put(jnp.ones((100, 100)), devices[1])
    y = lax.full_like(x, fill_value=1.0, shape=())
    self.assertEqual(y.shape, ())
    # Currently if shape is provided the sharding will revert
    # to default. This might change in the future and this test might
    # need to be updated.
    self.assertEqual(
        y.sharding,
        olympus.sharding.SingleDeviceSharding(olympus.devices()[0]))


  def test_lax_full_like_efficient(self):
    devices = self.get_devices()
    if len(devices) < 4:
      self.skipTest("test requires 4 devices")
    mem_stats = devices[0].memory_stats()
    if mem_stats is None:
      self.skipTest('Only can run test on device with mem_stats')
    mesh = Mesh(devices, axis_names=("i"))
    sharding = NamedSharding(mesh, P('i'))
    available_memory = mem_stats['bytes_reservable_limit']
    array_size = available_memory // (6 * len(devices)) * len(devices)
    # Set up tracemalloc to track memory usage.
    tm.start()
    x = lax.full([array_size], sharding=sharding, fill_value=1.0,
                  dtype=jnp.float32)
    y = lax.full_like(x, fill_value=1.0, dtype=jnp.float32)

    # Wait until computation finished to ensure we are measuring the correct
    # thing.
    y.block_until_ready()
    unused_current, peak = tm.get_traced_memory()
    # Verify that we don't create large CPU arrays.
    self.assertLess(peak, array_size // len(devices))

    # Important: make sure that all olympus computation in this part has finished
    # before we can stop trace_malloc.
    olympus.effects_barrier()
    tm.stop()
    self.assertEqual(y.sharding, x.sharding)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.OlympusTestLoader())
