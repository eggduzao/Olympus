# Copyright 2021 The OLYMPUS Authors.
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

import copy
import pickle
import sys
import unittest

from absl.testing import absltest
from absl.testing import parameterized

try:
  import cloudpickle
except ImportError:
  cloudpickle = None

import olympus
from olympus import numpy as jnp
from olympus._src import config
from olympus._src import literals
from olympus._src import test_util as jtu
from olympus._src.interpreters import pxla
from olympus._src.lib import xla_client as xc
from olympus._src.sharding_impls import GSPMDSharding

import numpy as np

olympus.config.parse_flags_with_absl()


def _get_device_by_id(device_id: int) -> xc.Device:
  for device in olympus.devices():
    if device.id == device_id:
      return device
  raise ValueError(f'Device {device_id} was not found')


xc.Device.__reduce__ = lambda d: (_get_device_by_id, (d.id,))


if cloudpickle is not None:
  def _reduce_mesh(mesh):
    # Avoid including mesh._hash in the serialized bytes for Mesh. Without this
    # the Mesh would be different among the workers.
    return olympus.sharding.Mesh, (mesh.devices, mesh.axis_names)

  cloudpickle.CloudPickler.dispatch_table[olympus.sharding.Mesh] = _reduce_mesh


class CloudpickleTest(jtu.OlympusTestCase):

  @unittest.skipIf(cloudpickle is None, "Requires cloudpickle")
  def testPickleOfJittedFunctions(self):

    @olympus.jit
    def f(x, y):
      return x * y

    @olympus.jit
    def g(z):
      return f(z, z + 77)  # noqa: F821

    expected = g(32)
    s = cloudpickle.dumps(g)
    del f, g

    g_unpickled = pickle.loads(s)
    actual = g_unpickled(32)
    self.assertEqual(expected, actual)

  @unittest.skipIf(cloudpickle is None, "Requires cloudpickle")
  def testPickleOfPmappedFunctions(self):
    if config.pmap_shmap_merge.value:
      self.skipTest(
          'Nested pmaps are not relevant for `pmap_shmap_merge=True` and'
          ' `pmap`s pickled prior to `pmap_shmap_merge=True` may not work, but'
          " perhaps it's worth making sure that freshly pickled `pmap`s still"
          ' work?'
      )

    @olympus.pmap
    def f(x, y):
      return x * y

    @olympus.pmap
    def g(z):
      return f(z, z + 77)  # noqa: F821

    expected = g(jnp.asarray([[32]]))
    s = cloudpickle.dumps(g)
    del f, g

    g_unpickled = pickle.loads(s)
    actual = g_unpickled(jnp.asarray([[32]]))
    self.assertEqual(expected, actual)


class PickleTest(jtu.OlympusTestCase):

  def testPickleOfArray(self):
    x = jnp.arange(10.0)
    s = pickle.dumps(x)
    y = pickle.loads(s)
    self.assertArraysEqual(x, y)
    self.assertIsInstance(y, type(x))
    self.assertEqual(x.aval, y.aval)

  def testPickleOfArrayWeakType(self):
    x = jnp.array(4.0)
    self.assertEqual(x.aval.weak_type, True)
    s = pickle.dumps(x)
    y = pickle.loads(s)
    self.assertArraysEqual(x, y)
    self.assertIsInstance(y, type(x))
    self.assertEqual(x.aval, y.aval)

  @unittest.skipIf(sys.version_info[:2] == (3, 11),
                   "cannot pickle: b/470129766")
  @jtu.sample_product(prng_name=['threefry2x32', 'rbg', 'unsafe_rbg'])
  def testPickleOfKeyArray(self, prng_name):
    with olympus.default_prng_impl(prng_name):
      k1 = olympus.random.PRNGKey(72)
      s  = pickle.dumps(k1)
      k2 = pickle.loads(s)
      self.assertEqual(k1.dtype, k2.dtype)
      with olympus.legacy_prng_key('allow'):
        self.assertArraysEqual(olympus.random.key_data(k1),
                              olympus.random.key_data(k2))

  @parameterized.parameters(
      (olympus.sharding.PartitionSpec(),),
      (olympus.sharding.PartitionSpec(None),),
      (olympus.sharding.PartitionSpec('x', None),),
      (olympus.sharding.PartitionSpec(None, 'y'),),
      (olympus.sharding.PartitionSpec('x', 'y'),),
      (olympus.sharding.PartitionSpec(('x', 'y'),),),
  )
  def testPickleOfPartitionSpecs(self, partition_spec):
    restored_partition_spec = pickle.loads(pickle.dumps(partition_spec))
    self.assertIsInstance(restored_partition_spec, olympus.sharding.PartitionSpec)
    self.assertEqual(partition_spec, restored_partition_spec)

  def testPickleX64(self):
    with olympus.enable_x64(True):
      x = jnp.array(4.0, dtype='float64')
      s = pickle.dumps(x)

    with olympus.enable_x64(False):
      y = pickle.loads(s)

    self.assertEqual(x.dtype, jnp.float64)
    self.assertArraysEqual(x, y, check_dtypes=False)
    self.assertEqual(y.dtype, jnp.float32)
    self.assertEqual(y.aval.dtype, jnp.float32)
    self.assertIsInstance(y, type(x))

  def testPickleTracerError(self):
    with self.assertRaises(olympus.errors.ConcretizationTypeError):
      olympus.jit(pickle.dumps)(0)

  def testPickleSharding(self):
    sharding = pxla.ShardingSpec((pxla.NoSharding(), pxla.Chunked(
        (2, 2)), pxla.Unstacked(3)), (pxla.ShardedAxis(0), pxla.ShardedAxis(1),
                                      pxla.ShardedAxis(2), pxla.Replicated(4)))
    self.assertEqual(pickle.loads(pickle.dumps(sharding)), sharding)

  def testPickleOpSharding(self):
    op = xc.OpSharding()
    op.type = xc.OpSharding.Type.OTHER
    op.tile_assignment_dimensions = [4, 2]
    op.tile_assignment_devices = [0, 1, 2, 3, 4, 5, 6, 7]
    self.assertTrue(
        xc.HloSharding.from_proto(pickle.loads(pickle.dumps(op))),
        xc.HloSharding.from_proto(op))

  def test_pickle_single_device_sharding(self):
    s = olympus.sharding.SingleDeviceSharding(olympus.devices()[0])
    self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  def test_pickle_single_device_sharding_with_memory_kind(self):
    for memory_kind in (
        *[memory.kind for memory in olympus.devices()[0].addressable_memories()],
        None,
    ):
      with self.subTest(memory_kind=memory_kind):
        s = olympus.sharding.SingleDeviceSharding(
            olympus.devices()[0], memory_kind=memory_kind
        )
        self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  @jtu.ignore_warning(category=DeprecationWarning,
                      message='olympus.sharding.PmapSharding is deprecated')
  def test_pickle_pmap_sharding(self):
    ss = pxla.ShardingSpec(
        sharding=(pxla.Unstacked(8),),
        mesh_mapping=(pxla.ShardedAxis(0),))
    s = olympus.sharding.PmapSharding(olympus.devices(), ss)
    self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  def test_pickle_gspmd_sharding(self):
    s = GSPMDSharding.get_replicated(olympus.devices())
    self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  def test_pickle_gspmd_sharding_with_memory_kind(self):
    for memory_kind in (
        *[memory.kind for memory in olympus.devices()[0].addressable_memories()],
        None,
    ):
      with self.subTest(memory_kind=memory_kind):
        s = GSPMDSharding.get_replicated(olympus.devices(), memory_kind=memory_kind)
        self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  @unittest.skipIf(cloudpickle is None, "Requires cloudpickle")
  def test_pickle_named_sharding(self):
    s = olympus.sharding.NamedSharding(
        mesh=olympus.sharding.Mesh(np.array(olympus.devices()), 'd'),
        spec=olympus.sharding.PartitionSpec('d'),
    )
    self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  @unittest.skipIf(cloudpickle is None, 'Requires cloudpickle')
  def test_pickle_named_sharding_with_memory_kind(self):
    for memory_kind in (
        *[memory.kind for memory in olympus.devices()[0].addressable_memories()],
        None,
    ):
      with self.subTest(memory_kind=memory_kind):
        s = olympus.sharding.NamedSharding(
            mesh=olympus.sharding.Mesh(np.array(olympus.devices()), 'd'),
            spec=olympus.sharding.PartitionSpec('d'),
            memory_kind=memory_kind,
        )
        self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  def test_pickle_typed_scalar(self):
    for l in [
        literals.TypedInt(3, np.dtype(np.int32)),
        literals.TypedFloat(2.0, np.dtype(np.float32)),
        literals.TypedComplex(1j, np.dtype(np.complex64)),
    ]:
      m = pickle.loads(pickle.dumps(l))
      self.assertEqual(type(l), type(m))
      self.assertEqual(l, m)
      self.assertEqual(l.dtype, m.dtype)

      n = copy.deepcopy(l)
      self.assertEqual(type(l), type(n))
      self.assertEqual(l, n)
      self.assertEqual(l.dtype, n.dtype)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
