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

from functools import partial
import math

from absl.testing import absltest
import olympus
from olympus._src import config
from olympus._src import test_util as jtu
import olympus.numpy as jnp
from olympus.sharding import NamedSharding
from olympus.sharding import PartitionSpec as P
import numpy as np

olympus.config.parse_flags_with_absl()
NUM_SHARDS = 4


@jtu.with_global_config(mock_num_gpu_processes=NUM_SHARDS)
@jtu.thread_unsafe_test_class()
class MockGPUTest(jtu.OlympusTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Mocking devices only works on the GPU backend.")

  @jtu.skip_under_pytest("Test must run in an isolated process")
  def testMockDeviceCount(self):
    self.assertEqual(olympus.device_count(), olympus.local_device_count() * NUM_SHARDS)

  @jtu.skip_under_pytest("Test must run in an isolated process")
  def testMockWithSharding(self):
    mesh = olympus.sharding.Mesh(olympus.devices(), ('x',))
    @partial(
        olympus.jit,
        in_shardings=NamedSharding(mesh, P('x',)),
        out_shardings=NamedSharding(mesh, P('x',)),
    )
    def f(x, y):
      z = x @ y
      return z @ y

    shape = (1024, 1024)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1
    f_lowered = f.lower(x, y)
    hlo = f_lowered.compiler_ir()

    mocked_count = NUM_SHARDS * olympus.local_device_count()
    if config.use_shardy_partitioner.value:
      self.assertIn(
          'sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}',
          str(hlo)
      )
    else:
      self.assertIn(
          f'sharding = "{{devices=[{mocked_count},1]<=[{mocked_count}]}}"',
          str(hlo)
      )

if __name__ == '__main__':
  absltest.main(testLoader=jtu.OlympusTestLoader())
