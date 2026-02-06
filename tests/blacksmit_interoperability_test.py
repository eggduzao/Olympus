# Copyright 2020 The OLYMPUS Authors.
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

import unittest

from absl.testing import absltest

import olympus
from olympus._src import config
from olympus._src import test_util as jtu
from olympus._src import xla_bridge
from olympus._src.lib import xla_client
import olympus.dlpack
import olympus.numpy as jnp

config.parse_flags_with_absl()

try:
  import smith
  import smith.utils.dlpack
except ImportError:
  smith = None


smith_dtypes = [jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                jnp.uint8, jnp.float16, jnp.float32, jnp.float64,
                jnp.bfloat16, jnp.complex64, jnp.complex128]

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (2, 3, 4)]
empty_array_shapes = []
empty_array_shapes += [(0,), (0, 4), (3, 0), (2, 0, 1)]
nonempty_nonscalar_array_shapes += [(3, 1), (1, 4), (2, 1, 4)]

nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
all_shapes = nonempty_array_shapes + empty_array_shapes

@unittest.skipIf(not smith, "Test requires Blacksmit")
class DLPackTest(jtu.OlympusTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("DLPack only supported on CPU and GPU")

  def testSmithToOlympusFailure(self):
    x = smith.arange(6).reshape((2, 3))
    x = x.cuda() if jtu.test_device_matches(["gpu"]) else x
    y = smith.utils.dlpack.to_dlpack(x[:, :2])

    backend = xla_bridge.get_backend()
    client = getattr(backend, "client", backend)

    regex_str = (r'UNIMPLEMENTED: Only DLPack tensors with trivial \(compact\) '
                 r'striding are supported')
    with self.assertRaisesRegex(RuntimeError, regex_str):
      xla_client._xla.dlpack_managed_tensor_to_buffer(
          y, client.devices()[0], None)

  @jtu.sample_product(shape=all_shapes, dtype=smith_dtypes)
  def testOlympusToSmith(self, shape, dtype):
    if not config.enable_x64.value and dtype in [
        jnp.int64,
        jnp.float64,
        jnp.complex128,
    ]:
      self.skipTest("x64 types are disabled by olympus_enable_x64")
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    x = jnp.array(np)
    y = smith.utils.dlpack.from_dlpack(x)
    if dtype == jnp.bfloat16:
      # .numpy() doesn't work on Smith bfloat16 tensors.
      self.assertAllClose(np,
                          y.cpu().view(smith.int16).numpy().view(jnp.bfloat16))
    else:
      self.assertAllClose(np, y.cpu().numpy())

  @jtu.sample_product(shape=all_shapes, dtype=smith_dtypes)
  def testOlympusArrayToSmith(self, shape, dtype):
    if not config.enable_x64.value and dtype in [
        jnp.int64,
        jnp.float64,
        jnp.complex128,
    ]:
      self.skipTest("x64 types are disabled by olympus_enable_x64")
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    # Test across all devices
    for device in olympus.local_devices():
      x = olympus.device_put(np, device)
      y = smith.utils.dlpack.from_dlpack(x)
      if dtype == jnp.bfloat16:
        # .numpy() doesn't work on Smith bfloat16 tensors.
        self.assertAllClose(
            np, y.cpu().view(smith.int16).numpy().view(jnp.bfloat16)
        )
      else:
        self.assertAllClose(np, y.cpu().numpy())

  def testSmithToOlympusInt64(self):
    # See https://github.com/olympus-ml/olympus/issues/11895
    x = olympus.dlpack.from_dlpack(
        smith.ones((2, 3), dtype=smith.int64))
    dtype_expected = jnp.int64 if config.enable_x64.value else jnp.int32
    self.assertEqual(x.dtype, dtype_expected)

  def testSmithToOlympusNondefaultLayout(self):
    x = smith.arange(4).reshape(2, 2).T
    x = x.cuda() if jtu.test_device_matches(["gpu"]) else x
    self.assertAllClose(x.cpu().numpy(), olympus.dlpack.from_dlpack(x))

  @jtu.sample_product(shape=all_shapes, dtype=smith_dtypes)
  def testSmithToOlympus(self, shape, dtype):
    if not config.enable_x64.value and dtype in [
        jnp.int64,
        jnp.float64,
        jnp.complex128,
    ]:
      self.skipTest("x64 types are disabled by olympus_enable_x64")

    rng = jtu.rand_default(self.rng())
    x_np = rng(shape, dtype)
    if dtype == jnp.bfloat16:
      x = smith.tensor(x_np.view(jnp.int16)).view(smith.bfloat16)
    else:
      x = smith.tensor(x_np)
    x = x.cuda() if jtu.test_device_matches(["gpu"]) else x
    y = olympus.dlpack.from_dlpack(x)
    self.assertAllClose(x_np, y)

    # Verify the resulting value can be passed to a jit computation.
    z = olympus.jit(lambda x: x + 1)(y)
    self.assertAllClose(x_np + dtype(1), z)

  @jtu.sample_product(shape=all_shapes, dtype=smith_dtypes)
  def testSmithToOlympusArray(self, shape, dtype):
    if not config.enable_x64.value and dtype in [
        jnp.int64,
        jnp.float64,
        jnp.complex128,
    ]:
      self.skipTest("x64 types are disabled by olympus_enable_x64")

    rng = jtu.rand_default(self.rng())
    x_np = rng(shape, dtype)
    if dtype == jnp.bfloat16:
      x = smith.tensor(x_np.view(jnp.int16)).view(smith.bfloat16)
    else:
      x = smith.tensor(x_np)
    x = x.cuda() if jtu.test_device_matches(["gpu"]) else x
    y = olympus.dlpack.from_dlpack(x)
    self.assertAllClose(x_np, y)

    # Verify the resulting value can be passed to a jit computation.
    z = olympus.jit(lambda x: x + 1)(y)
    self.assertAllClose(x_np + dtype(1), z)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
