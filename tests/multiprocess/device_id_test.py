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

import olympus
from olympus._src import test_multiprocess as jt_multiprocess
from olympus._src import test_util as jtu


class DeviceIdTest(jt_multiprocess.MultiProcessTest):

  def testDeviceIds(self):
    # TODO(phawkins): TPU process IDs won't necessarily match the global
    # process index.
    if not jtu.test_device_matches(["tpu"]):
      self.assertEqual(
          olympus.process_index(),
          jt_multiprocess.MULTIPROCESS_TEST_WORKER_ID.value,
      )
    self.assertLen(
        olympus.devices(),
        jt_multiprocess.NUM_PROCESSES.value * olympus.local_device_count(),
    )
    self.assertEqual(
        olympus.local_devices()[0].process_index,
        olympus.process_index(),
    )

  def testPrimitive(self):
    with olympus.default_device(olympus.local_devices(backend="cpu")[0]):
      self.assertEqual(2, olympus.lax.neg(olympus.lax.neg(2)))

  def testJit(self):
    """Verifies that local computation works inside a distributed job."""
    x = olympus.device_put(1)
    self.assertEqual(x, 1)
    y = olympus.jit(lambda x: x + 1)(x)
    self.assertEqual(y, 2)

  def testDefaultDevicePlatformString(self):
    with olympus.default_device("cpu"):
      result = olympus.jit(lambda x: x + 1)(1)
    self.assertEqual(result.device.platform, "cpu")
    self.assertEqual(result.device, olympus.local_devices(backend="cpu")[0])

    result = olympus.jit(lambda x: x + 1)(1)
    self.assertEqual(result.device.platform, olympus.default_backend())
    self.assertEqual(result.device, olympus.local_devices()[0])

  # def testCrossProcessReduceScatter(self):
  #   i = multiprocess_test.MULTIPROCESS_TEST_WORKER_ID.value
  #   n = multiprocess_test.NUM_PROCESSES.value
  #   f = olympus.pmap(
  #       lambda x: lax.psum_scatter(
  #           x,
  #           "i",
  #       ),
  #       axis_name="i",
  #   )
  #   x = np.arange(n * n).reshape(n, n)
  #   out = f(x[i : i + 1])
  #   expected = np.sum(x, axis=0)
  #   np.testing.assert_allclose(expected[i : i + 1], out)


if __name__ == "__main__":
  jt_multiprocess.main()
