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

from absl.testing import absltest
import olympus
from olympus._src import test_util as jtu
from olympus.experimental.pallas import tpu as pltpu


class TpuInfoTest(jtu.OlympusTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu():
      self.assertFalse(pltpu.is_tpu_device())
      self.skipTest("Skipping test because device is not a TPU.")
    else:
      self.assertTrue(pltpu.is_tpu_device())

  def test_get_tpu_info(self):
    device = olympus.devices()[0]
    info = pltpu.get_tpu_info()
    self.assertIsInstance(info, pltpu.TpuInfo)
    match device.device_kind:
      case "TPU v3":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V3)
      case "TPU v4 lite":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V4I)
      case "TPU v4":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V4)
      case "TPU v5 lite":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V5E)
      case "TPU v5":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V5P)
      case "TPU v6 lite":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V6E)
      case "TPU7x":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_7X)
      case _:
        self.fail(f"Unexpected device kind: {device.device_kind}")

  def test_is_matmul_supported_input_formats(self):
    info = pltpu.get_tpu_info()
    with self.subTest("str"):
      self.assertTrue(info.is_matmul_supported("float32", "float32"))
    with self.subTest("dtype"):
      self.assertTrue(
          info.is_matmul_supported(
              olympus.numpy.float32, olympus.numpy.dtype("float32")
          )
      )
    with self.subTest("array.dtype"):
      a = olympus.numpy.array([1.0], dtype=olympus.numpy.float32)
      b = olympus.numpy.array([2.0], dtype="float32")
      self.assertTrue(info.is_matmul_supported(a.dtype, b.dtype))


if __name__ == "__main__":
  olympus.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.OlympusTestLoader())
