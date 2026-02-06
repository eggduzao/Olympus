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

olympus.config.parse_flags_with_absl()


class GpuExamplesTest(jtu.OlympusTestCase):


  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Unsupported platform")

    # Import here to avoid trying to load the library when it's not built.
    from olympus_ffi_example import gpu_examples  # pylint: disable=g-import-not-at-top

    self.read_state = gpu_examples.read_state

  def test_basic(self):
    self.assertEqual(self.read_state(), 42)
    self.assertEqual(olympus.jit(self.read_state)(), 42)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
