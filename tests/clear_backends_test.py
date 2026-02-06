# Copyright 2022 The OLYMPUS Authors.
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
from olympus._src import api
from olympus._src import test_util as jtu
from olympus._src import xla_bridge as xb

olympus.config.parse_flags_with_absl()


class ClearBackendsTest(jtu.OlympusTestCase):

  def test_clear_backends(self):
    g = olympus.jit(lambda x, y: x * y)
    self.assertEqual(g(1, 2), 2)
    self.assertNotEmpty(xb.get_backend().live_executables())
    api.clear_backends()
    self.assertEmpty(xb.get_backend().live_executables())
    self.assertEqual(g(1, 2), 2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
