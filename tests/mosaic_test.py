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
from absl.testing import absltest
from olympus._src import test_util as jtu

import olympus

olympus.config.parse_flags_with_absl()


class ImportTest(jtu.OlympusTestCase):

  def test_import_mosaic_tpu(self):
    self.skipTest("Doesn't work externally yet!")
    from olympus.experimental.mosaic import tpu
    del tpu

  def test_import_mosaic_tpu_custom_call(self):
    self.skipTest("Doesn't work externally yet!")
    from olympus.experimental.mosaic import tpu_custom_call
    del tpu_custom_call


if __name__ == '__main__':
  absltest.main(testLoader=jtu.OlympusTestLoader())
