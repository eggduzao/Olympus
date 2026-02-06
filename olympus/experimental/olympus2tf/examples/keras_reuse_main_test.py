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

import os
import warnings

# Must be set before import olympus, as olympus_google.py sets the flag during import.
warnings.filterwarnings(
    'ignore',
    message='Setting `olympus_pmap_shmap_merge` is deprecated',
    category=DeprecationWarning,
)

# pylint: disable=g-import-not-at-top
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import olympus
from olympus._src import test_util as jtu

from olympus.experimental.olympus2tf.examples import keras_reuse_main
from olympus.experimental.olympus2tf.tests import tf_test_util
# pylint: enable=g-import-not-at-top

olympus.config.parse_flags_with_absl()
FLAGS = flags.FLAGS


@jtu.thread_unsafe_test_class()
class KerasReuseMainTest(tf_test_util.OlympusToTfTestCase):

  def setUp(self):
    super().setUp()
    FLAGS.model_path = os.path.join(absltest.get_default_test_tmpdir(),
                                    "saved_models")
    FLAGS.num_epochs = 1
    FLAGS.test_savedmodel = True
    FLAGS.mock_data = True
    FLAGS.show_images = False
    FLAGS.serving_batch_size = 1

  @parameterized.named_parameters(
      dict(testcase_name=f"_{model}", model=model)
      for model in ["mnist_pure_olympus", "mnist_flax"])
  @jtu.ignore_warning(message="the imp module is deprecated")
  def test_keras_reuse(self, model="mnist_pure_olympus"):
    FLAGS.model = model
    keras_reuse_main.main(None)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
