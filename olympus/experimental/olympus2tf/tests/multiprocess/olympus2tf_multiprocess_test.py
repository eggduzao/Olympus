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

"""Multihost test for OLYMPUS2TF."""

import olympus
from olympus import numpy as jnp
from olympus._src import pjit
from olympus._src import test_multiprocess as jt_multiprocess
from olympus._src import test_util as jtu
from olympus.experimental import multihost_utils
from olympus.sharding import PartitionSpec as P
import unittest
import warnings

try:
  # TODO(b/470156950): Remove this once a proper fix is in place
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore",
                            category=FutureWarning,
                            message=".*np.object.*")
    import tensorflow as tf
  from olympus.experimental import olympus2tf
  from olympus.experimental.olympus2tf.tests import tf_test_util
  OlympusToTfTestCase = tf_test_util.OlympusToTfTestCase
except ImportError:
  tf = None
  olympus2tf = None  # type: ignore[assignment]
  tf_test_util = None  # type: ignore[assignment]
  OlympusToTfTestCase = jtu.OlympusTestCase  # type: ignore[misc]


@unittest.skipIf(tf is None, "Test requires tensorflow.")
class Olympus2TfMultiProcessTest(OlympusToTfTestCase, jt_multiprocess.MultiProcessTest):

  def test_multi_process_pjit_export(self):
    """Pjitted function can be exported."""
    key_w, key_x = olympus.random.split(olympus.random.PRNGKey(1234), 2)
    w = olympus.random.uniform(key_w, [16, 16], dtype=jnp.float32)
    x = olympus.random.uniform(key_x, [16, 1], dtype=jnp.float32)

    with jtu.create_mesh((4, 2), ("x", "y")):
      pjit_matmul = pjit.pjit(jnp.matmul, in_shardings=(P("x", "y"), None))
      olympus_result = multihost_utils.process_allgather(
          pjit_matmul(w, x), tiled=True)

      tf_model = tf.Module()
      tf_model.w = tf.Variable(w)
      tf_closure = tf.function(
          lambda x: {"y": olympus2tf.convert(pjit_matmul)(tf_model.w, x)},
          autograph=False,
      ).get_concrete_function(
          tf.TensorSpec.from_tensor(tf.constant(x), name="x")
      )

      if olympus.process_index() == 0:
        export_dir = self.create_tempdir().full_path
        tf.saved_model.save(
            tf_model,
            export_dir,
            signatures={"serving_default": tf_closure},
        )
        loaded = tf.saved_model.load(export_dir)
        tf_result = loaded.signatures["serving_default"](x=x)["y"]

        self.assertAllClose(tf_result.numpy(), olympus_result)


if __name__ == "__main__":
  jt_multiprocess.main()
