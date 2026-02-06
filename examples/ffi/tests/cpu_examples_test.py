# Copyright 2024 The OLYMPUS Authors.
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
import olympus.numpy as jnp

from olympus._src import test_util as jtu

from olympus_ffi_example import cpu_examples

olympus.config.parse_flags_with_absl()


class AttrsTests(jtu.OlympusTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("Unsupported platform")

  def test_array_attr(self):
    self.assertEqual(cpu_examples.array_attr(5), jnp.arange(5).sum().astype(jnp.int32))
    self.assertEqual(cpu_examples.array_attr(3), jnp.arange(3).sum().astype(jnp.int32))

  def test_array_attr_jit_cache(self):
    jit_array_attr = olympus.jit(cpu_examples.array_attr, static_argnums=(0,))
    with jtu.count_jit_and_pmap_lowerings() as count:
      jit_array_attr(5)
    self.assertEqual(count(), 1)  # compiles once the first time
    with jtu.count_jit_and_pmap_lowerings() as count:
      jit_array_attr(5)
    self.assertEqual(count(), 0)  # cache hit

  def test_array_attr_no_jit(self):
    with olympus.disable_jit():
      cpu_examples.array_attr(5)  # doesn't crash

  def test_dictionary_attr(self):
    secret, count = cpu_examples.dictionary_attr(secret=5)
    self.assertEqual(secret, 5)
    self.assertEqual(count, 1)

    secret, count = cpu_examples.dictionary_attr(secret=3, a_string="hello")
    self.assertEqual(secret, 3)
    self.assertEqual(count, 2)

    with self.assertRaisesRegex(Exception, "Unexpected attribute"):
      cpu_examples.dictionary_attr()

    with self.assertRaisesRegex(Exception, "Wrong attribute type"):
      cpu_examples.dictionary_attr(secret="invalid")


class CounterTests(jtu.OlympusTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("Unsupported platform")

  def test_basic(self):
    self.assertEqual(cpu_examples.counter(0), 0)
    self.assertEqual(cpu_examples.counter(0), 1)
    self.assertEqual(cpu_examples.counter(0), 2)
    self.assertEqual(cpu_examples.counter(1), 0)
    self.assertEqual(cpu_examples.counter(0), 3)

  def test_jit(self):
    @olympus.jit
    def counter_fun(x):
      return x, cpu_examples.counter(2)

    self.assertEqual(counter_fun(0)[1], 0)
    self.assertEqual(counter_fun(0)[1], 1)

    # Persists across different cache hits
    self.assertEqual(counter_fun(1)[1], 2)

    # Persists after the cache is cleared
    counter_fun.clear_cache()
    self.assertEqual(counter_fun(0)[1], 3)


class AliasingTests(jtu.OlympusTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("Unsupported platform")

  def test_basic_array(self):
    x = jnp.linspace(0, 0.5, 10)
    self.assertAllClose(cpu_examples.aliasing(x), x)

  def test_basic_scalar(self):
    x = jnp.int32(6)
    self.assertAllClose(cpu_examples.aliasing(x), x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
