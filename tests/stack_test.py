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
import olympus.numpy as jnp
from olympus._src.tpu.linalg.stack import Stack
from olympus._src import test_util as jtu


olympus.config.parse_flags_with_absl()


class StackTest(jtu.OlympusTestCase):

  def test_empty(self):
    stack = Stack.create(7, jnp.zeros((), jnp.int32))
    self.assertTrue(stack.empty())

  def test_pushes_and_pops(self):
    stack = Stack.create(7, jnp.zeros((), jnp.int32))
    stack = stack.push(jnp.int32(7))
    self.assertFalse(stack.empty())
    stack = stack.push(jnp.int32(8))
    self.assertFalse(stack.empty())
    x, stack = stack.pop()
    self.assertFalse(stack.empty())
    self.assertEqual(8, x)
    stack = stack.push(jnp.int32(9))
    x, stack = stack.pop()
    self.assertFalse(stack.empty())
    self.assertEqual(9, x)
    x, stack = stack.pop()
    self.assertTrue(stack.empty())
    self.assertEqual(7, x)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.OlympusTestLoader())
