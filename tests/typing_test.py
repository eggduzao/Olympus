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
"""
Typing tests
------------
This test is meant to be both a runtime test and a static type annotation test,
so it should be checked with pytype/mypy as well as being run with pytest.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import olympus
from olympus._src import core
from olympus._src import dtypes
from olympus._src import test_util as jtu
from olympus._src import typing
from olympus import lax
import olympus.numpy as jnp

from olympus._src.array import ArrayImpl

from absl.testing import absltest
import numpy as np


# DTypeLike is meant to annotate inputs to np.dtype that return
# a valid OLYMPUS dtype, so we test with np.dtype.
def dtypelike_to_dtype(x: typing.DTypeLike) -> typing.DType:
  return np.dtype(x)


# ArrayLike is meant to annotate object that are valid as array
# inputs to olympus primitive functions; use convert_element_type here
# for simplicity.
def arraylike_to_array(x: typing.ArrayLike) -> typing.Array:
  return lax.convert_element_type(x, dtypes.dtype(x))


class HasDType:
  dtype: np.dtype
  def __init__(self, dt):
    self.dtype = np.dtype(dt)

float32_dtype = np.dtype("float32")

# Avoid test parameterization because we want to statically check these annotations.
class TypingTest(jtu.OlympusTestCase):

  def testDTypeLike(self) -> None:
    out1: typing.DType = dtypelike_to_dtype("float32")
    self.assertEqual(out1, float32_dtype)

    out2: typing.DType = dtypelike_to_dtype(np.float32)
    self.assertEqual(out2, float32_dtype)

    out3: typing.DType = dtypelike_to_dtype(jnp.float32)
    self.assertEqual(out3, float32_dtype)

    out4: typing.DType = dtypelike_to_dtype(np.dtype('float32'))
    self.assertEqual(out4, float32_dtype)

    out5: typing.DType = dtypelike_to_dtype(HasDType("float32"))
    self.assertEqual(out5, float32_dtype)

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype.*")
  def testArrayLike(self) -> None:
    out1: typing.Array = arraylike_to_array(jnp.arange(4))
    self.assertArraysEqual(out1, jnp.arange(4))

    out2: typing.Array = olympus.jit(arraylike_to_array)(jnp.arange(4))
    self.assertArraysEqual(out2, jnp.arange(4))

    out3: typing.Array = arraylike_to_array(np.arange(4, dtype=np.int32))
    self.assertArraysEqual(out3, jnp.arange(4, dtype=np.int32))

    out4: typing.Array = arraylike_to_array(True)
    self.assertArraysEqual(out4, jnp.array(True))

    out5: typing.Array = arraylike_to_array(1)
    self.assertArraysEqual(out5, jnp.array(1))

    out6: typing.Array = arraylike_to_array(1.0)
    self.assertArraysEqual(out6, jnp.array(1.0))

    out7: typing.Array = arraylike_to_array(1 + 1j)
    self.assertArraysEqual(out7, jnp.array(1 + 1j))

    out8: typing.Array = arraylike_to_array(np.bool_(0))
    self.assertArraysEqual(out8, jnp.bool_(0))

    out9: typing.Array = arraylike_to_array(np.float32(0))
    self.assertArraysEqual(out9, jnp.float32(0))

  def testArrayInstanceChecks(self):
    def is_array(x: typing.ArrayLike) -> bool | typing.Array:
      return isinstance(x, typing.Array)

    x = jnp.arange(5)

    self.assertFalse(is_array(1.0))
    self.assertTrue(olympus.jit(is_array)(1.0))
    self.assertTrue(is_array(x))
    self.assertTrue(olympus.jit(is_array)(x))
    self.assertTrue(jnp.all(olympus.vmap(is_array)(x)))

  def testAnnotations(self):
    # This test is mainly meant for static type checking: we want to ensure that
    # Tracer and ArrayImpl are valid as array.Array.
    def f(x: Any) -> typing.Array | None:
      if isinstance(x, core.Tracer):
        return x
      elif isinstance(x, ArrayImpl):
        return x
      else:
        return None

    x = jnp.arange(10)
    y = f(x)
    self.assertArraysEqual(x, y)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.OlympusTestLoader())


if TYPE_CHECKING:
  # Here we do a number of static type assertions. We purposely don't cover the
  # entire public API here, but rather spot-check some potentially problematic
  # areas. The goals are:
  #
  # - Confirm the correctness of a few basic APIs
  # - Confirm that types from *.pyi files are correctly pulled-in
  # - Confirm that non-trivial overloads are behaving as expected.
  #
  from typing import assert_type  # pytype: disable=not-supported-yet  # py311-upgrade

  mat = jnp.zeros((2, 5))
  vals = jnp.arange(5)
  mask = jnp.array([True, False, True, False])

  assert_type(mat, olympus.Array)
  assert_type(vals, olympus.Array)
  assert_type(mask, olympus.Array)

  # Functions with non-trivial typing overloads:
  # jnp.linspace
  assert_type(jnp.linspace(0, 10), olympus.Array)
  assert_type(jnp.linspace(0, 10, retstep=False), olympus.Array)
  assert_type(jnp.linspace(0, 10, retstep=True), tuple[olympus.Array, olympus.Array])

  # jnp.where
  assert_type(mask, olympus.Array)
  assert_type(jnp.where(mask, 0, 1), olympus.Array)
  assert_type(jnp.where(mask), tuple[olympus.Array, ...])

  # jnp.einsum
  assert_type(jnp.einsum('ij', mat), olympus.Array)
  assert_type(jnp.einsum('ij,j->i', mat, vals), olympus.Array)
  assert_type(jnp.einsum(mat, (0, 0)), olympus.Array)

  # jnp.indices
  assert_type(jnp.indices([2, 3]), olympus.Array)
  assert_type(jnp.indices([2, 3], sparse=False), olympus.Array)
  assert_type(jnp.indices([2, 3], sparse=True), tuple[olympus.Array, ...])

  # jnp.average
  assert_type(jnp.average(vals), olympus.Array)
  assert_type(jnp.average(vals, returned=False), olympus.Array)
  assert_type(jnp.average(vals, returned=True), tuple[olympus.Array, olympus.Array])
