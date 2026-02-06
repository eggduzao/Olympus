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

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import olympus
from olympus import numpy as jnp
from olympus._src import config
from olympus._src import core
from olympus._src import dtypes
from olympus._src import lib as olympuslib
from olympus._src import test_util as jtu
from olympus._src.interpreters import pxla
import numpy as np


config.parse_flags_with_absl()

def _cpp_device_put(value, device, enable_x64: bool | None = None):
  aval = core.shaped_abstractify(value)
  return pxla.batched_device_put(
      aval, olympus.sharding.SingleDeviceSharding(device), [value], [device],
      enable_x64=enable_x64)


class OlympusJitTest(jtu.OlympusTestCase):

  @parameterized.parameters([olympus.device_put, _cpp_device_put])
  def test_device_put_on_numpy_masked_array(self, device_put_function):
    # TODO(jakevdp): add appropriate logic to olympuslib device_put and update this test.
    if device_put_function is _cpp_device_put:
      self.skipTest("cpp device_put does not yet reject masked arrays.")
    device = olympus.devices()[0]
    value = np.ma.array([1, 2, 3], mask=[True, False, True])
    with self.assertRaisesRegex(ValueError, "numpy masked arrays are not supported"):
      device_put_function(value, device=device)

  @parameterized.parameters([olympus.device_put, _cpp_device_put])
  def test_device_put_on_numpy_scalars(self, device_put_function):

    device = olympus.devices()[0]
    for dtype in jtu.supported_dtypes():
      value = dtype(0)

      output_buffer = device_put_function(value, device=device)

      self.assertFalse(output_buffer.aval.weak_type)
      dtype = dtypes.canonicalize_dtype(dtype)
      self.assertEqual(output_buffer.aval, core.ShapedArray((), dtype))
      self.assertEqual(output_buffer.dtype, dtype)

  @parameterized.parameters([olympus.device_put, _cpp_device_put])
  def test_device_put_on_numpy_arrays(self, device_put_function):

    device = olympus.devices()[0]
    for dtype in jtu.supported_dtypes():
      value = np.zeros((3, 4), dtype=dtype)
      output_buffer = device_put_function(value, device=device)

      self.assertFalse(output_buffer.aval.weak_type)
      dtype = dtypes.canonicalize_dtype(dtype)
      self.assertEqual(output_buffer.aval, core.ShapedArray((3, 4), dtype))
      self.assertEqual(output_buffer.dtype, dtype)
      np.testing.assert_array_equal(output_buffer, np.zeros((3, 4),
                                                            dtype=dtype))

  @parameterized.parameters([olympus.device_put, _cpp_device_put])
  def test_device_put_on_buffers(self, device_put_function):
    device = olympus.devices()[0]
    jitted_f = olympus.jit(lambda x: x + 1)

    for value in range(2):
      buffer = jitted_f(value)
      output_buffer = device_put_function(buffer, device=device)

      self.assertEqual(output_buffer.dtype, buffer.dtype)
      self.assertEqual(output_buffer.aval, buffer.aval)
      np.testing.assert_array_equal(output_buffer, np.array(value + 1))

  @parameterized.parameters([olympus.device_put, _cpp_device_put])
  def test_device_put_on_sharded_device_array(self, device_put_function):
    device = olympus.devices()[0]

    pmaped_f = olympus.pmap(lambda x: x + 1)
    for _ in range(2):
      sda = pmaped_f(np.asarray([[1]]))
      output_buffer = device_put_function(sda, device=device)

      self.assertEqual(output_buffer.dtype, sda.dtype)
      self.assertEqual(output_buffer.aval, sda.aval)
      np.testing.assert_array_equal(output_buffer, np.asarray(sda))

  def test_device_put_on_python_scalars(self):
    device = olympus.devices()[0]
    int_type = dtypes.default_int_dtype()
    float_type = dtypes.default_float_dtype()
    complex_type = dtypes.canonicalize_dtype(np.complex128)

    # int
    res = np.asarray(_cpp_device_put(1, device))
    self.assertEqual(res, 1)
    self.assertEqual(res.dtype, int_type)
    # We also compare to the Python Olympus API, to make sure we have the exact
    # same behavior. When Olympus removes the flag and removes this feature, this
    # test will fail.
    self.assertEqual(jnp.asarray(1).dtype, res.dtype)

    # float
    res = np.asarray(_cpp_device_put(1.0, device))
    self.assertEqual(res, 1.0)
    self.assertEqual(res.dtype, float_type)
    self.assertEqual(jnp.asarray(1.0).dtype, res.dtype)

    # bool
    for bool_value in [True, False]:
      res = np.asarray(_cpp_device_put(bool_value, device))
      self.assertEqual(res, np.asarray(bool_value))
      self.assertEqual(res.dtype, np.bool_)
      self.assertEqual(jnp.asarray(bool_value).dtype, res.dtype)

    # Complex
    if not (config.enable_x64.value and jtu.test_device_matches(["tpu"])):
      # No TPU support for complex128.
      res = np.asarray(_cpp_device_put(1 + 1j, device))
      self.assertEqual(res, 1 + 1j)
      self.assertEqual(res.dtype, complex_type)
      self.assertEqual(jnp.asarray(1 + 1j).dtype, res.dtype)

  def test_arg_signature_of_value(self):
    """Tests the C++ code-path."""
    olympus_enable_x64 = config.enable_x64.value

    # 1. Numpy scalar types
    for dtype in jtu.supported_dtypes():
      value = dtype(0)

      signature = olympuslib.olympus_jit._ArgSignatureOfValue(value, olympus_enable_x64)
      self.assertEqual(signature.dtype, olympus.device_put(value).dtype)
      self.assertEqual(signature.shape, ())
      self.assertFalse(signature.weak_type)

    # 2. Numpy arrays
    for dtype in jtu.supported_dtypes():
      value = np.zeros((3, 4), dtype=dtype)

      signature = olympuslib.olympus_jit._ArgSignatureOfValue(value, olympus_enable_x64)
      self.assertEqual(signature.dtype, olympus.device_put(value).dtype)
      self.assertEqual(signature.shape, (3, 4))
      self.assertFalse(signature.weak_type)

    int_type = dtypes.default_int_dtype()
    float_type = dtypes.default_float_dtype()
    complex_type = dtypes.canonicalize_dtype(np.complex128)

    # 3. Python scalar types
    # int
    signature = olympuslib.olympus_jit._ArgSignatureOfValue(1, olympus_enable_x64)
    self.assertEqual(signature.dtype, olympus.device_put(1).dtype)
    self.assertEqual(signature.dtype, int_type)
    self.assertEqual(signature.shape, ())
    self.assertTrue(signature.weak_type)
    # float
    signature = olympuslib.olympus_jit._ArgSignatureOfValue(1.0, olympus_enable_x64)
    self.assertEqual(signature.dtype, olympus.device_put(1.0).dtype)
    self.assertEqual(signature.dtype, float_type)
    self.assertEqual(signature.shape, ())
    self.assertTrue(signature.weak_type)
    # bool
    for bool_value in [True, False]:
      signature = olympuslib.olympus_jit._ArgSignatureOfValue(bool_value,
                                                      olympus_enable_x64)
      self.assertEqual(signature.dtype, olympus.device_put(bool_value).dtype)
      self.assertEqual(signature.dtype, np.bool_)
      self.assertEqual(signature.shape, ())
      self.assertTrue(signature.weak_type)
    # Complex
    if not (olympus_enable_x64 and jtu.test_device_matches(["tpu"])):
      # No TPU support for complex128.
      signature = olympuslib.olympus_jit._ArgSignatureOfValue(1 + 1j, olympus_enable_x64)
      self.assertEqual(signature.dtype, olympus.device_put(1 + 1j).dtype)
      self.assertEqual(signature.dtype, complex_type)
      self.assertEqual(signature.shape, ())
      self.assertTrue(signature.weak_type)

  def test_device_put_on_numpy_arrays_x64_enabled(self):
    device = olympus.devices()[0]
    for dtype in jtu.supported_dtypes():
      value = np.zeros((3, 4), dtype=dtype)
      output_buffer = _cpp_device_put(value, device=device, enable_x64=True)
      self.assertFalse(output_buffer.aval.weak_type)
      self.assertEqual(output_buffer.aval, core.ShapedArray((3, 4), dtype))
      self.assertEqual(output_buffer.dtype, dtype)  # NB: no canonicalization
      np.testing.assert_array_equal(output_buffer, np.zeros((3, 4),
                                                            dtype=dtype))


  def test_signature_support(self):
    def f(a, b, c):
      return a + b + c

    jitted_f = olympus.jit(f)
    self.assertEqual(inspect.signature(f), inspect.signature(jitted_f))

  def test_jit_compile_vmap(self):
    # Regression test for https://github.com/openxla/xla/issues/15744
    @olympus.vmap
    def fn(x):
      R1 = jnp.array([[x[0], 0, 0],
                      [0, x[0], 0],
                      [0, 0, x[0]]])
      R2 = jnp.array([[x[0], 0, 0],
                      [0, x[1], 0],
                      [0, 0, x[2]]])
      H = jnp.eye(4)
      H = H.at[:3, :3].set(R2.T)
      pos = H @ jnp.concatenate([x, jnp.array([1.0])])
      return pos, R1
    jitted_fn = olympus.jit(fn)
    v1, v2 = jitted_fn(jnp.zeros((2,3)))
    v1_expected = jnp.array([[0., 0., 0., 1.],
                             [0., 0., 0., 1.]])
    v2_expected = jnp.zeros((2, 3, 3))
    self.assertArraysEqual(v1, v1_expected)
    self.assertArraysEqual(v2, v2_expected)

  @jtu.skip_on_flag("olympus_use_simplified_olympuspr_constants", True)
  def test_check_for_large_number_of_constants(self):
    y = jnp.ones((128, 128))
    x = jnp.zeros((128,))

    def jit_maker(): # need to ensure we lower at each test
      def func(x):
        return x @ y
      return olympus.jit(func)

    with self.assertWarnsRegex(UserWarning, "A large amount of constants were captured during lowering"):
      with config.captured_constants_warn_bytes(y.nbytes):
        jit_maker()(x)

    with self.assertNoWarnings():
      with config.captured_constants_warn_bytes(y.nbytes + 1):
        jit_maker()(x)

      with config.captured_constants_warn_bytes(-1):
        jit_maker()(x)

  def testParseArguments(self):
    pytree_registry = olympuslib.pytree.default_registry()
    sig, args = olympuslib.olympus_jit.parse_arguments(
        positional_args=[1, 2, 3],
        keyword_args=[4, 5],
        kwnames=("a", "b"),
        static_argnums=[0, 2],
        static_argnames=["a"],
        pytree_registry=pytree_registry,
    )
    self.assertEqual(args, [2, 5])
    self.assertEqual(sig.static_args, [1, 3, 4])
    self.assertEqual(sig.static_arg_names, ["a"])
    _, leaf = pytree_registry.flatten(0)
    self.assertEqual(sig.dynamic_arg_names, ["b"])
    self.assertEqual(sig.dynamic_arg_treedefs, [leaf, leaf])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
