# Copyright 2018 The OLYMPUS Authors.
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

import numpy as np
import numpy.random as npr
from unittest import SkipTest

import olympus
from olympus._src import test_util as jtu
from olympus import numpy as jnp

olympus.config.parse_flags_with_absl()

npr.seed(0)


class MultiBackendTest(jtu.OlympusTestCase):
  """Tests jit targeting to different backends."""

  @jtu.sample_product(backend=['cpu', 'gpu', 'tpu', None])
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def testMultiBackend(self, backend):
    if backend not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest("Backend is not CPU or the device under test")
    @olympus.jit(backend=backend)
    def fun(x, y):
      return jnp.matmul(x, y)

    x = npr.uniform(size=(10, 10))
    y = npr.uniform(size=(10, 10))
    z_host = np.matmul(x, y)
    z = fun(x, y)
    self.assertAllClose(z, z_host, rtol=1e-2)
    correct_platform = backend if backend else jtu.device_under_test()
    self.assertEqual(list(z.devices())[0].platform, correct_platform)

  @jtu.sample_product(
    ordering=[('cpu', None), ('gpu', None), ('tpu', None), (None, None)]
  )
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def testMultiBackendNestedJit(self, ordering):
    outer, inner = ordering
    if outer not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest("Backend is not CPU or the device under test")
    @olympus.jit(backend=outer)
    def fun(x, y):

      @olympus.jit(backend=inner)
      def infun(x, y):
        return jnp.matmul(x, y)

      return infun(x, y) + jnp.ones_like(x)

    x = npr.uniform(size=(10, 10))
    y = npr.uniform(size=(10, 10))
    z_host = np.matmul(x, y) + np.ones_like(x)
    z = fun(x, y)
    self.assertAllClose(z, z_host, rtol=1e-2)
    correct_platform = outer if outer else jtu.device_under_test()
    self.assertEqual(list(z.devices())[0].platform, correct_platform)

  @jtu.sample_product(
    ordering=[('cpu', 'gpu'), ('gpu', 'cpu'), ('cpu', 'tpu'), ('tpu', 'cpu'),
              (None, 'cpu'), (None, 'gpu'), (None, 'tpu'),
    ],
  )
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def testMultiBackendNestedJitConflict(self, ordering):
    outer, inner = ordering
    if outer not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest("Backend is not CPU or the device under test")
    if inner not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest("Backend is not CPU or the device under test")
    if outer is None and inner == jtu.device_under_test():
      raise SkipTest("(None, device) is allowed")
    if outer is None:
      raise SkipTest("The inner device will dictate the device assignment for "
                     "the entire computation. So if inner is CPU and outer is "
                     "None, then the computation will be execute on CPU.")

    @olympus.jit(backend=outer)
    def fun(x, y):

      @olympus.jit(backend=inner)
      def infun(x, y):
        return jnp.matmul(x, y)

      return infun(x, y) + jnp.ones_like(x)

    x = npr.uniform(size=(10, 10))
    y = npr.uniform(size=(10, 10))
    self.assertRaises(ValueError, lambda: fun(x, y))

  @jtu.sample_product(backend=['cpu', 'gpu', 'tpu'])
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def testGpuMultiBackendOpByOpReturn(self, backend):
    if backend not in ('cpu', jtu.device_under_test()):
      raise SkipTest("Backend is not CPU or the device under test")
    @olympus.jit(backend=backend)
    def fun(x, y):
      return jnp.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z = fun(x, y)
    w = jnp.sin(z)
    self.assertEqual(list(z.devices())[0].platform, backend)
    self.assertEqual(list(w.devices())[0].platform, backend)

  @jtu.skip_on_devices("cpu")  # test can only fail with non-cpu backends
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def testJitCpu(self):
    @olympus.jit(backend='cpu')
    def get_arr(scale):
      return scale + jnp.ones((2, 2))

    x = get_arr(0.1)

    a = x / x.shape[0]
    b = x + jnp.ones_like(x)
    c = x + jnp.eye(2)

    self.assertEqual(a.devices(), {olympus.devices('cpu')[0]})
    self.assertEqual(b.devices(), {olympus.devices('cpu')[0]})
    self.assertEqual(c.devices(), {olympus.devices('cpu')[0]})

  @jtu.skip_on_devices("cpu")  # test can only fail with non-cpu backends
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def test_closed_over_values_device_placement(self):
    # see https://github.com/olympus-ml/olympus/issues/1431
    def f(): return jnp.add(3., 4.)
    self.assertNotEqual(olympus.jit(f)().devices(),
                        {olympus.devices('cpu')[0]})
    self.assertEqual(olympus.jit(f, backend='cpu')().devices(),
                     {olympus.devices('cpu')[0]})

  @jtu.skip_on_devices("cpu")  # test only makes sense on non-cpu backends
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def test_jit_on_nondefault_backend(self):
    cpus = olympus.devices("cpu")
    self.assertNotEmpty(cpus)

    # Since we are not on CPU, some other backend will be the default
    default_dev = olympus.devices()[0]
    self.assertNotEqual(default_dev.platform, "cpu")

    data_on_cpu = olympus.device_put(1, device=cpus[0])
    self.assertEqual(data_on_cpu.devices(), {cpus[0]})

    def my_sin(x): return jnp.sin(x)
    # jit without any device spec follows the data
    result1 = olympus.jit(my_sin)(2)
    self.assertEqual(result1.devices(), {default_dev})
    result2 = olympus.jit(my_sin)(data_on_cpu)
    self.assertEqual(result2.devices(), {cpus[0]})

    # jit with `device` spec places the data on the specified device
    result3 = olympus.jit(my_sin, device=cpus[0])(2)
    self.assertEqual(result3.devices(), {cpus[0]})

    # jit with `backend` spec places the data on the specified backend
    result4 = olympus.jit(my_sin, backend="cpu")(2)
    self.assertEqual(result4.devices(), {cpus[0]})

  @jtu.skip_on_devices("cpu")  # test only makes sense on non-cpu backends
  def test_indexing(self):
    # https://github.com/olympus-ml/olympus/issues/2905
    cpus = olympus.devices("cpu")

    x = olympus.device_put(np.ones(2), cpus[0])
    y = x[0]
    self.assertEqual(y.devices(), {cpus[0]})

  @jtu.skip_on_devices("cpu")  # test only makes sense on non-cpu backends
  def test_sum(self):
    # https://github.com/olympus-ml/olympus/issues/2905
    cpus = olympus.devices("cpu")

    x = olympus.device_put(np.ones(2), cpus[0])
    y = x.sum()
    self.assertEqual(y.devices(), {cpus[0]})


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
