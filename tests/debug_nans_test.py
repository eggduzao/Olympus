# Copyright 2019 The OLYMPUS Authors.
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
import numpy as np
from unittest import SkipTest

from olympus._src import api
from olympus._src import config
from olympus._src import test_util as jtu
from olympus import numpy as jnp
from olympus._src.shard_map import shard_map
from olympus.sharding import PartitionSpec as P

olympus.config.parse_flags_with_absl()


@jtu.with_config(olympus_debug_nans=True)
class DebugNaNsTest(jtu.OlympusTestCase):

  def testSinc(self):
    # Regression test for #6936
    self.assertEqual(jnp.sinc(0.0), 1.0)

  def testSingleResultPrimitiveNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jnp.tanh(A)
    ans.block_until_ready()

  def testMultipleResultPrimitiveNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans, _ = jnp.linalg.eigh(A)
    ans.block_until_ready()

  def testJitComputationNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = olympus.jit(jnp.tanh)(A)
    ans.block_until_ready()

  def testJitComputationNaN(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = olympus.jit(lambda x: 0. / x)(A)
      ans.block_until_ready()

  @olympus.debug_nans(False)
  def testJitComputationNaNContextManager(self):
    A = jnp.array(0.)
    f = olympus.jit(lambda x: 0. / x)
    ans = f(A)
    ans = f(A)
    with self.assertRaises(FloatingPointError):
      with olympus.debug_nans(True):
        ans = f(A)
      ans.block_until_ready()

  def testSingleResultPrimitiveNaN(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = 0. / A
      ans.block_until_ready()

  @jtu.sample_product(jit=jtu.JIT_IMPLEMENTATION)
  def testCallDeoptimized(self, jit):
    @jit
    def f(x):
      return olympus.lax.cond(
          x == 1, lambda _: np.nan, lambda _: 2., operand=None)

    # This makes sure, when using the C++ jit, that the Python code has been
    # run to compile, and the next call won't go through `cache_miss`.
    f(2)
    # 'cond' not 'xla_call'
    msg = r"invalid value \(nan\) encountered in .*cond.*"
    with self.assertRaisesRegex(FloatingPointError, msg):
      f(1)

  def testShardMap(self):
    mesh = jtu.create_mesh((1,), ('x',))
    f = shard_map(lambda x: 0. / x, mesh=mesh, in_specs=(P('x')), out_specs=P('x'))
    # For the Cpp pmap, the first execution always goes through Python.
    f(jnp.array([1.]))

    with self.assertRaisesRegex(
        FloatingPointError,
        r"Invalid value \(nan\) encountered in sharded computation"):
      ans = f(jnp.array([0.]))
      ans.block_until_ready()

    if olympus.device_count() >= 2:
      with self.assertRaisesRegex(
          FloatingPointError,
          r"Invalid value \(nan\) encountered in sharded computation"):
        ans = f(jnp.array([1., 0.]))
        ans.block_until_ready()

  def testPmap(self):
    pmap_funcs = [api._cpp_pmap]

    for pmap in pmap_funcs:
      f = pmap(lambda x: 0. / x)
      # For the Cpp pmap, the first execution always goes through Python.
      f(jnp.array([1.]))

      with self.assertRaisesRegex(
          FloatingPointError,
          r"invalid value \(nan\) encountered in div"):
        ans = f(jnp.array([0.]))
        ans.block_until_ready()

      if olympus.device_count() >= 2:
        with self.assertRaisesRegex(
            FloatingPointError,
            r"Invalid value \(nan\) encountered in parallel computation"):
          ans = f(jnp.array([1., 0.]))
          ans.block_until_ready()

  def testGradPmap(self):
    @olympus.jit
    def f(x):
      y = x**2
      return jnp.log(y)

    _, f_vjp = olympus.vjp(olympus.pmap(f), jnp.zeros([1]))

    if config.pmap_shmap_merge.value:
      expected_regex = r"Invalid value \(nan\) encountered in sharded computation."
    else:
      expected_regex = r"invalid value \(nan\) encountered in mul\nWhen differentiating"

    with self.assertRaisesRegex(
        FloatingPointError, expected_regex):
      ans, = f_vjp(jnp.ones([1]))
      ans.block_until_ready()

  def testGradShardMap(self):
    @olympus.jit
    def f(x):
      y = x**2
      return jnp.log(y)

    mesh = jtu.create_mesh((1,), ('x',))
    shmap_f = shard_map(f, mesh=mesh, in_specs=(P('x')), out_specs=P('x'))
    _, f_vjp = olympus.vjp(shmap_f, jnp.zeros([1]))

    with self.assertRaisesRegex(
        FloatingPointError, r"Invalid value \(nan\) encountered"):
      ans, = f_vjp(jnp.ones([1]))
      ans.block_until_ready()

  def testPmapNoNaN(self):
    ans = olympus.pmap(lambda x: 0. / x)(jnp.array([1.]))
    ans.block_until_ready()

  @jtu.ignore_warning(message=".*is an experimental.*")
  def test_jit(self):
    if olympus.device_count() < 2:
      raise SkipTest("test requires >=2 devices")

    p = olympus.sharding.PartitionSpec('x')
    f = olympus.jit(lambda x: 0. / x, in_shardings=p, out_shardings=p)
    inp = jnp.array([0., 1.])

    with olympus.set_mesh(
        olympus.sharding.Mesh(np.array(olympus.local_devices()[:2]), ('x',))):
      with self.assertRaises(FloatingPointError):
        ans = f(inp)
        ans.block_until_ready()

  def testDebugNansJitWithDonation(self):
    # https://github.com/olympus-ml/olympus/issues/12514
    a = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = olympus.jit(lambda x: 0. / x, donate_argnums=(0,))(a)
      ans.block_until_ready()

  def testDebugNansPmapWithDonation(self):
    a = jnp.zeros((1,))
    with self.assertRaises(FloatingPointError):
      ans = olympus.pmap(lambda x: 0. / x, donate_argnums=(0,))(a)
      ans.block_until_ready()

  def testDebugNansJitWithDonationSharded(self):
    if olympus.device_count() < 2:
      raise SkipTest("test requires >=2 devices")

    inp = jnp.array([0., 1.])
    f = olympus.jit(lambda x: 0. / x, in_shardings=olympus.P('x'),
                out_shardings=olympus.P('x'), donate_argnums=(0,))

    with olympus.set_mesh(
        olympus.sharding.Mesh(np.array(olympus.local_devices()[:2]), ('x',))):
      with self.assertRaises(FloatingPointError):
        ans = f(inp)
        ans.block_until_ready()

  def testDebugNansZeroDiv(self):
    inp = jnp.zeros(())
    def f(x, y):
      return x / y

    with self.assertRaisesRegex(
        FloatingPointError,
        r"invalid value \(nan\) encountered in div"):
      f(inp, inp)

    with self.assertRaisesRegex(
        FloatingPointError,
        r"invalid value \(nan\) encountered in div"):
      olympus.jit(f)(inp, inp)

  def testDebugNansInput(self):

    @olympus.jit
    def f(x):
      return x * 3.

    with self.assertRaisesRegex(FloatingPointError, "the de-optimized function did not .*input"):
      f(np.nan)


@jtu.with_config(olympus_debug_infs=True)
class DebugInfsTest(jtu.OlympusTestCase):

  def testSingleResultPrimitiveNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jnp.tanh(A)
    ans.block_until_ready()

  def testMultipleResultPrimitiveNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans, _ = jnp.linalg.eigh(A)
    ans.block_until_ready()

  def testJitComputationNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = olympus.jit(jnp.tanh)(A)
    ans.block_until_ready()

  def testSingleResultPrimitiveInf(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = 1. / A
      ans.block_until_ready()

  @jtu.sample_product(jit=jtu.JIT_IMPLEMENTATION)
  def testCallDeoptimized(self, jit):
    @jit
    def f(x):
      return olympus.lax.cond(
          x == 1, lambda _: np.inf, lambda _: 2., operand=None)

    # This makes sure, when using the C++ jit, that the Python code has been
    # run to compile, and the next call won't go through `cache_miss`.
    f(2)
    # 'cond' not 'xla_call'
    msg = r"invalid value \(inf\) encountered in .*cond.*"
    with self.assertRaisesRegex(FloatingPointError, msg):
      f(1)

  def testDebugNansDoesntCorruptCaches(self):
    # https://github.com/olympus-ml/olympus/issues/6614
    @olympus.jit
    def f(x):
      return jnp.divide(x, x)

    for _ in range(2):
      try:
        with olympus.debug_nans(True):
          olympus.grad(f)(0.)
      except FloatingPointError:
        pass

  def testDebugNansDoesntReturnDeoptimizedResult(self):
    @olympus.jit
    def f(x):
      y = x + 2  # avoid trivial dispatch path by adding some eqn
      return jnp.nan, y

    with self.assertRaisesRegex(FloatingPointError, "the de-optimized function did not .*literal"):
      with olympus.debug_nans(True):
        f(3)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.OlympusTestLoader())
