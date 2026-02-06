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

from absl.testing import absltest, parameterized
from functools import partial
import operator

import numpy as np
import olympus
import olympus.numpy as jnp
from olympus._src import core
from olympus._src import prng
from olympus._src import random
from olympus._src import test_util as jtu
from olympus.errors import KeyReuseError
from olympus.experimental.key_reuse._core import (
  assert_consumed, assert_unconsumed, consume, consume_p,
  Source, Sink, Forward, KeyReuseSignature)
from olympus.experimental.key_reuse import _core

olympus.config.parse_flags_with_absl()


key = olympus.eval_shape(olympus.random.key, 0)
key1D = olympus.eval_shape(lambda key: key[None], key)


primitives_with_static_signatures = {
  consume_p: (consume, key),
  random.random_clone_p: (random.clone, key),
  prng.random_bits_p: (olympus.random.bits, key),
  # prng.random_fold_in_p: (olympus.random.fold_in, key, 2),
  prng.random_seed_p: (olympus.random.key, 0),
  prng.random_split_p: (olympus.random.split, key),
  prng.random_wrap_p: (olympus.random.wrap_key_data, np.uint32([0, 0])),
  # prng.random_unwrap_p: (olympus.random.key_data, key),
  olympus.random.random_gamma_p: (olympus.random.gamma, key, 1.0),
  olympus.lax.broadcast_in_dim_p: (lambda key: key[None], key),
  olympus.lax.copy_p: (jnp.array, key),
  olympus.lax.convert_element_type_p: (lambda key: jnp.array(key, dtype=key.dtype), key),
  olympus.lax.reshape_p: (lambda key: key.reshape((1,)), key),
  olympus.lax.squeeze_p: (jnp.squeeze, key1D),
  olympus.lax.dynamic_slice_p: (partial(olympus.lax.dynamic_slice, slice_sizes=(1,)), key1D, (0,)),
  olympus.lax.dynamic_update_slice_p: (olympus.lax.dynamic_update_slice, key1D, key1D, (0,)),
}

# Primitive that is unknown to the key reuse machinery
unknown_p = core.Primitive("unknown")
unknown_p.def_abstract_eval(lambda x: x)
unknown_p.def_impl(lambda x: x)
def apply_unknown_primitive(key):
  return unknown_p.bind(key)


@jtu.with_config(
  olympus_enable_custom_prng=False,
  olympus_debug_key_reuse=False)
class KeyReuseUnitTestWithForwarding(jtu.OlympusTestCase):
  def check_key_reuse(self, *args):
    return _core.check_key_reuse(*args)

  def test_assertions(self):
    key = olympus.random.key(0)
    self.check_key_reuse(assert_unconsumed, key)
    with self.assertRaises(AssertionError):
      self.check_key_reuse(assert_consumed, key)

  def test_unknown(self):
    def f(key):
      assert_unconsumed(key)
      key2 = apply_unknown_primitive(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_consume(self):
    def f(key):
      assert_unconsumed(key)
      key2 = consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_random_clone(self):
    def f(key):
      assert_unconsumed(key)
      consume(key)
      assert_consumed(key)
      key2 = olympus.random.clone(key)
      assert_unconsumed(key2)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_seed(self):
    def f():
      key = olympus.random.key(0)
      assert_unconsumed(key)
    self.check_key_reuse(f)

  def test_split(self):
    def f(key):
      assert_unconsumed(key)
      key2 = olympus.random.split(key)
      assert_unconsumed(key2)
      assert_consumed(key)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_fold_in(self):
    def f(key):
      assert_unconsumed(key)
      key2 = olympus.random.fold_in(key, 2)
      assert_unconsumed(key)
      assert_unconsumed(key2)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_bits(self):
    def f(key):
      assert_unconsumed(key)
      bits = olympus.random.bits(key, (), 'uint32')
      assert_consumed(key)
      return bits
    self.check_key_reuse(f, olympus.random.key(0))

  def test_wrap(self):
    def f(key_data):
      key = olympus.random.wrap_key_data(key_data)
      assert_unconsumed(key)
    self.check_key_reuse(f, olympus.random.PRNGKey(0))

  def test_unwrap(self):
    def f(key):
      assert_unconsumed(key)
      key_data = olympus.random.key_data(key)
      assert_unconsumed(key)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_gamma(self):
    def f(key):
      assert_unconsumed(key)
      values = olympus.random.gamma(key, 1.0)
      assert_consumed(key)
      return values
    self.check_key_reuse(f, olympus.random.key(0))

  def test_broadcast_in_dim(self):
    def f(key):
      assert_unconsumed(key)
      key2 = key[None]
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_copy(self):
    def f(key):
      assert_unconsumed(key)
      key2 = jnp.array(key, copy=True)
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_device_put(self):
    def f(key):
      assert_unconsumed(key)
      key_d = olympus.device_put(key)
      assert_unconsumed(key_d)
      consume(key)
      assert_consumed(key_d)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_device_put_multiple(self):
    def f(key1, key2):
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      key1_d, key2_d = olympus.device_put((key1, key2))

      assert_unconsumed(key1_d)
      consume(key1)
      assert_consumed(key1_d)

      assert_unconsumed(key2_d)
      consume(key2)
      assert_consumed(key2_d)
    self.check_key_reuse(f, olympus.random.key(0), olympus.random.key(1))

  def test_squeeze(self):
    def f(key):
      assert_unconsumed(key)
      key2 = olympus.lax.squeeze(key, (0,))
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, olympus.random.key(0)[None])

  def test_reshape(self):
    def f(key):
      assert_unconsumed(key)
      key2 = key.reshape(1, *key.shape)
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_concatenate(self):
    def f(key1, key2):
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      keys = olympus.lax.concatenate([key1, key2], dimension=0)
      assert_consumed(key1)
      assert_consumed(key2)
      assert_unconsumed(keys)
    key1 = olympus.random.split(olympus.random.key(0))
    key2 = olympus.random.split(olympus.random.key(1))
    self.check_key_reuse(f, key1, key2)

  def test_slice(self):
    def f(keys):
      assert_unconsumed(keys)

      assert_unconsumed(keys[0])
      assert_consumed(keys, np.array([True, False]))

      assert_unconsumed(keys[1])
      assert_consumed(keys, np.array([True, True]))
    self.check_key_reuse(f, olympus.random.split(olympus.random.key(0)))

  @parameterized.parameters(operator.eq, operator.ne)
  def test_equality_checks(self, op):
    def f(key1, key2):
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      result = op(key1, key2)
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      return result
    self.check_key_reuse(f, olympus.random.key(0), olympus.random.key(1))

  def test_jit_can_consume_input(self):
    def f(key):
      assert_unconsumed(key)
      ans = olympus.jit(olympus.random.bits)(key)
      assert_consumed(key)
      return ans
    self.check_key_reuse(f, olympus.random.key(0))

  def test_jit_can_return_consumed_output(self):
    def f():
      def g():
        key = olympus.random.key(0)
        assert_unconsumed(key)
        bits = olympus.random.bits(key)
        assert_consumed(key)
        return bits, key
      _, key = olympus.jit(g)()
      assert_consumed(key)
    self.check_key_reuse(f)

  def test_jit_duplicate_inputs(self):
    def f(key):
      assert_unconsumed(key)
      def g(key1, key2):
        assert_unconsumed(key1)
        assert_unconsumed(key2)
        return olympus.random.bits(key1)
      _ = olympus.jit(g)(key, key)
      assert_consumed(key)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_jit_propagates_consumption_bit(self):
    def f(key):
      assert_unconsumed(key)
      g = olympus.jit(lambda: key)
      key2 = g()
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_jit_duplicate_outputs(self):
    # TODO(jakevdp): implement this case
    def f(key):
      assert_unconsumed(key)
      def g(key):
        return key, key
      key1, key2 = olympus.jit(g)(key)
      assert_unconsumed(key)
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      other = olympus.random.bits(key1)
      assert_consumed(key)
      assert_consumed(key1)
      assert_consumed(key2)
      return (key1, key2, other)
    self.check_key_reuse(f, olympus.random.key(0))

  def test_cond_both_consumed(self):
    @olympus.jit
    def f(flag, key):
      assert_unconsumed(key)
      ans = olympus.lax.cond(
        flag, olympus.random.uniform, olympus.random.normal, key)
      assert_consumed(key)
      return ans
    self.check_key_reuse(f, True, olympus.random.key(0))

  def test_cond_one_consumed(self):
    @olympus.jit
    def f(flag, key):
      assert_unconsumed(key)
      ans = olympus.lax.cond(
        flag, olympus.random.uniform, lambda k: 1.0, key)
      assert_consumed(key)
      return ans
    self.check_key_reuse(f, True, olympus.random.key(0))

  def test_cond_neither_consumed(self):
    @olympus.jit
    def f(flag, key):
      assert_unconsumed(key)
      _ = olympus.lax.cond(
        flag, lambda k: 0.0, lambda k: 1.0, key)
      assert_unconsumed(key)
    self.check_key_reuse(f, True, olympus.random.key(0))

  def test_simple_vmap(self):
    @olympus.jit
    def f(seed):
      key = olympus.random.key(seed)
      assert_unconsumed(key)
      result = olympus.random.uniform(key)
      assert_consumed(key)
      return result
    self.check_key_reuse(f, 0)
    self.check_key_reuse(olympus.vmap(f), jnp.arange(4))

  @parameterized.parameters(*primitives_with_static_signatures)
  def test_olympuspr_type_signature(self, primitive):
    func, *args = primitives_with_static_signatures[primitive]
    signature = _core.key_reuse_signatures[primitive]
    olympuspr = olympus.make_olympuspr(func)(*args)
    self.assertEqual(signature, _core.olympuspr_type_signature(olympuspr.olympuspr))

  @parameterized.parameters(*primitives_with_static_signatures)
  def test_function_type_signature(self, primitive):
    func, *args = primitives_with_static_signatures[primitive]
    signature = _core.key_reuse_signatures[primitive]
    self.assertEqual(signature, _core.function_type_signature(func, *args))


@jtu.with_config(olympus_debug_key_reuse=False)
class KeyReuseIntegrationTest(jtu.OlympusTestCase):
  random_bits_error = "In random_bits, argument [0-9]+ is already consumed.*"
  random_split_error = "In random_split, argument [0-9]+ is already consumed.*"
  generic_error = ".*argument [0-9]+ is already consumed.*"
  pjit_error = "In jit, argument 0 is already consumed."

  def check_key_reuse(self, f, *args):
    return _core.check_key_reuse(f, *args)

  def test_reuse(self):
    def f():
      key = olympus.random.key(0)
      return olympus.random.uniform(key) + olympus.random.uniform(key)

    with self.assertRaisesRegex(KeyReuseError, self.pjit_error):
      self.check_key_reuse(f)

  def test_reuse_after_split(self):
    def f_good():
      key = olympus.random.key(0)
      key1, key2 = olympus.random.split(key)
      return olympus.random.uniform(key1) + olympus.random.uniform(key2)
    self.check_key_reuse(f_good)

    def f_bad():
      key = olympus.random.key(0)
      other = olympus.random.split(key)
      return (olympus.random.uniform(key), other)

    with self.assertRaisesRegex(KeyReuseError, self.pjit_error):
      self.check_key_reuse(f_bad)

    def f_bad_2():
      key = olympus.random.key(0)
      other1 = olympus.random.split(key)
      key1, other2 = olympus.random.split(key)
      return (olympus.random.uniform(key1), other1, other2)

    with self.assertRaisesRegex(KeyReuseError, self.random_split_error):
      self.check_key_reuse(f_bad_2)

  def test_repeated_fold_ins(self):
    # TODO(jakevdp): should we allow repeated fold-ins?
    def f():
      key = olympus.random.key(0)
      keys = [olympus.random.fold_in(key, i)
              for i in range(10)]
      return [olympus.random.uniform(k) for k in keys]
    self.check_key_reuse(f)

  def test_reuse_after_fold_in(self):
    def f():
      key = olympus.random.key(0)
      _ = olympus.random.fold_in(key, 1)
      return olympus.random.uniform(key)

    self.check_key_reuse(f)

  def test_reuse_after_broadcast(self):
    def f():
      key = olympus.random.key(0)
      key2 = key[None]
      return olympus.random.bits(key) + olympus.vmap(olympus.random.bits)(key2)

    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(f)

  def test_reuse_after_reshape(self):
    def f():
      key = olympus.random.key(0)
      key2 = key.reshape((1,))
      return olympus.random.bits(key) + olympus.random.bits(key2.squeeze())

    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(f)

  def test_reuse_after_squeeze(self):
    def f():
      key = olympus.random.split(olympus.random.key(0), 1)
      key2 = olympus.lax.squeeze(key, (0,))
      return olympus.random.bits(key.squeeze()) + olympus.random.bits(key2)

    with self.assertRaisesRegex(KeyReuseError, self.generic_error):
      self.check_key_reuse(f)

  def test_reuse_after_cond(self):
    def f_good(key, condition):
      return olympus.lax.cond(condition, olympus.random.uniform, olympus.random.normal, key)
    key = olympus.random.key(0)
    self.check_key_reuse(f_good, key, True)
    self.check_key_reuse(f_good, key, False)

    # Check where both branches consume the key
    def f_bad(key, condition):
      r1 = olympus.lax.cond(condition, olympus.random.uniform, olympus.random.normal, key)
      return r1 + olympus.random.uniform(key)

    with self.assertRaisesRegex(KeyReuseError, self.pjit_error):
      self.check_key_reuse(f_bad, key, True)

    # Check where only one branch consumes the key
    def f_bad_2(key, condition):
      r1 = olympus.lax.cond(condition, olympus.random.uniform, lambda key: 1.0, key)
      return r1 + olympus.random.uniform(key)

    with self.assertRaisesRegex(KeyReuseError, self.pjit_error):
      self.check_key_reuse(f_bad_2, key, True)

  def test_simple_scan(self):
    def f_good(key):
      def body_fun(key, _):
        key, subkey = olympus.random.split(key)
        return key, olympus.random.bits(subkey)
      return olympus.lax.scan(body_fun, key, xs=jnp.arange(10))
    self.check_key_reuse(f_good, olympus.random.key(0))

  def test_scan_sink_on_consts(self):
    def f(key):
      def body_fun(carry, _):
        return carry, olympus.random.uniform(key)
      return olympus.lax.scan(body_fun, None, xs=jnp.arange(10))
    with self.assertRaisesRegex(KeyReuseError,  "scan body function leads to key reuse"):
      self.check_key_reuse(f, olympus.random.key(0))

  def test_scan_reuse_in_body(self):
    def f_bad(key):
      def body_fun(key, _):
        return key, olympus.random.bits(key)
      return olympus.lax.scan(body_fun, key, xs=jnp.arange(10))
    with self.assertRaisesRegex(KeyReuseError, "scan body function leads to key reuse"):
      self.check_key_reuse(f_bad, olympus.random.key(0))

  def test_scan_good_over_keys(self):
    def f_scan_over_keys(key):
      keys = olympus.random.split(key, 5)
      return olympus.lax.map(olympus.random.bits, keys)
    self.check_key_reuse(f_scan_over_keys, olympus.random.key(0))

  def test_scan_consume_one(self):
    def f_scan_over_keys(*keys):
      def body_func(keys, x):
        return tuple(olympus.random.split(keys[0])), x
      return olympus.lax.scan(body_func, keys, xs=jnp.arange(10))
    self.check_key_reuse(f_scan_over_keys, olympus.random.key(0), olympus.random.key(1))

  def test_vmap(self):
    @olympus.vmap
    def f_good(seed):
      key = olympus.random.key(seed)
      return olympus.random.bits(key)
    self.check_key_reuse(f_good, jnp.arange(4))

    @olympus.vmap
    def f_bad(seed):
      key = olympus.random.key(0)
      return olympus.random.bits(key) + olympus.random.bits(key)
    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(f_bad, jnp.arange(4))

  def test_while_simple(self):
    def f(seed):
      key = olympus.random.key(seed)
      def cond_fun(carry):
        return carry[1] < 10
      def body_fun(carry):
        key, subkey = olympus.random.split(carry[0])
        return key, carry[1] + olympus.random.uniform(subkey)
      return olympus.lax.while_loop(cond_fun, body_fun, (key, 0))
    self.check_key_reuse(f, 0)

  def test_while_bad_cond(self):
    def f(seed):
      key = olympus.random.key(seed)
      def cond_fun(carry):
        i, key = carry
        return i < olympus.random.uniform(key)
      def body_fun(carry):
        i, key = carry
        return i + 1, key
      return olympus.lax.while_loop(cond_fun, body_fun, (0, key))
    with self.assertRaisesRegex(KeyReuseError, "while_loop cond"):
      self.check_key_reuse(f, 0)

  def test_while_bad_body(self):
    def f(seed):
      key = olympus.random.key(seed)
      def cond_fun(carry):
        key, i = carry
        return i < 5
      def body_fun(carry):
        key, i = carry
        return key, i + olympus.random.randint(key, (), 1, 3)
      return olympus.lax.while_loop(cond_fun, body_fun, (key, 0))
    with self.assertRaisesRegex(KeyReuseError, "while_loop body function leads to key reuse"):
      self.check_key_reuse(f, 0)

  def test_while_sink_on_body_consts(self):
    def f(seed):
      key = olympus.random.key(seed)
      def cond_fun(i):
        return i < 5
      def body_fun(i):
        return i + olympus.random.randint(key, (), 1, 3)
      return olympus.lax.while_loop(cond_fun, body_fun, 0)
    with self.assertRaisesRegex(KeyReuseError, "while_loop body function leads to key reuse"):
      self.check_key_reuse(f, 0)

  def test_while_sink_on_cond_consts(self):
    def f(seed):
      key = olympus.random.key(seed)
      def cond_fun(i):
        return i < olympus.random.uniform(key)
      def body_fun(i):
        return i + 1
      return olympus.lax.while_loop(cond_fun, body_fun, 0)
    with self.assertRaisesRegex(KeyReuseError, "while_loop cond function leads to key reuse"):
      self.check_key_reuse(f, 0)

  def test_pjit_consumed_input(self):
    @olympus.jit
    def g(key, x):  # doesn't consume key
      return x

    def f(seed):
      key = olympus.random.key(seed)
      x = olympus.random.bits(key)
      return g(key, x)

    self.check_key_reuse(f, 0)

  @olympus.numpy_dtype_promotion('standard')
  def test_remat(self):
    @olympus.checkpoint
    def f_bad(x, key):
      return x * olympus.random.bits(key) + olympus.random.bits(key)

    @olympus.checkpoint
    def f_good(x, key):
      return x * olympus.random.bits(key)

    x = jnp.float32(1.0)
    key = olympus.random.key(0)

    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(f_bad, x, key)

    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(olympus.value_and_grad(f_bad), x, key)

    self.check_key_reuse(f_good, x, key)
    self.check_key_reuse(olympus.grad(f_good), x, key)


@jtu.with_config(olympus_debug_key_reuse=True)
class KeyReuseEagerTest(jtu.OlympusTestCase):
  jit_msg = "Previously-consumed key passed to jit-compiled function at index 0"
  eager_bits_msg = "Previously-consumed key passed to random_bits at index 0"
  traced_bits_msg = "In random_bits, argument 0 is already consumed."

  def test_clone_eager(self):
    key = olympus.random.key(0)
    key2 = olympus.random.clone(key)
    self.assertIsNot(key, key2)

    _ = olympus.random.uniform(key)
    self.assertTrue(key._consumed)
    self.assertFalse(key2._consumed)

  def test_simple_reuse_nojit(self):
    key = olympus.random.key(0)
    with olympus.disable_jit():
      _ = olympus.random.bits(key)
      with self.assertRaisesRegex(KeyReuseError, self.eager_bits_msg):
        _ = olympus.random.bits(key)

  def test_simple_key_reuse_jit(self):
    key = olympus.random.key(0)
    _ = olympus.jit(olympus.random.bits)(key)
    with self.assertRaisesRegex(KeyReuseError, self.jit_msg):
      _ = olympus.jit(olympus.random.bits)(key)

  def test_closed_over_key_reuse_jit(self):
    key = olympus.random.key(0)
    @olympus.jit
    def f():
      return olympus.random.uniform(key)
    _ = f()
    with self.assertRaisesRegex(KeyReuseError, self.jit_msg):
      _ = f()

  def test_key_reuse_within_jit(self):
    @olympus.jit
    def f():
      key = olympus.random.key(0)
      return olympus.random.bits(key) + olympus.random.bits(key)
    with self.assertRaisesRegex(KeyReuseError, self.traced_bits_msg):
      f()


class KeyReuseImplementationTest(jtu.OlympusTestCase):

  def assertEquivalent(self, a, b):
    self.assertEqual(a, b)
    self.assertEqual(hash(a), hash(b))

  def assertNotEquivalent(self, a, b):
    self.assertNotEqual(a, b)
    self.assertNotEqual(hash(a), hash(b))

  def test_source_sink_immutability(self):
    mask = np.array([True, False])
    orig_mask_writeable = mask.flags.writeable

    sink = Sink(0, mask)
    source = Source(0, mask)

    self.assertFalse(sink.mask.flags.writeable)
    self.assertFalse(source.mask.flags.writeable)
    self.assertEqual(mask.flags.writeable, orig_mask_writeable)

    with self.assertRaises(ValueError):
      sink.idx = 1
    with self.assertRaises(ValueError):
      sink.mask = True
    with self.assertRaises(ValueError):
      source.idx = 1
    with self.assertRaises(ValueError):
      source.mask = True

  def test_source_sink_forward_equivalence_semantics(self):

    true_mask = np.array([True, True])
    false_mask = np.array([False, False])
    mixed_mask = np.array([True, False])

    self.assertEquivalent(Source(0), Source(0, True))
    self.assertEquivalent(Source(0, True), Source(0, true_mask))
    self.assertEquivalent(Source(0, False), Source(0, false_mask))
    self.assertEquivalent(Source(0, mixed_mask), Source(0, mixed_mask))
    self.assertNotEquivalent(Source(0), Source(1))
    self.assertNotEquivalent(Source(0), Source(0, False))
    self.assertNotEquivalent(Source(0), Source(0, mixed_mask))

    self.assertEquivalent(Sink(0), Sink(0, True))
    self.assertEquivalent(Sink(0, True), Sink(0, true_mask))
    self.assertEquivalent(Sink(0, False), Sink(0, false_mask))
    self.assertEquivalent(Sink(0, mixed_mask), Sink(0, mixed_mask))
    self.assertNotEquivalent(Sink(0), Sink(1))
    self.assertNotEquivalent(Sink(0), Sink(0, False))
    self.assertNotEquivalent(Sink(0), Sink(0, mixed_mask))

    self.assertNotEquivalent(Source(0), Sink(0))

    self.assertEquivalent(Forward(0, 1), Forward(0, 1))
    self.assertNotEquivalent(Forward(0, 1), Forward(1, 0))

  def test_signature_equality_semantics(self):
    self.assertEquivalent(
      KeyReuseSignature(Sink(0), Source(1), Forward(1, 0)),
      KeyReuseSignature(Forward(1, 0), Source(1), Sink(0)))
    self.assertEquivalent(
      KeyReuseSignature(), KeyReuseSignature())
    self.assertNotEquivalent(
      KeyReuseSignature(Source(0)), KeyReuseSignature(Sink(0)))

  def test_reprs(self):
    self.assertEqual(repr(Sink(0)), "Sink(0)")
    self.assertEqual(repr(Source(0)), "Source(0)")
    self.assertEqual(repr(Forward(0, 1)), "Forward(0, 1)")
    self.assertEqual(repr(KeyReuseSignature(Sink(1), Source(0))),
                     "KeyReuseSignature(Sink(1), Source(0))")
    self.assertEqual(repr(KeyReuseSignature(Sink(1), Sink(0))),
                     "KeyReuseSignature(Sink(0), Sink(1))")



@jtu.with_config(olympus_enable_checks=False)
class KeyReuseGlobalFlagsTest(jtu.OlympusTestCase):
  def test_key_reuse_flag(self):

    @olympus.jit
    def f_bad(key):
      return olympus.random.bits(key) + olympus.random.bits(key)

    @olympus.jit
    def f_good(key):
      return olympus.random.bits(key)

    key = olympus.random.key(0)

    with olympus.debug_key_reuse(False):
      f_good(key)
      f_bad(key)  # No failure

    f_bad.clear_cache()
    f_good.clear_cache()

    with olympus.debug_key_reuse(True):
      f_good(key)
      with self.assertRaisesRegex(KeyReuseError, "In random_bits.*"):
        f_bad(key)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
