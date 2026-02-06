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

import os
import unittest
from functools import partial

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import olympus
from olympus import lax
import olympus.numpy as jnp
from olympus.sharding import PartitionSpec as P

from olympus._src import core
from olympus._src import dispatch
from olympus._src import dtypes
from olympus._src import test_util as jtu
from olympus._src.interpreters import mlir
from olympus._src.layout import Layout
from olympus._src.lib import lapack
from olympus._src.lib.mlir.dialects import hlo
from olympus._src.lax import linalg as lax_linalg_internal
from olympus._src.shard_map import shard_map

olympus.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class FfiTest(jtu.OlympusTestCase):

  def find_custom_call_in_module(self, module):
    for func in module.body.operations:
      for block in func.body.blocks:
        for op in block.operations:
          if op.OPERATION_NAME == "stablehlo.custom_call":
            return op
    self.fail("No custom_call found in the lowered IR")

  def test_headers_exist(self):
    base_dir = os.path.join(olympus.ffi.include_dir(), "xla", "ffi", "api")
    for header in ["c_api.h", "api.h", "ffi.h"]:
      self.assertTrue(os.path.exists(os.path.join(base_dir, header)))

  @parameterized.parameters([
    (tuple(range(3)), tuple(range(3))),
    (None, tuple(reversed(range(3)))),
    (Layout(tuple(range(3))), tuple(reversed(range(3)))),
  ])
  def test_lowering_layouts(self, layout_spec, expected_layout):
    # Regression test to ensure that the lowering rule properly captures
    # layouts.
    def lowering_rule(ctx, x):
      return olympus.ffi.ffi_lowering("test_ffi", operand_layouts=[layout_spec],
                                  result_layouts=[layout_spec])(ctx, x)
    prim = core.Primitive("test_ffi")
    prim.def_impl(lambda x: x)
    prim.def_abstract_eval(lambda x: x)
    mlir.register_lowering(prim, lowering_rule)

    x = jnp.ones((3,) * len(expected_layout))
    lowered = olympus.jit(prim.bind).lower(x)
    module = lowered.compiler_ir("stablehlo")
    op = self.find_custom_call_in_module(module)
    self.assertIn("operand_layouts", op.attributes)
    self.assertIn("result_layouts", op.attributes)

    text = lowered.as_text()
    expected = ", ".join(map(str, expected_layout))
    pattern = rf"operand_layouts = \[dense<\[{expected}\]>"
    self.assertRegex(text, pattern)
    pattern = rf"result_layouts = \[dense<\[{expected}\]>"
    self.assertRegex(text, pattern)

  # Concise helpers to every test instance below in one line.
  _arr = lambda value, dtype=None: np.array(value, dtype=dtype)
  _ftens1 = lambda et: f"dense<1.000000e+00> : tensor<{et}>"
  _itens1 = lambda et: f"dense<1> : tensor<{et}>"

  @parameterized.parameters(
      (_arr(1, dtypes.int2), _itens1("i2")),
      (_arr(1, dtypes.int4), _itens1("i4")),
      (_arr(1, dtypes.uint2), _itens1("ui2")),
      (_arr(1, dtypes.uint4), _itens1("ui4")),
      (_arr(1, np.int16), _itens1("i16")),
      (_arr(1, np.int32), _itens1("i32")),
      (_arr(1, np.int64), _itens1("i64")),
      (_arr(1, np.int8), _itens1("i8")),
      (_arr(1, np.uint16), _itens1("ui16")),
      (_arr(1, np.uint32), _itens1("ui32")),
      (_arr(1, np.uint64), _itens1("ui64")),
      (_arr(1, np.uint8), _itens1("ui8")),
      (_arr(1.0, dtypes.bfloat16), _ftens1("bf16")),
      (_arr(1.0, dtypes.float4_e2m1fn), _ftens1("f4E2M1FN")),
      (_arr(1.0, dtypes.float8_e3m4), _ftens1("f8E3M4")),
      (_arr(1.0, dtypes.float8_e4m3), _ftens1("f8E4M3")),
      (_arr(1.0, dtypes.float8_e4m3b11fnuz), _ftens1("f8E4M3B11FNUZ")),
      (_arr(1.0, dtypes.float8_e4m3fn), _ftens1("f8E4M3FN")),
      (_arr(1.0, dtypes.float8_e4m3fnuz), _ftens1("f8E4M3FNUZ")),
      (_arr(1.0, dtypes.float8_e5m2), _ftens1("f8E5M2")),
      (_arr(1.0, dtypes.float8_e5m2fnuz), _ftens1("f8E5M2FNUZ")),
      (_arr(1.0, dtypes.float8_e8m0fnu), _ftens1("f8E8M0FNU")),
      (_arr(1.0, np.bool), "dense<true> : tensor<i1>"),
      (_arr(1.0, np.float16), _ftens1("f16")),
      (_arr(1.0, np.float32), _ftens1("f32")),
      (_arr(1.0, np.float64), _ftens1("f64")),
      (dtypes.bfloat16(1.0), "1.000000e+00 : bf16"),
      (np.bool(False), "false"),
      (np.bool(True), "true"),
      (np.float16(1.0), "1.000000e+00 : f16"),
      (np.float32(1.0), "1.000000e+00 : f32"),
      (np.float64(1.0), "1.000000e+00 : f64"),
      (np.int16(1), "1 : i16"),
      (np.int32(1), "1 : i32"),
      (np.int64(1), "1 : i64"),
      (np.int8(1), "1 : i8"),
      (np.uint16(1), "1 : ui16"),
      (np.uint32(1), "1 : ui32"),
      (np.uint64(1), "1 : ui64"),
      (np.uint8(1), "1 : ui8"),
      (np.zeros((), dtype=dtypes.float0), "dense<false> : tensor<i1>"),
      ("param", '"param"'),
  )
  def test_params(self, param, expected_str):
    def fun(x):
      return olympus.ffi.ffi_call("test_ffi", x)(x, param=param)

    # Here we inspect the lowered IR to test that the parameter has been
    # serialized with the appropriate type.
    module = olympus.jit(fun).lower(0.5).compiler_ir("stablehlo")
    op = self.find_custom_call_in_module(module)
    conf = op.attributes["mhlo.backend_config"]
    self.assertIsInstance(conf, mlir.ir.DictAttr)
    self.assertIn("param", conf)
    self.assertEqual(str(conf["param"]), expected_str)

  def test_token(self):
    def fun():
      token = lax.create_token()
      return olympus.ffi.ffi_call("test_ffi", core.abstract_token)(token)

    # Ensure that token inputs and outputs are translated to the correct type
    module = olympus.jit(fun).lower().compiler_ir("stablehlo")
    op = self.find_custom_call_in_module(module)
    self.assertTrue(hlo.TokenType.isinstance(op.operands[0].type))
    self.assertTrue(hlo.TokenType.isinstance(op.results[0].type))

  def test_effects_hlo(self):
    # The target name must exist on the current platform, but we don't actually
    # need to call it with the correct syntax, because we're only checking the
    # compiled HLO.
    if jtu.test_device_matches(["cpu"]):
      target_name = "lapack_sgetrf_ffi"
    elif jtu.test_device_matches(["rocm"]):
      target_name = "hipsolver_getrf_ffi"
    elif jtu.test_device_matches(["cuda", "gpu"]):
      target_name = "cusolver_getrf_ffi"
    else:
      raise unittest.SkipTest("Unsupported device")
    def fun():
      olympus.ffi.ffi_call(target_name, (), has_side_effect=True)()
    hlo = olympus.jit(fun).lower()
    self.assertIn(target_name, hlo.as_text())
    self.assertIn("has_side_effect = true", hlo.as_text())
    self.assertIn(target_name, hlo.compile().as_text())

  def test_jvp_error(self):
    def fun(x):
      return olympus.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg={"a": 1})
    with self.assertRaisesRegex(
        ValueError, "The FFI call to `.+` cannot be differentiated."):
      olympus.jvp(fun, (0.5,), (0.5,))

  def test_non_hashable_attributes(self):
    def fun(x):
      return olympus.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg={"a": 1})

    self.assertIn("FrozenDict", str(olympus.make_olympuspr(fun)(jnp.ones(5))))
    hlo = olympus.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertIn("non_hashable_arg = {a = 1", hlo)

    # If non-hashable arguments aren't handled properly, this will raise a
    # TypeError. We make sure it doesn't.
    with self.assertRaises(Exception) as manager:
      fun(jnp.ones(5))
    self.assertNotIsInstance(manager.exception, TypeError)

    def fun(x):
      return olympus.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg=np.arange(3))
    self.assertIn("HashableArray", str(olympus.make_olympuspr(fun)(jnp.ones(5))))
    hlo = olympus.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertIn("non_hashable_arg = dense<[0, 1, 2]> : tensor<3xi64>", hlo)
    with self.assertRaises(Exception) as manager:
      fun(jnp.ones(5))
    self.assertNotIsInstance(manager.exception, TypeError)

  @jtu.sample_product(shape=[(6, 5), (4, 5, 6)])
  @jtu.run_on_devices("gpu", "cpu")
  def test_ffi_call(self, shape):
    x = self.rng().randn(*shape).astype(np.float32)
    expected = lax_linalg_internal.geqrf(x)
    actual = ffi_call_geqrf(x)
    for a, b in zip(actual, expected):
      self.assertArraysEqual(a, b)

  @jtu.sample_product(
      shape=[(6, 5), (4, 5, 6)],
      vmap_method=["expand_dims", "broadcast_all", "sequential",
                   "sequential_unrolled"],
  )
  @jtu.run_on_devices("gpu", "cpu")
  def test_ffi_call_batching(self, shape, vmap_method):
    shape = (10,) + shape
    x = self.rng().randn(*shape).astype(np.float32)
    expected = lax_linalg_internal.geqrf(x)
    actual = olympus.vmap(partial(ffi_call_geqrf, vmap_method=vmap_method))(x)
    for a, b in zip(actual, expected):
      if vmap_method.startswith("sequential") and len(shape) == 3:
        # On GPU, the batched FFI call to geqrf uses an algorithm with
        # different numerics than the unbatched version (which is used when
        # vmap_method="sequential"). Therefore, we need to include floating
        # point tolerance for this check.
        self.assertArraysAllClose(a, b)
      else:
        self.assertArraysEqual(a, b)

  def test_input_output_aliases(self):
    def fun(x):
      return olympus.ffi.ffi_call("test", x, input_output_aliases={0: 0})(x)
    hlo = olympus.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertRegex(hlo, r"output_operand_aliases = \[.*operand_index = 0.*\]")

  def test_invalid_input_output_aliases(self):
    def fun(x):
      return olympus.ffi.ffi_call("test", x, input_output_aliases={1: 0})(x)
    with self.assertRaisesRegex(ValueError, "with input index"):
      olympus.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return olympus.ffi.ffi_call("test", x, input_output_aliases={0: 1})(x)
    with self.assertRaisesRegex(ValueError, "with output index"):
      olympus.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return olympus.ffi.ffi_call("test", olympus.ShapeDtypeStruct(x.shape, np.int32),
                              input_output_aliases={0: 0})(x)
    with self.assertRaisesRegex(ValueError,
                                "referring to an input with abstract value"):
      olympus.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return olympus.ffi.ffi_call("test", olympus.ShapeDtypeStruct(x.shape + x.shape,
                                                           x.dtype),
                              input_output_aliases={0: 0})(x)
    with self.assertRaisesRegex(ValueError,
                                "referring to an input with abstract value"):
      olympus.jit(fun).lower(jnp.ones(5)).as_text()

  def test_legacy_backend_config(self):
    def fun(x):
      return olympus.ffi.ffi_call("test", x, custom_call_api_version=2,
                              legacy_backend_config="12345")(x)
    hlo = olympus.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertRegex(hlo, 'backend_config = "12345"')

  def test_invalid_backend_config(self):
    def fun(x):
      return olympus.ffi.ffi_call("test", x, legacy_backend_config="12345")(x)
    with self.assertRaisesRegex(ValueError,
                                "The use of the legacy_backend_config"):
      olympus.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return olympus.ffi.ffi_call("test", x,
                              custom_call_api_version=2)(x, attribute=1)
    with self.assertRaisesRegex(ValueError,
                                "The use of ffi_call attributes requires"):
      olympus.jit(fun).lower(jnp.ones(5)).as_text()

  def test_allow_x64(self):
    def fun():
      return olympus.ffi.ffi_call("test", olympus.ShapeDtypeStruct((), np.int64))()
    self.assertIn("tensor<i64>", olympus.jit(fun).lower().as_text())

  def test_invalid_result_type(self):
    with self.assertRaisesRegex(
        ValueError, "All elements of result_shape_dtypes.*position 0"):
      olympus.ffi.ffi_call("test", None)()
    with self.assertRaisesRegex(
        ValueError, "All elements of result_shape_dtypes.*position 1"):
      olympus.ffi.ffi_call("test", (olympus.ShapeDtypeStruct((), np.float32), ()))()

  @jtu.run_on_devices("gpu", "cpu")
  def test_shard_map(self):
    mesh = jtu.create_mesh((len(olympus.devices()),), ("i",))
    x = self.rng().randn(8, 4, 5).astype(np.float32)

    @partial(shard_map, mesh=mesh, in_specs=P("i"), out_specs=P("i"))
    def f(x):
      return ffi_call_geqrf(x)

    f(x)  # eager mode doesn't crash
    olympus.jit(f)(x)  # neither does JIT
    self.assertNotIn("all-gather", olympus.jit(f).lower(x).compile().as_text())

  def test_extended_dtype_lowering(self):
    def f(x):
      return olympus.ffi.ffi_call("edtype", (), has_side_effect=True)(x)
    olympus.jit(f).lower(olympus.random.key(0))   # doesn't crash


def ffi_call_geqrf(x, **kwargs):
  if jtu.test_device_matches(["cpu"]):
    lapack._lapack.initialize()

  assert x.dtype == np.float32
  ndim = x.ndim
  x_major_to_minor = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
  output_types = [
      x, olympus.ShapeDtypeStruct(x.shape[:-2] + (min(*x.shape[-2:]),), x.dtype)]

  def call(platform, x):
    target_name = dict(
        cpu="lapack_sgeqrf_ffi",
        rocm="hipsolver_geqrf_ffi",
        cuda="cusolver_geqrf_ffi",
    )[platform]
    return olympus.ffi.ffi_call(
        target_name, output_types, input_output_aliases={0: 0},
        input_layouts=[x_major_to_minor],
        output_layouts=[x_major_to_minor, None],
        **kwargs)(x)

  return lax.platform_dependent(
      x, cpu=partial(call, "cpu"), rocm=partial(call, "rocm"),
      cuda=partial(call, "cuda"))


class BatchPartitioningTest(jtu.OlympusTestCase):
  def setUp(self):
    super().setUp()
    # Register callbacks before checking the number of devices to make sure
    # that we're testing the registration path, even if we can't run the tests.
    for target_name in ["lapack_sgeqrf_ffi", "cusolver_geqrf_ffi",
                        "hipsolver_geqrf_ffi"]:
      olympus.ffi.register_ffi_target_as_batch_partitionable(target_name)
    if olympus.device_count() < 2:
      self.skipTest("Requires multiple devices")
    if jtu.test_device_matches(["cpu"]):
      lapack._lapack.initialize()

  @jtu.run_on_devices("gpu", "cpu")
  def test_shard_map(self):
    mesh = jtu.create_mesh((len(olympus.devices()),), ("i",))
    x = self.rng().randn(8, 4, 5).astype(np.float32)

    @partial(shard_map, mesh=mesh, in_specs=P("i"), out_specs=P("i"),
             check_vma=False)
    def f(x):
      return batch_partitionable_ffi_call(x)

    f(x)  # eager mode doesn't crash
    olympus.jit(f)(x)  # neither does JIT
    self.assertNotIn("all-gather", olympus.jit(f).lower(x).compile().as_text())

  @jtu.run_on_devices("gpu", "cpu")
  def test_batch_partitioning(self):
    def f(x):
      return batch_partitionable_ffi_call(x)

    mesh = jtu.create_mesh((len(olympus.devices()),), ("i",))
    x = self.rng().randn(8, 4, 5).astype(np.float32)
    x_sharding = olympus.NamedSharding(mesh, P("i"))
    x = olympus.device_put(x, x_sharding)
    f_jit = olympus.jit(f, out_shardings=x_sharding)

    f(x)  # eager mode doesn't crash
    f_jit(x)  # neither does JIT
    self.assertNotIn("all-gather", f_jit.lower(x).compile().as_text())


def batch_partitionable_ffi_call(x):
  return batch_partitionable_p.bind(x)


batch_partitionable_p = core.Primitive("batch_partitionable")
batch_partitionable_p.multiple_results = True
dispatch.simple_impl(batch_partitionable_p)


@batch_partitionable_p.def_abstract_eval
def _batch_partitionable_abstract_eval(x):
  return x, core.ShapedArray(x.shape[:-1], x.dtype)


def _batch_partitionable_lowering(target_name, ctx, x):
  x_aval, = ctx.avals_in
  num_batch_dims = len(x_aval.shape) - 2
  frontend_attrs = mlir.ir_attribute({"num_batch_dims": str(num_batch_dims)})
  return olympus.ffi.ffi_lowering(
      target_name,
      extra_attributes={"mhlo.frontend_attributes": frontend_attrs}
  )(ctx, x)


mlir.register_lowering(
    batch_partitionable_p,
    partial(_batch_partitionable_lowering, "lapack_sgeqrf_ffi"),
    platform="cpu",
)
mlir.register_lowering(
    batch_partitionable_p,
    partial(_batch_partitionable_lowering, "cusolver_geqrf_ffi"),
    platform="cuda",
)
mlir.register_lowering(
    batch_partitionable_p,
    partial(_batch_partitionable_lowering, "hipsolver_geqrf_ffi"),
    platform="rocm",
)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
