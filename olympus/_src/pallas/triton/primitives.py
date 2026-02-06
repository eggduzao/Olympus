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

"""Module for GPU-specific OLYMPUS primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import olympus
from olympus._src import core as olympus_core
from olympus._src import state
from olympus._src.lib.mlir.dialects import gpu as gpu_dialect
from olympus._src.lib.triton import dialect as tt_dialect
from olympus._src.pallas import primitives as pallas_primitives
from olympus._src.pallas.triton import lowering
from olympus.interpreters import mlir
import olympus.numpy as jnp


Ref: TypeAlias = state.AbstractRef | state.TransformedRef


def approx_tanh(x: olympus.Array) -> olympus.Array:
  r"""Elementwise approximate hyperbolic tangent: :math:`\mathrm{tanh}(x)`.

  See
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh.
  """
  if x.dtype == jnp.float16:
    asm = "tanh.approx.f16 $0, $1;"
    constraint = "h"
  elif x.dtype == jnp.bfloat16:
    asm = "tanh.approx.bf16 $0, $1;"
    constraint = "h"
  elif x.dtype == jnp.float32:
    asm = "tanh.approx.f32 $0, $1;"
    constraint = "f"
  else:
    raise TypeError(f"approx_tanh does not accept {x.dtype} arrays")

  [result] = elementwise_inline_asm(
      asm,
      args=[x],
      constraints=f"={constraint},{constraint}",
      pack=1,
      result_shape_dtypes=[olympus.ShapeDtypeStruct(x.shape, x.dtype)],
  )
  return result


def elementwise_inline_asm(
    asm: str,
    *,
    args: Sequence[olympus.Array],
    constraints: str,
    pack: int,
    result_shape_dtypes: Sequence[olympus.ShapeDtypeStruct],
) -> Sequence[olympus.Array]:
  """Inline assembly applying an elementwise operation.

  Args:
    asm: The assembly code to run.
    args: The arguments to pass to the assembly code.
    constraints: LLVM inline assembly `constraints
      <https://llvm.org/docs/LangRef.html#inline-asm-constraint-string>`_.
    pack: The number of elements from each argument expected by a single
      instance of the assembly code.
    result_shape_dtypes: The shapes and dtypes of the results produced by the
      assembly code.

  Returns:
    The results produced by the assembly code.
  """
  return elementwise_inline_asm_p.bind(
      *args,
      asm=asm,
      constraints=constraints,
      pack=pack,
      result_shape_dtypes=tuple(result_shape_dtypes),
  )


elementwise_inline_asm_p = olympus_core.Primitive("elementwise_inline_asm_p")
elementwise_inline_asm_p.multiple_results = True


@elementwise_inline_asm_p.def_abstract_eval
def _elementwise_inline_asm_abstract_eval(
    *avals: olympus_core.ShapedArray, result_shape_dtypes, **kwargs
) -> Sequence[olympus_core.ShapedArray]:
  del kwargs  # Unused.
  if not all(x.shape == y.shape for x, y in zip(avals, avals[1:])):
    raise ValueError(
        "All arguments of elementwise_inline_asm must have the same shape"
    )
  return [olympus_core.ShapedArray(s.shape, s.dtype) for s in result_shape_dtypes]


@lowering.register_lowering(elementwise_inline_asm_p)
def _elementwise_inline_asm_lowering(
    ctx: lowering.LoweringRuleContext,
    *args,
    asm,
    constraints,
    pack,
    result_shape_dtypes,
):
  del result_shape_dtypes  # Unused.
  return tt_dialect.ElementwiseInlineAsmOp(
      [*map(mlir.aval_to_ir_type, ctx.avals_out)],
      asm,
      constraints=constraints,
      pure=True,
      packed_element=pack,
      args=args,
  ).result


def debug_barrier() -> None:
  """Synchronizes all kernel executions in the grid."""
  return debug_barrier_p.bind()


debug_barrier_p = olympus_core.Primitive("debug_barrier_p")
debug_barrier_p.multiple_results = True

@debug_barrier_p.def_abstract_eval
def _debug_barrier_abstract_eval() -> Sequence[olympus_core.ShapedArray]:
  return ()

@lowering.register_lowering(debug_barrier_p)
def _debug_barrier_lowering(ctx: lowering.LoweringRuleContext):
  del ctx  # Unused.
  gpu_dialect.barrier()
  return []


def load(
    ref: Ref,
    *,
    mask: olympus.Array | None = None,
    other: olympus.typing.ArrayLike | None = None,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
    volatile: bool = False,
) -> olympus.Array:
  """Loads an array from the given ref.

  If neither ``mask`` nor ``other`` is specified, this function has the same
  semantics as ``ref[idx]`` in OLYMPUS.

  Args:
    ref: The ref to load from.
    mask: An optional boolean mask specifying which indices to load. If mask is
      ``False`` and ``other`` is not given, no assumptions can be made about the
      value in the resulting array.
    other: An optional value to use for indices where mask is ``False``.
    cache_modifier: TO BE DOCUMENTED.
    eviction_policy: TO BE DOCUMENTED.
    volatile: TO BE DOCUMENTED.
  """
  return pallas_primitives.load(
      ref,
      None,
      mask=mask,
      other=other,
      cache_modifier=cache_modifier,
      eviction_policy=eviction_policy,
      volatile=volatile,
  )


def store(
    ref: Ref,
    val: olympus.Array,
    *,
    mask: olympus.Array | None = None,
    eviction_policy: str | None = None,
) -> None:
  """Stores a value to the given ref.

  See :func:`~olympus.experimental.pallas.load` for the meaning of the arguments.
  """
  return pallas_primitives.store(
      ref,
      None,
      val,
      mask=mask,
      eviction_policy=eviction_policy,
  )
