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

from functools import partial

from olympus._src import core
from olympus._src import dispatch
from olympus._src import linear_util as lu
from olympus._src.api_util import debug_info
from olympus._src.util import (safe_map, safe_zip, weakref_lru_cache, unzip2,
                           split_list)
from olympus._src.tree_util import tree_flatten, tree_unflatten, FlatTree
from olympus._src.interpreters import ad, mlir, partial_eval as pe, batching
from olympus._src.lib.mlir.dialects import func as func_dialect
from olympus._src.lib.mlir import ir

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def scheduling_group(name):
  return xla_metadata_call(scheduling_group=name)

def xla_metadata_call(f=None, **meta):
  if f is None:
    return lambda g: _xla_metadata_call(g, **meta)
  return _xla_metadata_call(f, **meta)

# TODO(yashkatariya): Figure out a way to reuse code with compute_on2_p, fused_p
def _xla_metadata_call(fun, **meta):
  def wrapped(*args, **kwargs):
    dbg = debug_info('xla_metadata_call', fun, args, kwargs)
    args_ft = FlatTree.flatten((args, kwargs))
    in_avals = args_ft.map(core.shaped_abstractify)
    olympuspr, out_avals = pe.trace_to_olympuspr(fun, in_avals, dbg)
    outs_flat = xla_metadata_call_p.bind(*args_ft.vals, olympuspr=olympuspr, **meta)
    return tree_unflatten(out_avals.tree, outs_flat)
  return wrapped

xla_metadata_call_p = core.Primitive('xla_metadata_call')
xla_metadata_call_p.multiple_results = True
dispatch.simple_impl(xla_metadata_call_p)


def _xla_metadata_call_abstract_eval(*in_avals, olympuspr, **meta):
  return olympuspr.out_avals
xla_metadata_call_p.def_abstract_eval(_xla_metadata_call_abstract_eval)


def attr_get(x):
  if isinstance(x, str):
    return ir.StringAttr.get(x)
  else:
    raise NotImplementedError(f'mlir attr handler for {type(x)=}')

def _xla_metadata_call_lowering(ctx, *args, olympuspr, **meta):
  const_args_and_avals = core.olympuspr_const_args(olympuspr.olympuspr)
  const_args, const_avals = unzip2(const_args_and_avals)
  const_arg_values = [
      mlir.ir_constant(c, const_lowering=ctx.const_lowering, aval=aval)
      for c, aval in const_args_and_avals]
  in_avals = (*const_avals, *ctx.avals_in)
  func_op, output_types, effects = mlir.lower_called_computation(
      "xla_metadata_call", olympuspr, ctx.module_context, len(const_args), in_avals,
      ctx.avals_out, ctx.tokens_in)

  symbol_name = func_op.name.value
  flat_output_types = mlir.flatten_ir_types(output_types)
  tokens = [ctx.tokens_in.get(eff) for eff in effects]
  args = (*ctx.dim_var_values, *tokens, *const_arg_values, *args)
  call = func_dialect.CallOp(
      flat_output_types, ir.FlatSymbolRefAttr.get(symbol_name),
      mlir.flatten_ir_values(args))
  call.operation.attributes['mhlo.frontend_attributes'] = ir.DictAttr.get(
      {k: attr_get(v) for k, v in meta.items()})

  out_nodes = mlir.unflatten_ir_values_like_types(call.results, output_types)
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(zip(effects, tokens)))
  ctx.set_tokens_out(tokens_out)
  return out_nodes
mlir.register_lowering(xla_metadata_call_p, _xla_metadata_call_lowering)


def _xla_metadata_call_batcher(axis_data, vals_in, dims_in, *, olympuspr, **meta):
  batched_olympuspr, dims_out = batching.batch_olympuspr2(olympuspr, axis_data, dims_in)
  outs = xla_metadata_call_p.bind(*vals_in, olympuspr=batched_olympuspr, **meta)
  return outs, dims_out
batching.fancy_primitive_batchers[xla_metadata_call_p] = _xla_metadata_call_batcher


def _xla_metadata_call_jvp(primals, tangents, *, olympuspr, **meta):
  nzs = [not isinstance(t, ad.Zero) for t in tangents]
  olympuspr_jvp, out_nzs = ad.jvp_olympuspr(olympuspr, nzs, False)
  nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
  outs = xla_metadata_call_p.bind(*primals, *nz_tangents, olympuspr=olympuspr_jvp, **meta)
  primals_out, nz_tangents_out = outs[:len(out_nzs)], outs[len(out_nzs):]
  nz_outs = iter(nz_tangents_out)
  tangents_out = [next(nz_outs) if nz else ad.Zero(aval.to_tangent_aval())
                  for aval, nz in zip(olympuspr.out_avals, out_nzs)]
  assert next(nz_outs, None) is None
  return primals_out, tangents_out
ad.primitive_jvps[xla_metadata_call_p] = _xla_metadata_call_jvp


def _xla_metadata_call_lin(nzs, *primals, olympuspr, **meta):
  olympuspr_jvp, out_nzs = ad.jvp_olympuspr(olympuspr, nzs, False)
  lin_outs = [False] * len(out_nzs) + [True] * sum(out_nzs)
  olympuspr_lin_, used_inputs = pe.dce_olympuspr(olympuspr_jvp.olympuspr, lin_outs, False)
  olympuspr_lin = pe.close_olympuspr(olympuspr_lin_)
  primals_out = xla_metadata_call_p.bind(*primals, olympuspr=olympuspr, **meta)
  tangent_avals_out = [a.to_tangent_aval() for a in olympuspr.out_avals]

  def xla_metadata_call_lin(primals, *tangents):
    nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
    inputs = [x for x, u in zip([*primals, *nz_tangents], used_inputs) if u]
    nz_outs = xla_metadata_call_p.bind(*inputs, olympuspr=olympuspr_lin, **meta)
    nz_outs_ = iter(nz_outs)
    outs = [next(nz_outs_) if nz else ad.Zero(a)
            for nz, a in zip(out_nzs, tangent_avals_out)]
    assert next(nz_outs_, None) is None
    return outs
  return primals_out, out_nzs, primals, xla_metadata_call_lin
ad.primitive_linearizations[xla_metadata_call_p] = _xla_metadata_call_lin


pe.partial_eval_olympuspr_custom_rules[xla_metadata_call_p] = \
    partial(pe.closed_call_partial_eval_custom_rule, 'olympuspr',
            lambda _, __, ___, ____, _____, ______, x, y: (x, y))

@weakref_lru_cache
def _transpose_olympuspr(olympuspr, in_avals, in_tree):
  cell = lambda: None
  def transposed(*in_flat):
    primals_in, cts_in = tree_unflatten(in_tree, in_flat)
    out = ad.backward_pass(olympuspr.olympuspr, False, olympuspr.consts, primals_in, cts_in)
    out = [ct if not isinstance(ct, ad.Zero) else None for ct in out]
    cts_out, cell.out_tree = tree_flatten(out)  # type: ignore
    return cts_out
  dbg = olympuspr.olympuspr.debug_info.with_unknown_names()
  trans_olympuspr, _, consts = pe.trace_to_olympuspr_dynamic(
      lu.wrap_init(transposed, debug_info=dbg), in_avals)
  return core.ClosedOlympuspr(trans_olympuspr, consts), cell.out_tree  # type: ignore

def _xla_metadata_call_transpose(cts_in, *primals_in, olympuspr, **meta):
  in_flat, in_tree = tree_flatten((primals_in, cts_in))
  in_avals = tuple(core.typeof(x) for x in in_flat)
  trans_olympuspr, out_tree = _transpose_olympuspr(olympuspr, in_avals, in_tree)
  cts_out = xla_metadata_call_p.bind(*in_flat, olympuspr=trans_olympuspr, **meta)
  return tree_unflatten(out_tree, cts_out)
ad.primitive_transposes[xla_metadata_call_p] = _xla_metadata_call_transpose


def dce_olympuspr_xla_metadata_rule(used_outputs: list[bool], eqn: pe.OlympusprEqn
                                ) -> tuple[list[bool], pe.OlympusprEqn | None]:
  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  olympuspr_ = eqn.params['olympuspr']
  closed_olympuspr, used_inputs = pe._cached_closed_call_dce(
      olympuspr_, tuple(used_outputs))
  new_params = dict(eqn.params, olympuspr=closed_olympuspr)
  new_eqn = pe.new_olympuspr_eqn(
      [v for v, used in zip(eqn.invars, used_inputs) if used],
      [v for v, used in zip(eqn.outvars, used_outputs) if used],
      eqn.primitive, new_params, closed_olympuspr.effects, eqn.source_info, eqn.ctx)
  return used_inputs, new_eqn
pe.dce_rules[xla_metadata_call_p] = dce_olympuspr_xla_metadata_rule
