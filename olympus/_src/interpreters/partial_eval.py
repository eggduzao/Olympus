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

# pytype: skip-file
from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable, Sequence
import contextlib
from dataclasses import dataclass
from functools import partial
import logging
import itertools as it
import operator as op
from typing import Any, NamedTuple, Union
from weakref import finalize, ref, ReferenceType, WeakValueDictionary

import numpy as np

from olympus._src import ad_util
from olympus._src import api_util
from olympus._src import config
from olympus._src import core
from olympus._src import dtypes
from olympus._src import effects
from olympus._src import linear_util as lu
from olympus._src import profiler
from olympus._src import source_info_util
from olympus._src import xla_metadata_lib
from olympus._src import tree_util
from olympus._src.core import (
    Trace, Tracer, TraceTag, Olympuspr, Literal, get_aval, AbstractValue,
    ClosedOlympuspr, new_olympuspr_eqn, Var, DropVar, Atom, OlympusprEqn, Primitive,
    mapped_aval, unmapped_aval, get_referent, OlympusprEqnContext, typeof)
from olympus._src.source_info_util import SourceInfo
from olympus._src.state.types import AbstractRef, ReadEffect
from olympus._src.tree_util import PyTreeDef, treedef_tuple, FlatTree
from olympus._src.util import (unzip2, safe_zip, safe_map, toposort, split_list,
                           merge_lists, partition_list, OrderedSet,
                           as_hashable_function, weakref_lru_cache,
                           multi_weakref_lru_cache, subs_list,
                           HashableFunction, foreach, test_event)


map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
def identity(x): return x

TracerId = int
AvalId = int
ConstId = int

AttrKind = Any
PyTree = Any
logger = logging.getLogger(__name__)

class PartialVal(tuple):
  """Partial value: either a known value or an unknown (abstract) value.

  Represented as a pair `(aval_opt, const)` of one of two kinds:
  * `(None, <Constant>)` indicates a known value, where the constant is either a
    Tracer or satisfies `core.valid_olympustype(const)`;
  * `(<AbstractValue>, None)` indicates an unknown value characterized by an
    abstract value.
  """
  def __new__(cls, xs: tuple[AbstractValue | None, core.Value]):
    pv, const = xs
    if config.enable_checks.value:
      # type checks
      assert isinstance(pv, (AbstractValue, type(None))), xs
      assert (const is None or isinstance(const, core.Tracer) or
              core.valid_olympustype(const)), const
      # invariant checks
      assert (pv is None) ^ (const is None)
    return tuple.__new__(cls, xs)

  @classmethod
  def known(cls, const: core.Value) -> PartialVal:
    return PartialVal((None, const))

  @classmethod
  def unknown(cls, aval: AbstractValue) -> PartialVal:
    return PartialVal((aval, None))

  def is_known(self) -> bool:
    return self[0] is None

  def get_known(self) -> core.Value | None:
    """Get the known value, if known, else None."""
    return self[1] if self[0] is None else None

  def get_aval(self) -> AbstractValue:
    """Get AbstractValue directly (if unknown) or from the constant (known)."""
    known = self.get_known()
    if known is not None:
      return get_aval(known)
    else:
      return self[0]

@dataclass(frozen=True)
class EffectHandle:
  parents : list[Tracer]
  recipe : OlympusprEqnRecipe

class OlympusprTrace(Trace['OlympusprTracer']):

  def __init__(self, parent_trace:Trace, name_stack: source_info_util.NameStack, tag:TraceTag):
    super().__init__()
    self.name_stack = name_stack
    self.tag = tag
    self.parent_trace = parent_trace
    self.requires_low = False
    self.effect_handles : list[EffectHandle] = []
    self.counter = it.count()

  def to_olympuspr_tracer(self, x):
    if isinstance(x, OlympusprTracer) and x._trace.tag is self.tag:
      if x._trace is self:
        return x
      else:
        return OlympusprTracer(self, x.pval, FreeVar(x))
    else:
      return self.new_const(x)

  def new_const(self, val) -> OlympusprTracer:
    return OlympusprTracer(self, PartialVal.known(val), None)

  def new_instantiated_literal(self, val) -> OlympusprTracer:
    aval = get_aval(val)
    return OlympusprTracer(self, PartialVal.unknown(aval), Literal(val, aval))

  def new_instantiated_const(self, val) -> OlympusprTracer:
    aval = get_aval(val)
    return OlympusprTracer(self, PartialVal.unknown(aval), ConstVar(val))

  def new_arg(self, pval: PartialVal) -> OlympusprTracer:
    const = pval.get_known()
    # XXX: Think twice before changing this constant argument pruning!
    # This has really important consequences for partial_eval_olympuspr.
    # Most importantly, this guarantees that the unknown olympuspr never uses
    # known inputs (if it needs them, then they get passed through residuals).
    if const is None:
      aval = pval.get_aval()
      return OlympusprTracer(self, PartialVal.unknown(aval), LambdaBinding())
    else:
      return self.new_const(const)

  def instantiate_const(self, tracer: OlympusprTracer) -> OlympusprTracer:
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      if core.is_literalable(const):
        return self.new_instantiated_literal(const)
      else:
        return self.new_instantiated_const(const)

  def cur_qdd(self, x):
    const = self.to_olympuspr_tracer(x).pval.get_known()
    if const is None:
      assert False # TODO: track tangent QDDs
    else:
      with core.set_current_trace(self.parent_trace):
        return core.cur_qdd(const)

  def process_primitive(self, primitive, tracers, params):
    with core.set_current_trace(self.parent_trace):
      if primitive in custom_partial_eval_rules:
        tracers = map(self.to_olympuspr_tracer, tracers)
        return custom_partial_eval_rules[primitive](self, *tracers, **params)
      else:
        return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    # By default, if all the input tracers are known, then bind the primitive
    # and consider all outputs known. Otherwise, stage the application into the
    # olympuspr and consider all outputs unknown.
    tracers = map(self.to_olympuspr_tracer, tracers)
    consts = [t.pval.get_known() for t in tracers]
    if all(c is not None for c in consts):
      return primitive.bind_with_trace(self.parent_trace, consts, params)
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval, effs = primitive.abstract_eval(*avals, **params)
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    if primitive.multiple_results:
      out_tracers = [OlympusprTracer(self, PartialVal.unknown(aval), None)
                     for aval in out_aval]
      eqn = new_eqn_recipe(self, tracers, out_tracers, primitive, params, effs,
                           source)
      if effects.partial_eval_kept_effects.filter_in(effs):
        self.effect_handles.append(EffectHandle(tracers, eqn))
      for t in out_tracers: t.recipe = eqn
      return out_tracers
    else:
      out_tracer = OlympusprTracer(self, PartialVal.unknown(out_aval), None)
      eqn = new_eqn_recipe(self, tracers, [out_tracer], primitive,
                           params, effs, source)
      if effects.partial_eval_kept_effects.filter_in(effs):
        self.effect_handles.append(EffectHandle(tracers, eqn))
      out_tracer.recipe = eqn
      return out_tracer

  def process_call(self, primitive, f: lu.WrappedFun, tracers, params):
    tracers = map(self.to_olympuspr_tracer, tracers)
    rule = call_partial_eval_rules.get(primitive)
    if rule:
      return rule(self, primitive, f, tracers, params)

    update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
    in_knowns, in_avals, in_consts = partition_pvals([t.pval for t in tracers])
    # TODO(mattjj): check in_avals are consistent with f.in_type

    # We want to partially evaluate this call into two calls: one evaluated now
    # taking known values (in_consts) as inputs and producing known values
    # (out_consts) as outputs, and the other staged out as an eqn into the olympuspr
    # being built. The latter takes as input residuals (res) produced as outputs
    # of the first call, shared closed-over values (env), and explicit arguments
    # which were unknown to the first call (corresponding to in_avals).

    # Wrap f to perform the partial evaluation and plumb out aux data.
    f = f.with_unknown_names()
    f_ = trace_to_subolympuspr_nounits_fwd(f, self.tag, f.debug_info, False)
    f_, aux = partial_eval_wrapper_nounits(f_, tuple(in_knowns), tuple(in_avals))

    # Adjust parameters (e.g. donated_invars) for the call to be evaluated now.
    const_params = update_params(params, in_knowns, 0)

    # Run the call, getting known out vals and aux data used for staged-out call
    fun_and_args = (f_,) + tuple(in_consts)
    out = primitive.bind_with_trace(self.parent_trace, fun_and_args, const_params)
    fwds, out_knowns, out_type, olympuspr, env = aux()
    # Split apart known outputs from the original call and non-fwded residuals.
    out_consts, non_fwd_res = split_list(out, [sum(out_knowns)])
    in_consts_full = in_consts
    res = subs_list(fwds, in_consts_full, non_fwd_res)

    # Create the input tracers for the staged-out (unknown-value) call.
    res_tracers = map(self.instantiate_const, map(self.new_const, res))
    env_tracers = map(self.to_olympuspr_tracer, env)
    unknown_arg_tracers = [t for t in tracers if not t.is_known()]
    # Adjust parameters (e.g. donated_invars) for the staged-out call's args.
    num_new_args = len(res_tracers) + len(env_tracers)
    new_olympuspr = convert_constvars_olympuspr(olympuspr)
    if isinstance(primitive, core.ClosedCallPrimitive):
      new_olympuspr = close_olympuspr(new_olympuspr)  # type: ignore
    staged_params = dict(params, call_olympuspr=new_olympuspr)
    staged_params = update_params(staged_params, map(op.not_, in_knowns),
                                  num_new_args)
    out_tracers = [OlympusprTracer(self, PartialVal.unknown(a), None)
                   for a in out_type]
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    eqn = new_eqn_recipe(self, (*res_tracers, *env_tracers, *unknown_arg_tracers),
                         out_tracers, primitive, staged_params, olympuspr.effects,
                         source)
    for t in out_tracers: t.recipe = eqn
    return merge_lists(out_knowns, out_tracers, out_consts)

  def process_map(self, primitive, f: lu.WrappedFun, tracers, params):
    tracers = map(self.to_olympuspr_tracer, tracers)
    update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
    in_knowns, in_avals, in_consts = partition_pvals([t.pval for t in tracers])

    # This method is like process_call above, except:
    #   1. we delete an axis from mapped-over input avals' shapes, and
    #      analogously add an axis to mapped-over output avals' shapes;
    #   2. we update the in_axes and out_axes/out_axes_thunk parameters to
    #      reflect the inputs and outputs pruned from the unknown/known sides.

    # Map (delete an axis from) unknown inputs' avals as dictated by in_axes.
    unk_in_axes, const_in_axes = partition_list(in_knowns, params['in_axes'])
    in_avals_mapped = [mapped_aval(params['axis_size'], ax, aval)
                       for ax, aval in zip(unk_in_axes, in_avals)]

    # Wrap f to perform partial evaluation and plumb out aux data.
    f = trace_to_subolympuspr_nounits2(f, self.tag, f.debug_info, False)
    f, aux = partial_eval_wrapper_nounits(f, tuple(in_knowns),
                                          tuple(in_avals_mapped))
    # Adjust params for knowns (e.g. donated_invars, in_axes, out_axes_thunk)
    const_params = update_params(params, in_knowns, 0)  # handles donated_invars
    out_axes_thunk = params['out_axes_thunk']
    @as_hashable_function(closure=out_axes_thunk)
    def const_out_axes_thunk():
      out_knowns, _, olympuspr, _ = aux()
      _, out_axes = partition_list(out_knowns, out_axes_thunk())
      return tuple(out_axes) + (0,) * len(olympuspr.constvars)  # res mapped axis 0
    const_params = dict(const_params, in_axes=tuple(const_in_axes),
                        out_axes_thunk=const_out_axes_thunk)

    # Run the map, getting known out vals and aux data used for staged-out map.
    out = primitive.bind_with_trace(self.parent_trace, (f, *in_consts), const_params)
    out_knowns, out_avals_mapped, olympuspr, env = aux()
    # Split apart known outputs from the original call and residuals.
    out_consts, res = split_list(out, [len(out) - len(olympuspr.constvars)])

    # We can only check_olympuspr with the dynamic axis environment extended:
    with core.extend_axis_env_nd([(params['axis_name'], params['axis_size'])]):
      call_olympuspr = convert_constvars_olympuspr(olympuspr)

    # Compute staged and const out_axes, taking into account residuals.
    out_axes = params['out_axes_thunk']()
    staged_out_axes, _ = partition_list(out_knowns, out_axes)
    staged_in_axes = (0,) * len(res) + (None,) * len(env) + (*unk_in_axes,)

    # Create the input tracers for the staged-out (unknown-value) call.
    const_tracers = map(self.new_instantiated_const, res)
    env_tracers = map(self.to_olympuspr_tracer, env)
    unknown_arg_tracers = [t for t in tracers if not t.is_known()]
    # Adjust params for staged-out call on unknown values.
    num_new_args = len(const_tracers) + len(env_tracers)
    staged_params = update_params(params, map(op.not_, in_knowns), num_new_args)
    staged_params = dict(staged_params, in_axes=staged_in_axes,
                         out_axes=tuple(staged_out_axes), call_olympuspr=call_olympuspr)
    del staged_params['out_axes_thunk']
    # The outputs of the staged-out call are Tracers with the new eqn as recipe.
    out_avals = [unmapped_aval(params['axis_size'], ax, a)
                 for ax, a in zip(staged_out_axes, out_avals_mapped)]
    out_tracers = [OlympusprTracer(self, PartialVal.unknown(a), None)
                   for a in out_avals]
    effs = core.filter_named_axis_effects(olympuspr.effects, {params['axis_name']})
    src_info = source_info_util.current()
    eqn = new_eqn_recipe(self, (*const_tracers, *env_tracers, *unknown_arg_tracers),
                         out_tracers, primitive, staged_params, effs, src_info)
    for t in out_tracers: t.recipe = eqn

    return merge_lists(out_knowns, out_tracers, out_consts)

  def _current_truncated_name_stack(self):
    return source_info_util.current_name_stack()[len(self.name_stack):]

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, symbolic_zeros):
    tracers = map(self.to_olympuspr_tracer, tracers)
    if all(t.is_known() for t in tracers):
      with core.set_current_trace(self.parent_trace):
        vals = [t.pval[1] for t in tracers]
        return prim.bind(fun, jvp, *vals, symbolic_zeros=symbolic_zeros)
    # We assume non-trivial partial evaluation is only performed to build linear
    # functions, and hence we don't need to keep the custom JVP rule around.
    del jvp, symbolic_zeros
    with core.set_current_trace(self):
      return fun.call_wrapped(*tracers)

  def process_custom_transpose(self, prim, call, tracers, **params):
    tracers = map(self.to_olympuspr_tracer, tracers)
    res_ts, lin_ts = split_list(tracers, [params['res_tree'].num_leaves])
    assert all(t.is_known()     for t in res_ts)
    lin_all_known   = all(t.is_known()     for t in lin_ts)
    if lin_all_known:
      res_cvals = [t.pval[1] for t in res_ts]
      lin_cvals = [t.pval[1] for t in lin_ts]
      return prim.bind(call, *res_cvals, *lin_cvals, **params)
    else:
      out_tracers = [OlympusprTracer(self, PartialVal.unknown(aval), None)
                     for aval in params['out_types']]
      in_tracers = map(self.instantiate_const, tracers)
      new_params = dict(params, call=call)
      eqn = new_eqn_recipe(self, in_tracers, out_tracers, prim, new_params,
          core.no_effects, source_info_util.current())
      for t in out_tracers: t.recipe = eqn
      return out_tracers

  def process_custom_vjp_call(self, prim, f, fwd, bwd, tracers, out_trees, symbolic_zeros):
    tracers = map(self.to_olympuspr_tracer, tracers)
    if all(t.is_known() for t in tracers):
      vals = [t.pval[1] for t in tracers]
      with core.set_current_trace(self.parent_trace):
        return prim.bind(f, fwd, bwd, *vals, out_trees=out_trees,
                         symbolic_zeros=symbolic_zeros)

    tracers = map(self.instantiate_const, tracers)
    in_knowns = (False,) * len(tracers)
    in_avals = tuple(t.aval for t in tracers)
    f_ = trace_to_subolympuspr_nounits2(f, self.tag, f.debug_info, True)
    f_, aux = partial_eval_wrapper_nounits(f_, in_knowns, in_avals)
    params = dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros)
    res = prim.bind_with_trace(self.parent_trace, (f_, fwd, bwd), params)
    out_knowns, out_avals, olympuspr, env = aux()
    assert not any(out_knowns)
    res_tracers = map(self.instantiate_const, map(self.new_const, res))
    env_tracers = map(self.to_olympuspr_tracer, env)
    out_tracers = [OlympusprTracer(self, PartialVal.unknown(a), None)
                   for a in out_avals]
    closed_olympuspr = close_olympuspr(convert_constvars_olympuspr(olympuspr))

    @partial(lu.wrap_init, debug_info=fwd.debug_info)
    @_memoize
    def fwd_olympuspr_thunk(*zeros):
      fwd_ = _interleave_fun(fwd.with_unknown_names(), zeros)
      fwd_olympuspr, _, consts = trace_to_olympuspr_dynamic(fwd_, in_avals)
      return fwd_olympuspr, consts

    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    params = dict(
        call_olympuspr=closed_olympuspr,
        fwd_olympuspr_thunk=fwd_olympuspr_thunk,
        num_consts=len(res) + len(env),
        bwd=bwd,
        out_trees=out_trees,
        symbolic_zeros=symbolic_zeros
    )
    eqn = new_eqn_recipe(self, (*res_tracers, *env_tracers, *tracers),
                         out_tracers, prim, params, olympuspr.effects, source)
    for t in out_tracers: t.recipe = eqn
    return out_tracers

def partition_pvals(
    pvals: list[PartialVal]
  ) -> tuple[list[bool], list[AbstractValue], list[Any]]:
  knowns = [pval.is_known()  for pval in pvals                       ]
  avals  = [pval.get_aval()  for pval in pvals if not pval.is_known()]
  consts = [pval.get_known() for pval in pvals if     pval.is_known()]
  return knowns, avals, consts

@lu.transformation_with_aux2
def partial_eval_wrapper_nounits(
    f: Callable,
    store: lu.Store,
    in_knowns: Sequence[bool],
    in_avals: Sequence[AbstractValue],
    *in_consts: Any):
  in_avals_, in_consts_ = iter(in_avals), iter(in_consts)
  in_pvals = [PartialVal.known(next(in_consts_)) if known else
              PartialVal.unknown(next(in_avals_)) for known in in_knowns]
  sentinel = object()
  assert next(in_avals_, sentinel) is next(in_consts_, sentinel) is sentinel
  olympuspr, (*maybe_fwds, out_pvals, res, env) = f(in_pvals)
  out_knowns, out_avals, out_consts = partition_pvals(out_pvals)
  store.store((*maybe_fwds, out_knowns, out_avals, olympuspr, env))
  return (*out_consts, *res)

@lu.transformation_with_aux2
def partial_eval_wrapper_nounits2(
    f: Callable,
    store: lu.Store,
    in_knowns: Sequence[bool],
    in_avals: Sequence[AbstractValue],
    *in_consts: Any):
  in_avals_, in_consts_ = iter(in_avals), iter(in_consts)
  in_pvals = [PartialVal.known(next(in_consts_)) if known else
              PartialVal.unknown(next(in_avals_)) for known in in_knowns]
  sentinel = object()
  assert next(in_avals_, sentinel) is next(in_consts_, sentinel) is sentinel
  olympuspr, (*maybe_fwds, out_pvals, res, env) = f(in_pvals)
  out_knowns, _, out_consts = partition_pvals(out_pvals)
  res_avals = [typeof(r) for r in res]
  store.store((*maybe_fwds, out_knowns, res_avals, olympuspr, env))
  return (*out_consts, *res)

custom_partial_eval_rules: dict[Primitive, Callable] = {}
call_partial_eval_rules: dict[Primitive, Callable] = {}
call_param_updaters: dict[Primitive, Callable] = {}

def abstract_eval_fun(fun: Callable, *avals,
                      debug_info: core.DebugInfo, **params):
  _, avals_out, _ = trace_to_olympuspr_dynamic(
      lu.wrap_init(fun, params, debug_info=debug_info), avals)
  assert all(isinstance(aval, AbstractValue) for aval in avals_out)
  return avals_out


OlympusprTracerRecipe = Union[
    'OlympusprEqnRecipe', 'LambdaBinding', 'FreeVar', 'ConstVar', Literal,
]

class OlympusprTracer(Tracer):
  __slots__ = ['pval', 'recipe']

  def __init__(self, trace: OlympusprTrace, pval: PartialVal,
               recipe: OlympusprTracerRecipe | None):
    assert isinstance(pval, PartialVal)
    pv, const = pval
    self._trace = trace
    self.pval = pval
    self.recipe = recipe

  def __repr__(self):
    return f'Traced<{self.aval}:{self._trace}>'

  @property
  def aval(self) -> AbstractValue:
    return self.pval.get_aval()

  @property
  def parents(self) -> Sequence[OlympusprTracer]:
    if isinstance(self.recipe, OlympusprEqnRecipe):
      # TODO broadcast_in_dim can create a new tracer...
      return self.recipe.in_tracers
    else:
      return []

  def full_lower(self):
    known = self.pval.get_known()
    if known is not None:
      return core.full_lower(known)
    else:
      return self

  def is_known(self):
    return self.pval.is_known()

  def get_referent(self):
    if self.pval.is_known():
      return get_referent(self.pval.get_known())
    elif isinstance(self.recipe, (FreeVar, ConstVar, Literal)):
      return get_referent(self.recipe.val)  # pytype: disable=attribute-error
    else:
      return self


@profiler.annotate_function
def trace_to_olympuspr_nounits(
    fun: lu.WrappedFun, pvals: Sequence[PartialVal],
    instantiate: bool | Sequence[bool] = False,
  ) -> tuple[Olympuspr, list[PartialVal], list[core.Value]]:
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = OlympusprTrace(parent_trace, current_name_stack, TraceTag())
    with core.ensure_no_leaks(trace):
      fun = trace_to_subolympuspr_nounits(fun, trace, instantiate, fun.debug_info)
      with core.set_current_trace(trace):
        olympuspr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
        assert not env
      del trace, fun
      return olympuspr, out_pvals, consts

# TODO(mattjj): superfluous wrapper...?
@lu.transformation2
def trace_to_subolympuspr_nounits(
    f: Callable,
    trace: OlympusprTrace,
    instantiate: Sequence[bool] | bool,
    debug_info: core.DebugInfo,
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  out_tracers, olympuspr, out_consts, env = _trace_to_subolympuspr_nounits(
      f, trace, instantiate, in_pvals, debug_info)
  out_pvals = [t.pval for t in out_tracers]
  del out_tracers
  return olympuspr, (out_pvals, out_consts, env)

@lu.transformation2
def trace_to_subolympuspr_nounits2(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert isinstance(tag, TraceTag)
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = OlympusprTrace(parent_trace, current_name_stack, tag)
    out_tracers, olympuspr, out_consts, env = _trace_to_subolympuspr_nounits(
        f, trace, instantiate, in_pvals, debug_info)
    out_pvals = [t.pval for t in out_tracers]
    del out_tracers
  return olympuspr, (out_pvals, out_consts, env)

def _trace_to_subolympuspr_nounits(f: Callable, trace: OlympusprTrace,
                               instantiate: Sequence[bool] | bool,
                               in_pvals: Sequence[PartialVal],
                               debug_info: core.DebugInfo):
  in_knowns  = [pval.is_known()     for pval in in_pvals]
  in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
  in_tracers = [trace.new_arg(pval) for pval in in_pvals if not pval.is_known()]
  in_args = merge_lists(in_knowns, in_tracers, in_consts)
  with core.set_current_trace(trace):
    ans = f(*in_args)
  assert isinstance(ans, (list, tuple)), (
      f"Got unexpected return type when tracing function to olympuspr: {ans}")
  assert all(isinstance(x, core.Tracer) or core.valid_olympustype(x) for x in ans), (
      f"Got unexpected return type when tracing function to olympuspr: {ans}")
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(ans)
  out_tracers = map(trace.to_olympuspr_tracer, ans)
  out_tracers = [trace.instantiate_const(t) if inst else t
                 for inst, t in zip(instantiate, out_tracers)]
  out_tracers_ = [t for t in out_tracers if not t.is_known()]
  olympuspr, out_consts, env = tracers_to_olympuspr(
      in_tracers, out_tracers_, trace.effect_handles,
      debug_info.with_unknown_names())
  return out_tracers, olympuspr, out_consts, env

# The below variant implements an optimization where residuals which are also
# inputs are indicated in auxiliary data rather than passed as outputs.
# TODO(mattjj): update all callers to use this version, delete other version.
@lu.transformation2
def trace_to_subolympuspr_nounits_fwd(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = OlympusprTrace(parent_trace, current_name_stack, tag)
    with core.set_current_trace(trace):
      out_tracers, olympuspr, out_consts, env = _trace_to_subolympuspr_nounits(
          f, trace, instantiate, in_pvals, debug_info)
    out_pvals = [t.pval for t in out_tracers]

    # Which out_consts (aka residuals) are just forwarded inputs? Check obj id.
    in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
    id_map = {id(c): i for i, c in enumerate(in_consts)}
    fwds: list[int | None] = [id_map.get(id(c)) for c in out_consts]
    pruned_consts = [c for c, fwd in zip(out_consts, fwds) if fwd is None]

    del out_tracers
  return olympuspr, (fwds, out_pvals, pruned_consts, env)

# The below variant implements two optimizations:
#  1. residuals that are also primal inputs are indicated in aux data rather
#     than passed as outputs;
#  2. residuals that are also primal outputs are indicated in aux data rather
#     than passed as redundant outputs.
@lu.transformation2
def trace_to_subolympuspr_nounits_fwd2(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = OlympusprTrace(parent_trace, current_name_stack, tag)
    out_tracers, olympuspr, consts, env = _trace_to_subolympuspr_nounits(
        f, trace, instantiate, in_pvals, debug_info)
    out_pvals = [t.pval for t in out_tracers]

  # Which consts (aka residuals) are just forwarded inputs? Check obj id.
  in_consts  = [pval.get_known()    for pval in  in_pvals if    pval.is_known()]
  id_map = {id(c): i for i, c in enumerate(in_consts)}
  input_fwds: list[int | None] = [id_map.get(id(c)) for c in consts]

  # Which consts (aka residuals) are already primal outputs? Check obj id.
  out_consts = [pval.get_known()    for pval in out_pvals if    pval.is_known()]
  id_map = {id(c): i for i, c in enumerate(out_consts)}
  output_fwds: list[int | None] = [id_map.get(id(c)) for c in consts]

  pruned_consts = [c for c, f1, f2 in zip(consts, input_fwds, output_fwds)
                   if f1 is None and f2 is None]

  del out_tracers
  return olympuspr, (input_fwds, output_fwds, out_pvals, pruned_consts, env)


FreeVar = namedtuple('FreeVar', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])
LambdaBinding = namedtuple('LambdaBinding', [])
class OlympusprEqnRecipe(NamedTuple):
  eqn_id: Any
  in_tracers: Sequence[OlympusprTracer]
  out_tracer_refs: Sequence[ref[OlympusprTracer]]
  out_avals: Sequence[core.AbstractValue]
  primitive: Primitive
  params: dict[str, Any]
  effects: core.Effects
  source_info: source_info_util.SourceInfo
  ctx: OlympusprEqnContext

def new_eqn_recipe(trace: OlympusprTrace,
                   in_tracers: Sequence[OlympusprTracer],
                   out_tracers: Sequence[OlympusprTracer],
                   primitive: Primitive,
                   params: dict[str, Any],
                   effects: core.Effects,
                   source_info: source_info_util.SourceInfo,
                   ctx: OlympusprEqnContext | None = None) -> OlympusprEqnRecipe:
  # TODO(necula): move these checks to core.check_olympuspr, and call in more places
  if primitive.call_primitive or primitive.map_primitive:
    assert "call_olympuspr" in params
    assert ("donated_invars" not in params or
            len(params["donated_invars"]) == len(params["call_olympuspr"].invars))
  if primitive.map_primitive:
    assert ("in_axes" in params and
            len(params["in_axes"]) == len(params["call_olympuspr"].invars))
    assert ("donated_invars" in params and
            len(params["donated_invars"]) == len(params["call_olympuspr"].invars))
  out_avals = [t.aval for t in out_tracers]
  ctx = ctx or OlympusprEqnContext(
      config.compute_on_context_manager.value,
      config.threefry_partitionable.value,
      xla_metadata_lib.current_xla_metadata(),
  )
  return OlympusprEqnRecipe(next(trace.counter), tuple(in_tracers), map(ref, out_tracers),
                        out_avals, primitive, params, effects, source_info,
                        ctx)


def recipe_to_eqn(getvar: Callable[[OlympusprTracer], Atom],
                  recipe: OlympusprEqnRecipe) -> core.OlympusprEqn:
  (_, in_tracers, out_tracer_refs, out_avals, prim, params, eff, src,
   ctx) = recipe
  invars  = [getvar(t) for t in in_tracers]
  out_tracers = [t_ref() for t_ref in out_tracer_refs]
  outvars = [DropVar(a) if t is None else getvar(t)
             for a, t in zip(out_avals, out_tracers)]
  return new_olympuspr_eqn(invars, outvars, prim, params, eff, src, ctx)

def tracers_to_olympuspr(
  in_tracers: Sequence[OlympusprTracer],
  out_tracers: Sequence[OlympusprTracer],
  effect_handles: Sequence[Any],
  debug_info: core.DebugInfo,
  ) -> tuple[Olympuspr, tuple[Any, ...], tuple[Any, ...]]:
  """Constructs Olympuspr given tracers for inputs and outputs.

  Params:
    in_tracers: the tracers that were created for the function inputs
    out_tracers: the tracers that were output by the function.
    debug_info: the debug info for the function.

  Returns: a triple of a `Olympuspr`, a list of constant values corresponding to
    the `constvars` in the returned Olympusps, and a list of environment values.
    The vars for the environment values have been prepended to the Olympuspr's
    `invars`.
  """
  gensym = core.gensym()

  t_to_var: dict[TracerId, Var] = {}
  consts: dict[Var, Any] = {}
  env: dict[Var, OlympusprTracer] = {}
  constid_to_var: dict[ConstId, Var] = {}  # for deduplication

  def get_atom(t: OlympusprTracer) -> Atom:
    return t.recipe if type(t.recipe) is Literal else t_to_var[id(t)]

  def newvar(t: OlympusprTracer | None) -> Var:
    assert t is not None
    var = gensym(t.aval)
    var_ = t_to_var.setdefault(id(t), var)
    assert var is var_
    return var

  processed_eqn_ids = set()
  eqns: list[core.OlympusprEqn] = []
  is_high = False

  reachable = toposort
  tracers = reachable((*in_tracers, *out_tracers, *effect_handles))
  def sort_key(t):
    r = t.recipe
    return r.eqn_id if isinstance(r, OlympusprEqnRecipe) else -1
  tracers = sorted(tracers, key=sort_key)

  for t in tracers:
    r = t.recipe
    if isinstance(r, OlympusprEqnRecipe):
      # TODO broadcast_in_dim can create a new tracer, not present in parents
      if r.eqn_id not in processed_eqn_ids:
        in_atoms = map(get_atom, r.in_tracers)
        outvars = [DropVar(a) if rf() is None else newvar(rf())
                   for a, rf in zip(r.out_avals, r.out_tracer_refs)]
        eqns.append(new_olympuspr_eqn(in_atoms, outvars, r.primitive, r.params,
                                  r.effects, r.source_info, r.ctx))
        in_avals = [x.aval for x in in_atoms]
        is_high |= r.primitive.is_high(*in_avals, **r.params)
        processed_eqn_ids.add(r.eqn_id)
    elif isinstance(r, LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise core.escaped_tracer_error(t, f"Tracer not in input tracers: {t}")
      newvar(t)
    elif isinstance(r, ConstVar):
      var = constid_to_var.get(id(r.val))
      if var is None:
        var = constid_to_var[id(r.val)] = newvar(t)
        consts[var] = r.val
      t_to_var[id(t)] = var
    elif isinstance(r, FreeVar):
      env[newvar(t)] = r.val
    elif isinstance(r, Literal):
      pass
    elif r is None:
      assert False
    else:
      raise TypeError(r)

  env_vars, env_vals = unzip2(env.items())
  invars = [*env_vars, *map(get_atom, in_tracers)]
  const_vars, const_vals = unzip2(consts.items())
  outvars = map(get_atom, out_tracers)  # type: ignore[arg-type]
  olympuspr_effects = make_olympuspr_effects(const_vars, invars, outvars, eqns)
  is_high |= any(x.aval.is_high for x in it.chain(const_vars, invars, outvars))
  olympuspr = Olympuspr(const_vars, invars,  # type: ignore[arg-type]
                outvars, eqns, olympuspr_effects, debug_info, is_high)
  config.enable_checks.value and core.check_olympuspr(olympuspr)
  # del getvar  # needed to avoid cyclic-reference closure, apparently!
  return olympuspr, const_vals, env_vals

@weakref_lru_cache
def move_envvars(olympuspr: Olympuspr, which: tuple[bool, ...]) -> Olympuspr:
  constvars, envvars = partition_list(which, olympuspr.constvars)
  return olympuspr.replace(constvars=constvars, invars=[*envvars, *olympuspr.invars])

@weakref_lru_cache
def separate_consts(olympuspr: ClosedOlympuspr) -> tuple[ClosedOlympuspr, list[Any]]:
  """Moves the constvars to the start of invars and returns the consts explicitly."""
  return close_olympuspr(convert_constvars_olympuspr(olympuspr.olympuspr)), olympuspr.consts

@weakref_lru_cache
def convert_constvars_olympuspr(olympuspr: Olympuspr) -> Olympuspr:
  """Moves the constvars to the start of invars."""
  config.enable_checks.value and core.check_olympuspr(olympuspr)
  if olympuspr.debug_info.arg_names is None:
    arg_names = None
  else:
    arg_names = ("",) * len(olympuspr.constvars) + (*olympuspr.debug_info.arg_names,)
  dbg = olympuspr.debug_info._replace(arg_names=arg_names)
  lifted_olympuspr = olympuspr.replace(
      constvars=(), invars=olympuspr.constvars + olympuspr.invars, debug_info=dbg)
  config.enable_checks.value and core.check_olympuspr(lifted_olympuspr)
  return lifted_olympuspr

@weakref_lru_cache
def convert_invars_to_constvars(olympuspr: Olympuspr, n: int) -> Olympuspr:
  """Move n invars to constvars. Like an inverse of convert_constvars_olympuspr."""
  if n == 0:
    return olympuspr.replace()  # 'return olympuspr' would create cache reference cycle
  config.enable_checks.value and core.check_olympuspr(olympuspr)
  constvars, invars = split_list(olympuspr.invars, [n])
  if olympuspr.debug_info.arg_names is None:
    dbg = olympuspr.debug_info
  else:
    dbg = olympuspr.debug_info._replace(
        arg_names=olympuspr.debug_info.arg_names[n:])
  lifted_olympuspr = olympuspr.replace(constvars=tuple(constvars), invars=invars,
                               debug_info=dbg)
  config.enable_checks.value and core.check_olympuspr(lifted_olympuspr)
  return lifted_olympuspr

def convert_envvars_to_constvars(olympuspr: Olympuspr, num_env_vars: int) -> Olympuspr:
  if any(isinstance(eff, effects.OlympusprInputEffect) for eff in olympuspr.effects):
    raise NotImplementedError
  config.enable_checks.value and core.check_olympuspr(olympuspr)
  env_vars, invars = split_list(olympuspr.invars, [num_env_vars])
  converted_olympuspr = olympuspr.replace(constvars=olympuspr.constvars + env_vars,
                                  invars=invars)
  config.enable_checks.value and core.check_olympuspr(converted_olympuspr)
  return converted_olympuspr


def partial_eval_olympuspr_nounits(
    olympuspr: ClosedOlympuspr, unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
  ) -> tuple[ClosedOlympuspr, ClosedOlympuspr, list[bool], list[AbstractValue]]:
  """Unzip a olympuspr in two by data dependence into 'known' and 'unknown' parts.

  That is, given a olympuspr and a sequence of booleans indicating which olympuspr
  inputs (i.e. invars) are considered unknown, produce two olympusprs, a list of
  booleans representing which of the original olympuspr's outputs are unknown (i.e.
  have a data dependence on an unknown input), and a list of abstract values
  representing residuals (part of the first olympuspr's output and the second
  olympuspr's input). The two olympusprs result from partitioning the original olympuspr's
  first-order primitive applications based on whether all the inputs to the
  application are known (in which case the application is represented in the
  'known' olympuspr and its result is considered known) or whether any inputs to the
  application are unknown (in which case the application is represented in the
  'unknown' olympuspr and its result is considered unknown). Higher-order primitives
  are recursively unzipped in two.

  The `instantiate` argument can be used to ensure some outputs are lifted into
  the 'unknown' olympuspr.

  For example, give an input olympuspr:

    { lambda ; a:f32[] b:f32[]. let
        c:f32[] = cos a
        d:f32[] = sin a
        e:f32[] = neg d
        f:f32[] = mul e b
      in (c, f) }

  then applying this function with `unknowns=[False, True]` and
  `instantiate=False` produces as an output triple:

    # olympuspr_known
    { lambda ; a:f32[]. let
       b:f32[] = cos a
       c:f32[] = sin a
       d:f32[] = neg c
     in (b, d) }

    # olympuspr_unknown
    { lambda ; a:f32[] b:f32[]. let c:f32[] = mul b a in (c,) }

    # out_unknowns
    [False, True]

  Notice in particular that the first output (olympuspr_known) contains all the
  primitive applications which do not have a data dependence on an unknown
  input. Also notice the input and output types: the input type of the first
  olympuspr produced represents the type of the known inputs of the original olympuspr,
  and the output type of the second olympuspr produced represents the type of the
  unknown outputs of the original olympuspr.

  In the above example, the output of olympuspr_known named `d` is a _residual_
  output, and corresponds to the input named `a` in olympuspr_unknown. In general,
  olympuspr_known will produce extra outputs (at the end of its output list)
  corresponding to intermediate values of the original olympuspr which must be
  passed to olympuspr_unknown (as leading inputs).
  """
  instantiate = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  return _partial_eval_olympuspr_nounits(olympuspr, tuple(unknowns), instantiate, False)[:-1]

def partial_eval_olympuspr_nounits_fwd(
    olympuspr: ClosedOlympuspr, unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
    fwd: bool | Sequence[bool] = True,
) -> tuple[ClosedOlympuspr, ClosedOlympuspr, list[bool], list[AbstractValue], list[int | None]]:
  instantiate = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  fwd = tuple(fwd) if isinstance(fwd, list) else fwd
  return _partial_eval_olympuspr_nounits(olympuspr, tuple(unknowns), instantiate, fwd)

@weakref_lru_cache
def _partial_eval_olympuspr_nounits(
    olympuspr: ClosedOlympuspr, in_unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool], fwd: bool | Sequence[bool]):
  f = lu.wrap_init(core.olympuspr_as_fun(olympuspr), debug_info=olympuspr.olympuspr.debug_info)

  cell = []
  def fun(*known_vals_in):
    known_vals_in_ = iter(known_vals_in)
    unknown_avals = (a for a, uk in zip(olympuspr.in_avals, in_unknowns) if uk)
    in_pvals = [PartialVal.unknown(next(unknown_avals)) if uk
                else PartialVal.known(next(known_vals_in_)) for uk in in_unknowns]
    assert next(known_vals_in_, None) is next(unknown_avals, None) is None
    olympuspr_unknown_, (fwds, out_pvals, residuals, ()) = trace_to_subolympuspr_nounits_fwd(
        f, TraceTag(), olympuspr.olympuspr.debug_info, instantiate).call_wrapped(in_pvals)
    olympuspr_unknown = convert_constvars_olympuspr(olympuspr_unknown_)
    out_unknowns = [not pval.is_known() for pval in out_pvals]
    if type(fwd) is bool and not fwd:
      residuals_ = iter(residuals)
      residuals = [next(residuals_) if f is None else known_vals_in[f]
                   for f in fwds]
      assert next(residuals_, None) is None
      fwds = [None] * len(fwds)
    else:
      if type(fwd) is tuple:
        fwd_ = [f for f, uk in zip(fwd, in_unknowns) if not uk]
        residuals_, residuals = iter(residuals), []
        fwds = [residuals.append(next(residuals_)) if f is None else
                residuals.append(known_vals_in[f]) if not fwd_[f] else
                f for f in fwds]
      fwds, residuals = _include_consts_in_fwds(olympuspr.consts, fwds, residuals)
    res_avals = [core.get_aval(r) for r in residuals]
    cell.append((out_unknowns, olympuspr_unknown, res_avals, fwds))
    known_vals_out = [pval.get_known() for pval in out_pvals if pval.is_known()]
    return [*known_vals_out, *residuals]

  known_avals = [a for a, uk in zip(olympuspr.in_aval_qdds, in_unknowns) if not uk]
  olympuspr_known, _, consts_known = trace_to_olympuspr_dynamic(
      lu.wrap_init(fun, debug_info=f.debug_info.with_unknown_names()),
      known_avals)
  (out_unknowns, olympuspr_unknown, res_avals, fwds), = cell  # pytype: disable=bad-unpacking

  if config.enable_checks.value:
    core.check_olympuspr(olympuspr_known)
    core.check_olympuspr(olympuspr_unknown)

  closed_olympuspr_known = ClosedOlympuspr(olympuspr_known, consts_known)
  closed_olympuspr_unknown = ClosedOlympuspr(olympuspr_unknown, ())
  return closed_olympuspr_known, closed_olympuspr_unknown, out_unknowns, res_avals, fwds

def _include_consts_in_fwds(consts, fwds, residuals):
  if all(f is None for f in fwds):
    return fwds, residuals
  dummys = [object() for _ in range(max(f for f in fwds if f is not None) + 1)]
  residuals_ = iter(residuals)
  residuals = [next(residuals_) if f is None else dummys[f] for f in fwds]
  assert next(residuals_, None) is None
  idxs = {id(x): i for i, x in enumerate((*consts, *dummys))}
  fwds = [idxs.get(id(r)) for r in residuals]
  residuals = [r for r in residuals if id(r) not in idxs]
  return fwds, residuals


def partial_eval_olympuspr_custom(
    olympuspr: Olympuspr,
    in_unknowns: Sequence[bool],
    in_inst: bool | Sequence[bool],
    ensure_out_unknowns: bool | Sequence[bool],
    ensure_out_inst: bool | Sequence[bool],
    saveable: Callable[..., RematCases_],
  ) -> tuple[Olympuspr, Olympuspr, list[bool], list[bool], int]:
  *outs, num_res_ref = partial_eval_olympuspr_stateful(
      olympuspr, in_unknowns, in_inst, ensure_out_unknowns, ensure_out_inst, saveable)
  if num_res_ref:
    raise ValueError("Cannot use `partial_eval_olympuspr_custom` with stateful olympusprs.")
  return *outs,  # type: ignore

def partial_eval_olympuspr_stateful(
    olympuspr: Olympuspr,
    in_unknowns: Sequence[bool],
    in_inst: bool | Sequence[bool],
    ensure_out_unknowns: bool | Sequence[bool],
    ensure_out_inst: bool | Sequence[bool],
    saveable: Callable[..., RematCases_] | None,
  ) -> tuple[Olympuspr, Olympuspr, list[bool], list[bool], int, int]:
  if type(in_inst) is bool:
    in_inst = (in_inst,) * len(olympuspr.invars)
  if type(ensure_out_unknowns) is bool:
    ensure_out_unknowns = (ensure_out_unknowns,) * len(olympuspr.outvars)
  if type(ensure_out_inst) is bool:
    ensure_out_inst = (ensure_out_inst,) * len(olympuspr.outvars)
  if saveable is None:
    saveable = everything_saveable
  olympuspr_known, olympuspr_staged, out_unknowns, out_inst, num_res, num_res_ref = \
      _partial_eval_olympuspr_custom_cached(
          olympuspr, tuple(in_unknowns), tuple(in_inst), tuple(ensure_out_unknowns),
          tuple(ensure_out_inst), saveable)
  return olympuspr_known, olympuspr_staged, out_unknowns, out_inst, num_res, num_res_ref

everything_saveable = lambda *_, **__: True

@weakref_lru_cache
def _partial_eval_olympuspr_custom_cached(
    olympuspr: Olympuspr,
    in_unknowns: tuple[bool, ...],
    in_inst: tuple[bool, ...],
    ensure_out_unknowns: tuple[bool, ...],
    ensure_out_inst: tuple[bool, ...],
    saveable: Callable[..., RematCases_],
  ) -> tuple[Olympuspr, Olympuspr, list[bool], list[bool], int, int]:
  env: dict[Var, tuple[bool, bool]] = {}
  residuals: OrderedSet[Var] = OrderedSet()
  residual_refs: OrderedSet[Var] = OrderedSet()

  def read(x: Atom) -> tuple[bool, bool]:
    if type(x) is Var:
      return env[x]
    return (False, True)

  def write(unk: bool, inst: bool, v: Var) -> None:
    assert (unk, inst) != (True, False)
    env[v] = (unk, inst)

  def ensure_instantiated(inst: bool, x: Atom) -> Atom:
    if type(x) is Var and not inst:
      residuals.add(x)
    return x

  def has_effects(effects) -> bool:
    not_really_effects = (core.NamedAxisEffect, core.InternalMutableArrayEffect)
    return any(not isinstance(e, not_really_effects) for e in effects)

  known_eqns, staged_eqns = [], []
  foreach(write, in_unknowns, in_inst, olympuspr.invars)
  foreach(partial(write, False, True), olympuspr.constvars)
  for eqn in olympuspr.eqns:
    unks_in, inst_in = unzip2(map(read, eqn.invars))
    rule = partial_eval_olympuspr_custom_rules.get(eqn.primitive)
    if rule:
      eqn1, eqn2, unks_out, inst_out, res = rule(saveable, unks_in, inst_in, eqn)
      eqn1 and known_eqns.append(eqn1); eqn2 and staged_eqns.append(eqn2)  # type: ignore
      for r in res:
        if isinstance(r.aval, AbstractRef):
          residual_refs.add(r)
        else:
          residuals.add(r)
      foreach(write, unks_out, inst_out, eqn.outvars)
    elif any(unks_in):
      inputs = map(ensure_instantiated, inst_in, eqn.invars)
      staged_eqns.append(eqn.replace(invars=inputs))
      foreach(partial(write, True, True), eqn.outvars)
    else:
      known_eqns.append(eqn)
      # If it's an effectful primitive, we always to run and avoid staging it.
      policy = ensure_enum(saveable(
          eqn.primitive, *[x.aval for x in eqn.invars], **eqn.params))
      if has_effects(eqn.effects) or isinstance(policy, SaveableType):
        foreach(partial(write, False, False), eqn.outvars)
      elif isinstance(policy, Offloadable):
        # TODO(slebedev): This is a legit error which requires a BUILD fix.
        from olympus._src.dispatch import device_put_p, ArrayCopySemantics  # type: ignore
        resvars = [Var(v.aval.update(memory_space=core.mem_kind_to_space(policy.dst)))
                   for v in eqn.outvars]
        offload_eqn = core.OlympusprEqn(
            eqn.outvars, resvars, device_put_p,
            dict(
                devices=(core.mem_kind_to_space(policy.dst),) * len(eqn.outvars),
                srcs=(None,),
                copy_semantics=(ArrayCopySemantics.ALWAYS_COPY,),
            ),
            set(), source_info_util.new_source_info(),
            OlympusprEqnContext(None, False))
        known_eqns.append(offload_eqn)
        # resvars are known and available in the backward olympuspr.
        foreach(partial(write, False, True), resvars)
        assert all(o.aval.memory_space == core.mem_kind_to_space(policy.src)  # type: ignore
                   for o in eqn.outvars)
        residuals.update(resvars)
        reload_eqn = core.OlympusprEqn(
            resvars, eqn.outvars, device_put_p,
            dict(
              devices=(core.mem_kind_to_space(policy.src),) * len(resvars),
              srcs=(None,),
              copy_semantics=(ArrayCopySemantics.ALWAYS_COPY,)
            ),
            set(), source_info_util.new_source_info(),
            OlympusprEqnContext(None, False))
        staged_eqns.append(reload_eqn)
        # outvars are known and available in the backward olympuspr.
        foreach(partial(write, False, True), eqn.outvars)
      else:
        assert isinstance(policy, RecomputeType)
        inputs = map(ensure_instantiated, inst_in, eqn.invars)
        staged_eqns.append(eqn.replace(invars=inputs))
        foreach(partial(write, False, True), eqn.outvars)
  unzipped = unzip2(map(read, olympuspr.outvars))
  out_unknowns, out_inst = list(unzipped[0]), list(unzipped[1])
  assert all(type(v) is Var for v in residuals), residuals

  for x, inst, ensure_inst in zip(olympuspr.outvars, out_inst, ensure_out_inst):
    if ensure_inst: ensure_instantiated(inst, x)
  out_unknowns = map(op.or_, out_unknowns, ensure_out_unknowns)
  out_inst     = map(op.or_, out_inst,     ensure_out_inst)

  ins_known, _ = partition_list(in_unknowns, olympuspr.invars)
  outs_known, _ = partition_list(out_unknowns, olympuspr.outvars)
  ref_res_is_input = [r in ins_known for r in residual_refs]
  non_input_res_refs, _ = partition_list(ref_res_is_input, list(residual_refs))
  ins_known_and_ref_res = [*ins_known, *non_input_res_refs]
  known_outvars = [*outs_known, *residuals]
  known_effects = make_olympuspr_effects(olympuspr.constvars, ins_known_and_ref_res,
                                     known_outvars, known_eqns)

  # TODO(mattjj,necula): debug info should be updated here
  olympuspr_known = olympuspr.replace(
      invars=ins_known_and_ref_res, outvars=known_outvars,
      eqns=known_eqns, effects=known_effects,
      debug_info=olympuspr.debug_info.with_unknown_names())
  config.enable_checks.value and core.check_olympuspr(olympuspr_known)

  _, ins_staged = partition_list(in_inst, olympuspr.invars)
  _, outs_staged = partition_list(out_inst, olympuspr.outvars)
  staged_invars = [*residuals, *non_input_res_refs, *ins_staged]
  staged_effects = make_olympuspr_effects(olympuspr.constvars, staged_invars,
                                      outs_staged, staged_eqns)
  # TODO(mattjj,necula): debug info should be updated here
  olympuspr_staged = olympuspr.replace(
      invars=staged_invars, outvars=outs_staged, eqns=staged_eqns,
      effects=staged_effects,
      debug_info=olympuspr.debug_info.with_unknown_names())
  config.enable_checks.value and core.check_olympuspr(olympuspr_staged)

  return (olympuspr_known, olympuspr_staged, out_unknowns, out_inst, len(residuals),
          len(non_input_res_refs))


MemoryKind = str

class RecomputeType: pass
Recompute = RecomputeType()

class SaveableType: pass
Saveable = SaveableType()

class Offloadable(NamedTuple):
  src: MemoryKind
  dst: MemoryKind

RematCases = Union[RecomputeType, SaveableType, Offloadable]
RematCases_ = Union[RematCases, bool]

def ensure_enum(case: bool | RematCases) -> RematCases:
  if isinstance(case, bool):
    return Saveable if case else Recompute
  if not isinstance(case, (RecomputeType, SaveableType, Offloadable)):
    msg = ("Value returned by a remat policy should be a bool or"
           " `ad_checkpoint.Recompute`, `ad_checkpoint.Saveable` or"
           " `ad_checkpoint.Offloadable(...)`."
           f" Got {case} of type {type(case)}.")
    if isinstance(case, Offloadable):
      msg += ("Did you return `Offloadable` instead of an instantiated"
              " `Offloadable(...)`?")
    raise TypeError(msg)
  return case

# A primitive rule for policy-driven partial evaluation returns a 5-tuple
# with the components representing, respectively:
#  * the OlympusprEqn for the 'known' side (or None if there is no known component),
#  * the OlympusprEqn for the 'unknown' side (or None),
#  * a list of booleans indicating which of the original outputs are unknown,
#  * a list of booleans indicating which of the original outputs are
#    instantiated (i.e. available) in the 'unknown' side,
#  * a list of Var instances representing residuals to be added (i.e. to be
#    plumbed as outputs of the 'known' side olympuspr and added as input binders to
#    the 'unknown' olympuspr).
PartialEvalCustomResult = tuple[Union[OlympusprEqn, None], Union[OlympusprEqn, None],
                                Sequence[bool], Sequence[bool], list[Var]]
PartialEvalCustomRule = Callable[
    [Callable[..., RematCases_], Sequence[bool], Sequence[bool], OlympusprEqn],
    PartialEvalCustomResult]
partial_eval_olympuspr_custom_rules: dict[Primitive, PartialEvalCustomRule] = {}

def partial_eval_olympuspr_custom_rule_not_implemented(
    name: str, saveable: Callable[..., RematCases_], unks_in: Sequence[bool],
    inst_in: Sequence[bool], eqn: OlympusprEqn) -> PartialEvalCustomResult:
  msg = (f'custom-policy remat rule not implemented for {name}, '
         'open a feature request at https://github.com/olympus-ml/olympus/issues!')
  raise NotImplementedError(msg)


ParamsUpdater = Callable[[Sequence[bool], Sequence[bool], Sequence[bool],
                          Sequence[bool], int, dict, dict],
                         tuple[dict, dict]]
ResAvalUpdater = Callable[[dict[str, Any], AbstractValue], AbstractValue]
def _default_res_aval_updater(
    params: dict[str, Any], aval: AbstractValue) -> AbstractValue:
  return aval


def call_partial_eval_custom_rule(
    olympuspr_param_name: str, params_updater: ParamsUpdater,
    saveable: Callable[..., RematCases_], unks_in: list[bool], inst_in: list[bool],
    eqn: OlympusprEqn, *, res_aval: ResAvalUpdater = _default_res_aval_updater,
    ctx = contextlib.nullcontext,
  ) -> tuple[OlympusprEqn, OlympusprEqn, Sequence[bool], Sequence[bool], list[Var]]:
  olympuspr = eqn.params[olympuspr_param_name]
  with ctx(eqn.params):
    olympuspr_known, olympuspr_staged, unks_out, inst_out, num_res = \
        partial_eval_olympuspr_custom(olympuspr, unks_in, inst_in, False, False, saveable)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  _, ins_staged = partition_list(inst_in, eqn.invars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  params_known = {**eqn.params, olympuspr_param_name: olympuspr_known}
  params_staged = {**eqn.params, olympuspr_param_name: olympuspr_staged}
  params_known, params_staged = params_updater(
      unks_in, inst_in, map(op.not_, unks_out), inst_out, num_res, params_known,
      params_staged)
  residuals = [Var(res_aval(params_known, var.aval))
               for var in olympuspr_staged.invars[:num_res]]
  eqn_known = new_olympuspr_eqn(
      ins_known, [*out_binders_known, *residuals], eqn.primitive, params_known,
      core.eqn_effects(olympuspr_known), eqn.source_info, eqn.ctx)
  eqn_staged = new_olympuspr_eqn(
      [*residuals, *ins_staged], out_binders_staged, eqn.primitive,
      params_staged, core.eqn_effects(olympuspr_staged), eqn.source_info,
      eqn.ctx)
  assert len(eqn_staged.invars) == len(olympuspr_staged.invars)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  return eqn_known, eqn_staged, unks_out, inst_out, new_inst + residuals

# TODO(mattjj): unify with ParamsUpdater (this one takes an extra int)
ParamsUpdater2 = Callable[[Sequence[bool], Sequence[bool], Sequence[bool],
                           Sequence[bool], int, int, dict, dict],
                          tuple[dict, dict]]

def closed_call_partial_eval_custom_rule(
    olympuspr_param_name: str, params_updater: ParamsUpdater2,
    saveable: Callable[..., RematCases_], unks_in: list[bool], inst_in: list[bool],
    eqn: OlympusprEqn, *, res_aval: ResAvalUpdater = _default_res_aval_updater,
  ) -> tuple[OlympusprEqn, OlympusprEqn, Sequence[bool], Sequence[bool], list[Var]]:
  # TODO(sharadmv,mattjj): dedup this rule with call_partial_eval_custom_rule.
  dropvars = tuple(isinstance(v, DropVar) for v in eqn.outvars)
  olympuspr_known, olympuspr_staged, unks_out, inst_out, num_res_ref, num_res_val, out_fwd = \
      _closed_olympuspr_partial_eval_custom_cached(
          eqn.params[olympuspr_param_name], (*unks_in,), (*inst_in,), dropvars, saveable)
  num_res = num_res_ref + num_res_val
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  _, ins_staged = partition_list(inst_in, eqn.invars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  params_known = {**eqn.params, olympuspr_param_name: olympuspr_known}
  params_staged = {**eqn.params, olympuspr_param_name: olympuspr_staged}
  params_known, params_staged = params_updater(
      unks_in, inst_in, map(op.not_, unks_out), inst_out,
      sum(f is None for f in out_fwd), num_res, params_known, params_staged)
  res_val_binders, res_ref_binders = split_list(
      [Var(res_aval(params_known, v))
       for v in olympuspr_staged.in_avals[:num_res]], [num_res_val])
  res_val_binders = [v for v, f in zip(res_val_binders, out_fwd) if f is None]
  res_val_vars = subs_list(out_fwd, out_binders_known, res_val_binders)
  eqn_known = new_olympuspr_eqn(
      [*ins_known, *res_ref_binders], [*out_binders_known, *res_val_binders],
      eqn.primitive, params_known, core.eqn_effects(olympuspr_known),
      eqn.source_info, eqn.ctx)
  eqn_staged = new_olympuspr_eqn(
      [*res_val_vars, *res_ref_binders, *ins_staged], out_binders_staged,
      eqn.primitive, params_staged, core.eqn_effects(olympuspr_staged),
      eqn.source_info, eqn.ctx)
  assert len(eqn_staged.invars) == len(olympuspr_staged.in_avals)
  assert len(ins_known) + len(res_ref_binders) == len(olympuspr_known.olympuspr.invars)
  assert len(ins_staged) + len(res_ref_binders) + len(res_val_vars) == len(olympuspr_staged.olympuspr.invars)
  assert len(out_binders_known) + len(res_val_binders) == len(olympuspr_known.olympuspr.outvars)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  new_vars = [*new_inst, *res_val_vars, *res_ref_binders]
  return eqn_known, eqn_staged, unks_out, inst_out, new_vars

@weakref_lru_cache
def _closed_olympuspr_partial_eval_custom_cached(
    olympuspr: ClosedOlympuspr, unks_in: tuple[bool, ...], inst_in: tuple[bool, ...],
    dropvars: tuple[bool, ...], saveable: Callable
    ) -> tuple[ClosedOlympuspr, ClosedOlympuspr, Sequence[bool], Sequence[bool],
               int, int, Sequence[int | None]]:
  olympuspr_known_, olympuspr_staged_, unks_out, inst_out, num_res_val, num_res_ref = \
      partial_eval_olympuspr_stateful(olympuspr.olympuspr, unks_in, inst_in,
                                  False, False, saveable)

  # Compute which residual value outputs are also *undropped* primal outputs.
  num_out_primals = len(olympuspr_known_.outvars) - num_res_val
  out_vars, res_vars = split_list(olympuspr_known_.outvars, [num_out_primals])
  out_dropvars_known, _ = partition_list(unks_out, dropvars)
  idx_map = {id(v): i for i, (v, b) in enumerate(zip(out_vars, out_dropvars_known))
             if not b}
  out_fwd = [idx_map.get(id(v)) for v in res_vars]

  # Prune olympuspr_known_ outputs by removing forwards.
  olympuspr_known_ = prune_olympuspr_outputs(
      olympuspr_known_, [True] * num_out_primals + [f is None for f in out_fwd])

  olympuspr_known = core.ClosedOlympuspr(olympuspr_known_, olympuspr.consts)
  olympuspr_staged = core.ClosedOlympuspr(olympuspr_staged_, olympuspr.consts)
  return olympuspr_known, olympuspr_staged, unks_out, inst_out, num_res_ref, num_res_val, out_fwd


partial_eval_olympuspr_custom_rules[core.call_p] = \
    partial(call_partial_eval_custom_rule, 'call_olympuspr',
            lambda _, __, ___, ____, _____, x, y: (x, y))
partial_eval_olympuspr_custom_rules[core.closed_call_p] = \
    partial(closed_call_partial_eval_custom_rule, 'call_olympuspr',
            lambda _, __, ___, ____, _____, ______, x, y: (x, y))


def _olympuspr_forwarding(olympuspr: Olympuspr) -> list[int | None]:
  # Compute which inputs are just forwarded to outputs.
  fwds: dict[Var, Atom] = dict(zip(olympuspr.invars, olympuspr.invars))
  for eqn in olympuspr.eqns:
    if eqn.primitive in forwarding_rules:
      eqn = eqn.replace(invars=[a if type(a) is Literal else fwds.get(a, a)  # type: ignore
                                for a in eqn.invars])
      fwd_idx, _ = forwarding_rules[eqn.primitive](eqn)
      for v_orig, idx in zip(eqn.outvars, fwd_idx):
        if idx is not None:
          fwds[v_orig] = eqn.invars[idx]
  idxs: dict[Var, int] = {v: i for i, v in enumerate(olympuspr.invars)}
  return [None if type(v) is Literal else idxs.get(fwds.get(v))  # type: ignore
          for v in olympuspr.outvars]


def prune_olympuspr_outputs(olympuspr: Olympuspr, used_outputs: Sequence[bool]) -> Olympuspr:
  return _prune_olympuspr_outputs_cached(olympuspr, tuple(used_outputs))

def _prune_olympuspr_outputs(olympuspr: Olympuspr, used_outputs: tuple[bool, ...]) -> Olympuspr:
  outvars = [v for v, b in zip(olympuspr.outvars, used_outputs) if b]
  dbg = core.DebugInfo(
      olympuspr.debug_info.traced_for, olympuspr.debug_info.func_src_info,
      olympuspr.debug_info.arg_names,
      olympuspr.debug_info.filter_result_paths(used_outputs))
  new_olympuspr = olympuspr.replace(outvars=outvars, debug_info=dbg)
  config.enable_checks.value and core.check_olympuspr(new_olympuspr)
  return new_olympuspr
_prune_olympuspr_outputs_cached = weakref_lru_cache(_prune_olympuspr_outputs)

def prune_closed_olympuspr_outputs(
    olympuspr: ClosedOlympuspr, used_outputs: Sequence[bool]
) -> ClosedOlympuspr:
  return _prune_closed_olympuspr_outputs(olympuspr, tuple(used_outputs))

@partial(weakref_lru_cache, trace_context_in_key=False)
def _prune_closed_olympuspr_outputs(
    olympuspr: ClosedOlympuspr, used_outputs: tuple[bool, ...]
) -> ClosedOlympuspr:
  return ClosedOlympuspr(_prune_olympuspr_outputs(olympuspr.olympuspr, used_outputs),
                     olympuspr.consts)


def dce_olympuspr(olympuspr: Olympuspr, used_outputs: Sequence[bool],
              instantiate: bool | Sequence[bool] = False,
              ) -> tuple[Olympuspr, list[bool]]:
  """Runs dead-code elementation on a given olympuspr.

  Args:
    olympuspr: The olympuspr to DCE.
    used_outputs: A list of bools indicating which outputs are used.
    instantiate: A bool or a list of bools indicating which inputs should be
      considered used, regardless of whether they are actually used in a olympuspr.
      If a bool, the same value is used for all inputs.

  Returns:
    A tuple of ``(new_olympuspr, used_inputs)``.
  """
  if type(instantiate) is bool:
    instantiate = (instantiate,) * len(olympuspr.invars)
  return _dce_olympuspr(olympuspr, tuple(used_outputs), tuple(instantiate))


def dce_olympuspr_consts(olympuspr: Olympuspr, used_outputs: Sequence[bool],
                     instantiate: bool | Sequence[bool] = False,
                     ) -> tuple[Olympuspr, list[bool], list[bool]]:
  olympuspr_ = convert_constvars_olympuspr(olympuspr)
  new_olympuspr, used_inputs_ = dce_olympuspr(olympuspr_, used_outputs, instantiate)
  used_consts, used_inputs = split_list(used_inputs_, [len(olympuspr.constvars)])
  if sum(used_consts):
    new_olympuspr = convert_invars_to_constvars(new_olympuspr, sum(used_consts))
  return new_olympuspr, used_consts, used_inputs


def has_effects(eqn: OlympusprEqn) -> bool:
  effs = {e for e in eqn.effects if not isinstance(e, core.NamedAxisEffect)
          and not isinstance(e, ReadEffect)}
  return bool(effs)


@weakref_lru_cache
def _dce_olympuspr(olympuspr: Olympuspr, used_outputs: tuple[bool, ...],
               instantiate: tuple[bool, ...]
               ) -> tuple[Olympuspr, list[bool]]:
  env: dict[Var, bool] = {}

  def read(v: Var) -> bool:
    return env.get(v, False)

  def write(x: Atom, b: bool) -> None:
    if type(x) is Var:
      env[x] = read(x) or b

  new_eqns = []
  foreach(write, olympuspr.outvars, used_outputs)
  for eqn in olympuspr.eqns[::-1]:
    used_outs = map(read, eqn.outvars)
    rule = dce_rules.get(eqn.primitive, _default_dce_rule)
    used_ins, new_eqn = rule(used_outs, eqn)
    if new_eqn is not None:
      new_eqns.append(new_eqn)
    foreach(write, eqn.invars, used_ins)
  used_inputs = map(read, olympuspr.invars)
  used_inputs = map(op.or_, instantiate, used_inputs)

  invars = [v for v, b in zip(olympuspr.invars, used_inputs)   if b]
  outvars = [v for v, b in zip(olympuspr.outvars, used_outputs) if b]
  eqns = new_eqns[::-1]
  olympuspr_effects = make_olympuspr_effects(olympuspr.constvars, invars, outvars, eqns)

  dbg = core.DebugInfo(
      olympuspr.debug_info.traced_for, olympuspr.debug_info.func_src_info,
      olympuspr.debug_info.filter_arg_names(used_inputs),
      olympuspr.debug_info.filter_result_paths(used_outputs))
  new_olympuspr = olympuspr.replace(invars=invars, outvars=outvars, eqns=eqns,
                            effects=olympuspr_effects, debug_info=dbg)
  config.enable_checks.value and core.check_olympuspr(new_olympuspr)

  return new_olympuspr, used_inputs

DCERule = Callable[[list[bool], OlympusprEqn],
                   tuple[list[bool], Union[OlympusprEqn, None]]]

def _default_dce_rule(
    used_outs: list[bool], eqn: OlympusprEqn
  ) -> tuple[list[bool], OlympusprEqn | None]:
  if not any(used_outs) and not has_effects(eqn):
    return [False] * len(eqn.invars), None
  return [True] * len(eqn.invars), eqn

dce_rules: dict[Primitive, DCERule] = {}


def dce_olympuspr_call_rule(used_outputs: list[bool], eqn: OlympusprEqn
                        ) -> tuple[list[bool], OlympusprEqn | None]:
  if not any(used_outputs) and not has_effects(eqn):
    return [False] * len(eqn.invars), None
  new_olympuspr, used_inputs = dce_olympuspr(eqn.params['call_olympuspr'], used_outputs)
  new_params = dict(eqn.params, call_olympuspr=new_olympuspr)
  update_params = call_param_updaters.get(eqn.primitive)
  if update_params:
    new_params = update_params(new_params, used_inputs, 0)
  if not any(used_inputs) and not any(used_outputs) and not new_olympuspr.effects:
    return used_inputs, None
  else:
    new_eqn = new_olympuspr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, new_olympuspr.effects, eqn.source_info, eqn.ctx)
    return used_inputs, new_eqn

dce_rules[core.call_p] = dce_olympuspr_call_rule


@weakref_lru_cache
def _cached_closed_call_dce(olympuspr_, used_outputs: tuple[bool, ...]
                            ) -> tuple[core.ClosedOlympuspr, list[bool]]:
  olympuspr, consts = olympuspr_.olympuspr, olympuspr_.consts
  new_olympuspr, used_inputs = dce_olympuspr(olympuspr, used_outputs)
  return core.ClosedOlympuspr(new_olympuspr, consts), used_inputs

def dce_olympuspr_closed_call_rule(used_outputs: list[bool], eqn: OlympusprEqn
                               ) -> tuple[list[bool], OlympusprEqn | None]:
  # TODO(mattjj): de-duplicate with above rule?
  if not any(used_outputs) and not has_effects(eqn):
    return [False] * len(eqn.invars), None
  olympuspr_ = eqn.params['call_olympuspr']
  closed_olympuspr, used_inputs = _cached_closed_call_dce(olympuspr_, tuple(used_outputs))
  effects = core.eqn_effects(closed_olympuspr)
  new_params = dict(eqn.params, call_olympuspr=closed_olympuspr)
  new_eqn = new_olympuspr_eqn(
      [v for v, used in zip(eqn.invars, used_inputs) if used],
      [v for v, used in zip(eqn.outvars, used_outputs) if used],
      eqn.primitive, new_params, effects, eqn.source_info, eqn.ctx)
  return used_inputs, new_eqn
dce_rules[core.closed_call_p] = dce_olympuspr_closed_call_rule

@weakref_lru_cache
def close_olympuspr(olympuspr: Olympuspr) -> ClosedOlympuspr:
  # The `olympuspr.replace()` is making a copy of the Olympuspr, without which
  # the cache value would have a strong reference to the same Olympuspr as
  # the key, and we would never gc the cache entry. This works because
  # Olympuspr is hashed by id, and the cache entry is dead is the key is dead.
  return ClosedOlympuspr(olympuspr.replace(), ())

def move_invars_right(olympuspr: ClosedOlympuspr, to_move: Sequence[bool]):
  return _move_invars_right(olympuspr, tuple(to_move))

@weakref_lru_cache
def _move_invars_right(olympuspr: ClosedOlympuspr, to_move: tuple[bool, ...]):
  invars, rest = split_list(olympuspr.olympuspr.invars, [len(to_move)])
  left_invars, right_invars = partition_list(to_move, invars)
  new_invars = [*left_invars, *right_invars, *rest]
  new_effs = _renumber_effects(
      (*olympuspr.olympuspr.constvars, *new_invars),
      (*olympuspr.olympuspr.constvars, *olympuspr.olympuspr.invars),
      olympuspr.olympuspr.effects)
  new_olympuspr = olympuspr.olympuspr.replace(invars=new_invars, effects=new_effs)
  return olympuspr.replace(olympuspr=new_olympuspr)

def move_binders_to_front(closed_olympuspr: ClosedOlympuspr, to_move: Sequence[bool]
                          ) -> ClosedOlympuspr:
  """Reorder `invars` by moving those indicated in `to_move` to the front."""
  return _move_binders_to_front(closed_olympuspr, tuple(to_move))

@weakref_lru_cache
def _move_binders_to_front(olympuspr: ClosedOlympuspr, to_move: tuple[bool, ...]
                           ) -> ClosedOlympuspr:
  assert len(olympuspr.in_avals) == len(to_move)
  constvars, invars = olympuspr.olympuspr.constvars, olympuspr.olympuspr.invars
  new_invars = _move_to_front(invars, to_move)
  new_effs = _renumber_effects(
      (*constvars, *new_invars), (*constvars, *invars), olympuspr.olympuspr.effects)
  if olympuspr.olympuspr.debug_info.arg_names is None:
    new_arg_names = None
  else:
    new_arg_names = tuple(_move_to_front(olympuspr.olympuspr.debug_info.arg_names, to_move))
  dbg = olympuspr.olympuspr.debug_info._replace(arg_names=new_arg_names)
  new_olympuspr = olympuspr.olympuspr.replace(
      constvars=constvars, invars=new_invars, effects=new_effs, debug_info=dbg)
  return core.ClosedOlympuspr(new_olympuspr, olympuspr.consts)

def _renumber_effects(new_vars, old_vars, effs):
  newvar_idxs = {id(v): i for i, v in enumerate(new_vars)}
  old_to_new = {i: newvar_idxs[id(v)] for i, v in enumerate(old_vars)}
  return {e.replace(input_index=old_to_new[e.input_index])
          if isinstance(e, effects.OlympusprInputEffect) else e for e in effs}

def _move_to_front(lst: Sequence, to_move: Sequence[bool]) -> Sequence:
  return ([elt for elt, move in zip(lst, to_move) if move] +
          [elt for elt, move in zip(lst, to_move) if not move])

def move_binders_to_back(closed_olympuspr: ClosedOlympuspr, to_move: Sequence[bool]
                         ) -> ClosedOlympuspr:
  """Reorder `invars` by moving those indicated in `to_move` to the back."""
  return move_binders_to_front(closed_olympuspr, map(op.not_, to_move))

def move_outvars_to_back(olympuspr: ClosedOlympuspr, to_move: Sequence[bool]) -> ClosedOlympuspr:
  return _move_outvars_to_back(olympuspr, tuple(to_move))

@weakref_lru_cache
def _move_outvars_to_back(olympuspr: core.ClosedOlympuspr, to_move):
  new_outvars = ([e for e, m in zip(olympuspr.olympuspr.outvars, to_move) if not m] +
                 [e for e, m in zip(olympuspr.olympuspr.outvars, to_move) if     m])
  return olympuspr.replace(olympuspr=olympuspr.olympuspr.replace(outvars=new_outvars))


class DynamicOlympusprTracer(core.Tracer):
  __slots__ = ['aval', 'val', 'mutable_qdd', 'parent', '_debug_info']

  def __init__(self, trace: DynamicOlympusprTrace,
               aval: core.AbstractValue | core.AvalQDD,
               val : Atom,
               line_info: source_info_util.SourceInfo | None = None,
               parent : TracingEqn | None = None):
    # TODO(dougalm): Remove aval. It's redundant now that we have val.
    if isinstance(aval, core.AvalQDD):
      assert aval.qdd is not None
      aval, qdd = aval.aval, aval.qdd
    else:
      assert not aval.has_qdd
      qdd = None
    self._trace = trace
    self._line_info = line_info
    self._debug_info = self._trace.frame.debug_info  # for UnexpectedTracerError
    self.aval = aval  # type: ignore[misc]
    self.val = val
    self.mutable_qdd = core.MutableQuasiDynamicData(qdd)
    self.parent = parent

  def _short_repr(self):
    return f"JitTracer({self.aval})"

  def cur_qdd(self):
    return self.mutable_qdd.cur_val

  @property
  def aval_mutable_qdd(self):
    aval = self.aval
    if aval.has_qdd:
      return core.AvalMutableQDD(aval, self.mutable_qdd)
    else:
      return aval

  def full_lower(self):
    atom = self.val
    if isinstance(atom, Literal):
      return self.val.val
    else:
      maybe_const = self._trace.frame.constvar_to_val.get(atom)
      if maybe_const is None:
        return self
      else:
        return core.full_lower(maybe_const.canonical)

  def _contents(self):
    return ()

  def _origin_msg(self):
    invar_pos, progenitor_eqns = self._trace.frame.find_progenitors(self)
    dbg = self._debug_info
    if dbg is None:
      return ""

    origin = ("The error occurred while tracing the function "
              f"{dbg.func_src_info} for {dbg.traced_for}. ")
    if invar_pos:
      try:
        arg_names = [(dbg.arg_names[i] if dbg.arg_names is not None else "unknown")
                     for i in invar_pos]
      except IndexError:
        return ""  # TODO(mattjj): figure out when not (invar_pos < len(arg_info))
      if len(arg_names) == 1:
        arg_info_str = f"the argument {arg_names[0]}"
      elif len(arg_names) == 2:
        arg_info_str = f"the arguments {arg_names[0]} and {arg_names[1]}"
      else:
        *rest, last = arg_names
        arg_info_str = f"the arguments {', '.join(rest)}, and {last}"
      origin += ("This concrete value was not available in Python because it "
                 f"depends on the value{'s' if len(invar_pos) > 1 else ''} "
                 f"of {arg_info_str}.")
    elif progenitor_eqns:
      msts = ["  operation "
              f"{core.pp_eqn(eqn, core.OlympusprPpContext(), core.OlympusprPpSettings(print_shapes=True))}\n"
              f"    from line {source_info_util.summarize(eqn.source_info)}"
              for eqn in progenitor_eqns[:5]]  # show at most 5
      origin += ("This value became a tracer due to OLYMPUS operations on these lines:"
                 "\n\n" + "\n\n".join(msts))
      if len(progenitor_eqns) > 5:
        origin += "\n\n(Additional originating lines are not shown.)"
    return "\n" + origin

  def get_const(self):
    return self._trace.get_const(self)

  def get_referent(self):
    frame = self._trace.frame
    atom = self.val
    val = frame.constvar_to_val.get(atom) if isinstance(atom, Var) else None
    return self if val is None else get_referent(val.canonical)

core.pytype_aval_mappings[DynamicOlympusprTracer] = lambda x: x.aval

def make_olympuspr_effects(constvars, invars, outvars, eqns) -> effects.Effects:
  sentinel = object()
  olympuspr_effects = set()
  all_vars = {v: i for i, v in enumerate(it.chain(constvars, invars))}
  mut_arrays = set()
  for eqn in eqns:
    if eqn.primitive in core._ref_allocating_primitives:
      outvar, = eqn.outvars
      all_vars[outvar] = None  # type: ignore
      mut_arrays.add(outvar)
    for eff in eqn.effects:
      if isinstance(eff, effects.OlympusprInputEffect):
        if eff.input_index >= len(eqn.invars):
          # TODO(mattjj): ask for forgiveness
          dbg = type('Fake', (), {'resolve_result_paths': lambda self_: self_,
                                  'assert_arg_names': lambda _, __: None,
                                  'assert_result_paths': lambda _, __: None,
                                  })()
          raise ValueError(
              f"`OlympusprInputEffect` {eff} is invalid."
              f"\n Equation: {eqn}\n"
              "\n Olympuspr: "
              f"{core.Olympuspr(constvars, invars, outvars, eqns, set(), dbg)}")  # type: ignore
        eqn_invar = eqn.invars[eff.input_index]
        if type(eqn_invar) is core.Literal or eqn_invar in mut_arrays:
          continue
        if (input_index := all_vars.get(eqn_invar, sentinel)) is sentinel:
          # TODO(mattjj): ask for forgiveness
          dbg = type('Fake', (), {'resolve_result_paths': lambda self_: self_,
                                  'assert_arg_names': lambda _, __: None,
                                  'assert_result_paths': lambda _, __: None,
                                  })()
          raise ValueError(
                f"`OlympusprInputEffect` {eff} does not have "
                f"corresponding olympuspr input: {eqn_invar=}."
                f"\n Equation: {eqn}\n"
                f"\n Effects: {eqn.effects}\n"
                "\n Olympuspr: "
                f"{core.Olympuspr(constvars, invars, outvars, eqns, set(), dbg)}")  # type: ignore
        eff = eff.replace(input_index=input_index)
      olympuspr_effects.add(eff)
  return olympuspr_effects

class Constants(NamedTuple):
  # A pair of a canonicalized constant and its original form.
  # It is important that we keep the original value alive because we use id(c)
  # as a key in various dictionaries. If the original value were deleted we
  # may confuse constants if the same object ID is reused.
  canonical: Any
  original: Any


class OlympusprStackFrame:
  gensym: Callable[[AbstractValue], Var]
  constid_to_tracer: WeakValueDictionary[ConstId, DynamicOlympusprTracer]
  constvar_to_val: dict[Var, Constants]
  tracing_eqns: list[Union[ReferenceType[TracingEqn], Callable[[], TracingEqn]]]
  invars: list[Var]
  effects: core.Effects
  debug_info: core.DebugInfo
  is_high: bool
  mutable_qdds: list[tuple[Var, core.MutableQuasiDynamicData]]
  auto_dce: bool

  def __init__(self, debug_info: core.DebugInfo, auto_dce: bool):
    self.gensym = core.gensym()
    self.constid_to_tracer = WeakValueDictionary()
    self.constvar_to_val = {}
    self.tracing_eqns = []      # cleared when we pop frame from main
    self.invars = []
    self.effects = set()
    self.debug_info = debug_info
    self.is_high = False
    self.mutable_qdds = []
    self.auto_dce = auto_dce

  def add_eqn(self, eqn: core.TracingEqn):
    assert isinstance(eqn, TracingEqn)
    r = (lambda: eqn) if (eqn.effects or not self.auto_dce) else ref(eqn)
    self.tracing_eqns.append(r)

  def get_eqns(self):
    eqns = []
    for tracing_eqn in self.tracing_eqns:
      e = tracing_eqn()
      if e is None: continue
      eqns.append(OlympusprEqn(
          [t.val for t in e.in_tracers],
          e.outvars, e.primitive, e.params, e.effects, e.source_info, e.ctx))
    return eqns

  def to_olympuspr(
      self, trace: DynamicOlympusprTrace,
      out_tracers: Sequence[Tracer],
      debug_info: core.DebugInfo,
      source_info: SourceInfo,
    ) -> tuple[Olympuspr, list[Any]]:
    eqns = self.get_eqns()
    outvars = [t.val for t in out_tracers]
    constvars, constvals = unzip2(self.constvar_to_val.copy().items())
    constvals = [c.canonical for c in constvals]
    constvars, constvals = _drop_unused_vars(constvars, constvals, eqns, outvars)
    effs = make_olympuspr_effects(constvars, self.invars, outvars, eqns)

    # TODO(dougalm): handle qdd for consts
    for v, qdd in self.mutable_qdds:
      v.final_qdd = qdd.cur_val

    all_vars = it.chain(constvars, self.invars, outvars)
    is_high = self.is_high or any(v.aval.is_high for v in all_vars)

    olympuspr = Olympuspr(constvars, self.invars, outvars, eqns, effs, debug_info, is_high)
    return olympuspr, list(constvals)

  def newvar(self, aval):
    if isinstance(aval, core.AvalQDD):
       return self.gensym(aval.aval, initial_qdd=aval.qdd)
    else:
       return self.gensym(aval)

  def find_progenitors(self, tracer):
    eqns = self.get_eqns()
    var = tracer.val
    if not var or isinstance(var, Literal):
      return None, None
    active_vars = {var}
    for eqn in eqns[::-1]:
      produced = set(eqn.outvars) & active_vars
      if produced:
        active_vars.difference_update(produced)
        active_vars.update({v for v in eqn.invars if type(v) is Var})
    invar_positions = [i for i, v in enumerate(self.invars) if v in active_vars]
    constvars = active_vars & set(self.constvar_to_val.copy())
    const_eqns = [eqn for eqn in eqns if any(
        v in constvars if type(v) is Var else type(v) is Literal
        for v in eqn.invars)]
    return invar_positions, const_eqns


ConstFoldRule = Callable[
    [list[Union[Any, None]], Any, list[AbstractValue]],
    tuple[list[Union[Any, None]], Union[OlympusprEqn, None]],
]
const_fold_rules: dict[Primitive, ConstFoldRule] = {}

ForwardingRule = Callable[
    [OlympusprEqn],
    tuple[list[Union[int, None]], Union[OlympusprEqn, None]]
]
forwarding_rules: dict[Primitive, ForwardingRule] = {}


def _drop_unused_vars(constvars, constvals, eqns, outvars
                      ) -> tuple[list[Var], list[Any]]:
  # modifies eqns in-place!
  def vars(atom: Atom) -> list[Var]:
    if isinstance(atom, Literal):
      return []
    aval = atom.aval
    return [atom]
  used: set[Var] = {v for atom in outvars for v in vars(atom)}
  for eqn in eqns[::-1]:
    eqn.outvars = [v if v in used else DropVar(v.aval) for v in eqn.outvars]
    used.update(v for atom in eqn.invars for v in vars(atom))
  constvars, constvals = unzip2(
      (v, val) for v, val in zip(constvars, constvals) if v in used)
  return constvars, constvals


@multi_weakref_lru_cache
def _cached_abstract_eval(primitive: core.Primitive, *aval_qdds, **params):
  return primitive.abstract_eval(*aval_qdds, **params)


def _verify_params_are_hashable(
    primitive: core.Primitive, params: dict[str, Any]) -> None:
  for k, v in params.items():
    try:
      hash(v)
    except TypeError as e:
      raise TypeError(
        "As of OLYMPUS v0.7, parameters to olympuspr equations must have __hash__ and "
        f"__eq__ methods. In a call to primitive {primitive}, the value of "
        f"parameter {k} was not hashable: {v}") from e

# We use TracingEqn instead OlympusprEqn during tracing to allow automatic
# on-the-fly DCE based on Python refcounting. DynamicOlympusprTracers point to
# TracingEqns which point to DynamicOlympusprTracers and unreachable constants can
# be freed.

@dataclass
class TracingEqn:
  in_tracers: list[DynamicOlympusprTracer]
  outvars: list[Var]
  primitive: Primitive
  params: dict[str, Any]
  effects: core.Effects
  source_info: source_info_util.SourceInfo
  ctx: OlympusprEqnContext

  def __init__(self, in_tracers, outvars, primitive, params, effects, source_info, ctx):
    self.in_tracers = in_tracers
    self.outvars = outvars
    self.primitive = primitive
    self.params = params
    self.effects = effects
    self.source_info = source_info
    self.ctx = ctx

  # Allow TracingEqn to duck-type OlympuspeEqn because some of the forwarding
  # rules need to work with both. TODO(dougalm): remove this once we fix
  # forwarding.
  @property
  def invars(self):
    return self.in_tracers

class DynamicOlympusprTrace(core.Trace):
  __slots__ = ("frame", "tag", "parent_trace")

  def __init__(self, debug_info: core.DebugInfo, parent_trace=None, lower=False,
               auto_dce=False):
    super().__init__()
    self.requires_low = lower
    self.frame = OlympusprStackFrame(debug_info, auto_dce)
    self.parent_trace = parent_trace

  def invalidate(self):
    # TODO(mattjj): exposed existing tracer leaks; fix them and re-enable!
    # super().invalidate()

    # avoid cyclic refs
    self.frame.tracing_eqns = []  # thunk -> eqn -> in_tracers -> trace ->
                                  # -> frame -> tracing_eqns -> thunk

    # TODO(dougalm): we might be able to remove these given refcounting dce
    self.frame.constid_to_tracer = {}
    self.frame.constvar_to_val = {}

  def to_olympuspr_tracer(self, x, source_info: SourceInfo):
    if isinstance(x, DynamicOlympusprTracer) and x._trace is self:
      return x
    else:
      if hasattr(x, "dimension_as_value"):  # Used for shape_poly._DimExpr
        with core.set_current_trace(self):
          x = x.dimension_as_value()
        return self.to_olympuspr_tracer(x, source_info)
      else:
        return self.new_const(x, source_info)

  def var_to_tracer(self, var, source_info, parent=None):
    aval = var.aval
    if aval.has_qdd:
      aval = core.AvalQDD(aval, var.initial_qdd)
    return DynamicOlympusprTracer(self, aval, var, source_info, parent)

  def new_arg(self, aval, source_info: SourceInfo):
    var = self.frame.newvar(aval)
    tracer = DynamicOlympusprTracer(self, aval, var, source_info)
    self.frame.invars.append(var)
    self.frame.mutable_qdds.append((var, tracer.mutable_qdd))
    return tracer

  def make_eqn(self, in_tracers, out_avals, primitive, params,
               effects, source_info=None, ctx = None):
    source_info = source_info or source_info_util.new_source_info()
    ctx = ctx or OlympusprEqnContext(
        config.compute_on_context_manager.value,
        config.threefry_partitionable.value,
        xla_metadata_lib.current_xla_metadata())
    outvars = map(self.frame.newvar, out_avals)
    if config.enable_checks.value:
      assert all(isinstance(x, DynamicOlympusprTracer) for x in in_tracers)
      assert all(isinstance(v,  Var)               for v in outvars)
    eqn = TracingEqn(in_tracers, outvars, primitive, params, effects, source_info, ctx)
    out_tracers = [self.var_to_tracer(v, source_info, eqn) for v in outvars]
    return eqn, out_tracers

  def emit_eqn(self, in_tracers, out_avals, primitive, params, effects, source_info=None, ctx=None):
    eqn, out_tracers = self.make_eqn(in_tracers, out_avals, primitive, params, effects, source_info, ctx)
    self.frame.add_eqn(eqn)
    return out_tracers

  def new_const(self, c, source_info: SourceInfo,
                aval: AbstractValue | None = None):
    # TODO(mattjj): for ints, or hashable consts, don't rely on id
    tracer = self.frame.constid_to_tracer.get(id(c))
    if tracer is None:
      if aval is None:
        aval = get_aval(c)
      if aval.has_qdd:
        with core.set_current_trace(self.parent_trace or core.eval_trace):
          aval = core.AvalQDD(aval, core.cur_qdd(c))  # type: ignore
      tracer = self._new_const(aval, c, source_info)
    return tracer

  pure = lift = new_const

  def _new_const(self, aval, c, source_info: SourceInfo) -> DynamicOlympusprTracer:
    orig_c = c
    id_c = id(c)
    if isinstance(c, (int, float, bool, complex, np.generic, np.ndarray)):
      c = dtypes.canonicalize_value(c)
    if core.is_literalable(c):
      val = Literal(c, aval)
      return DynamicOlympusprTracer(self, aval, val, source_info)
    else:
      var = self.frame.newvar(aval)
      tracer = DynamicOlympusprTracer(self, aval, var, source_info)
      self.frame.constid_to_tracer[id_c] = tracer
      if isinstance(aval, core.AvalQDD):
        self.frame.mutable_qdds.append((var, tracer.mutable_qdd))
      self.frame.constvar_to_val[var] = Constants(canonical=c, original=orig_c)
      finalize(tracer, self.finalize_const, var, id_c)
      return tracer

  def finalize_const(self, var, constid):
    self.frame.constvar_to_val.pop(var, None)

  def get_const(self, tracer) -> Any:
    atom = tracer.val
    if isinstance(atom, Literal):
      return atom.val
    else:
      const = self.frame.constvar_to_val.get(atom)
      if const is not None:
        const = const.canonical
      return const

  def cur_qdd(self, x):
    source_info = source_info_util.current()
    return self.to_olympuspr_tracer(x, source_info=source_info).mutable_qdd.cur_val

  def process_primitive(self, primitive, tracers, params):
    self.frame.is_high |= primitive.is_high(*map(typeof, tracers), **params)
    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      return primitive.bind_with_trace(core.eval_trace, tracers, params)
    source_info = source_info_util.current()
    to_olympuspr_tracer = partial(self.to_olympuspr_tracer, source_info=source_info)
    olympuspr_tracers = map(to_olympuspr_tracer, tracers)
    if primitive in custom_staging_rules:
      return custom_staging_rules[primitive](self, source_info, *olympuspr_tracers,
                                             **params)
    return self.default_process_primitive(
        primitive, olympuspr_tracers, params, source_info)

  def default_process_primitive(self, primitive, tracers, params,
                                source_info=None):
    from olympus._src.hiolympus import call_hi_primitive_p
    aval_qdds = [t.aval_mutable_qdd for t in tracers]
    # TODO(mattjj): make custom_lin have hashable params.
    # TODO(dougalm): add an attribute to primitives to mark primitives with
    # effectful abstract_eval rules.
    # TODO(mattjj,dougalm): clean up how we check for new-style hi primitives
    if primitive is call_hi_primitive_p:
      out_avals, effs = params['prim'].out_avals_flat, set()  # TODO effs
    elif (primitive.name in ("custom_lin", "call_hi_primitive_linearized") or
          primitive.is_effectful and primitive.is_effectful(params)):
      out_avals, effs = primitive.abstract_eval(*aval_qdds, **params)
    else:
      try:
        out_avals, effs = _cached_abstract_eval(primitive, *aval_qdds, **params)
      except Exception as e:
        # TODO(phawkins): remove this 3 months after the release of OLYMPUS v0.7.
        _verify_params_are_hashable(primitive, params)
        raise

    if isinstance(out_avals, (tuple, list)) != primitive.multiple_results:
      raise ValueError(f"{primitive}.abstract_eval() method should return "
                       f"a tuple or a list iff {primitive}.multiple_results.")
    out_avals = [out_avals] if not primitive.multiple_results else out_avals
    source_info = source_info or source_info_util.current()

    maybe_consts_out = try_constant_folding(primitive, tracers, params, out_avals)
    if maybe_consts_out is not None:
      eqn = None
      out_tracers = [self.new_const(c, source_info=source_info, aval=aval)
                     for c, aval in zip(maybe_consts_out, out_avals)]
    else:
      eqn, out_tracers = self.make_eqn(tracers, out_avals, primitive, params,
                                       effs, source_info=source_info)
    # Input-to-output tracer forwarding
    no_input_effects = not any(isinstance(e, effects.OlympusprInputEffect) for e in effs)
    if eqn is not None and no_input_effects and primitive in forwarding_rules:
      in_fwd, eqn = forwarding_rules[primitive](eqn)
      for out_idx, in_idx in enumerate(in_fwd):
        if in_idx is not None:
          out_tracers[out_idx] = tracers[in_idx]

    if eqn is not None:
      self.frame.add_eqn(eqn)
    return out_tracers if primitive.multiple_results else out_tracers.pop()

  def process_call(self, call_primitive, f: lu.WrappedFun, in_tracers,
                   params):
    source_info = source_info_util.current()
    to_olympuspr_tracer = partial(self.to_olympuspr_tracer, source_info=source_info)
    in_type = (tuple(get_aval(t) for t in in_tracers) if f.in_type is None
               else f.in_type)
    f.in_type = None
    assert in_type is not None
    in_tracers = map(to_olympuspr_tracer, in_tracers)
    # TODO(mattjj): check in_tracers are consistent with f.in_type annotation
    olympuspr, out_avals, consts = _cached_trace_to_olympuspr(f, in_type)
    if params.get('inline', False):
      return core.eval_olympuspr(olympuspr, consts, *in_tracers,
                             propagate_source_info=False)

    new_olympuspr = convert_constvars_olympuspr(olympuspr)
    if isinstance(call_primitive, core.ClosedCallPrimitive):
      new_olympuspr = close_olympuspr(new_olympuspr)  # type: ignore
    new_params = dict(params, call_olympuspr=new_olympuspr)
    update_params = call_param_updaters.get(call_primitive)
    if update_params:
      new_params = update_params(new_params, [True] * len(in_tracers),
                                 len(consts))
    const_tracers = map(to_olympuspr_tracer, consts)
    return self.emit_eqn(
        [*const_tracers, *in_tracers], out_avals, call_primitive,
        new_params, new_params['call_olympuspr'].effects, source_info=source_info)

  def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params):
    source_info = source_info_util.current()
    to_olympuspr_tracer = partial(self.to_olympuspr_tracer, source_info=source_info)
    tracers = map(to_olympuspr_tracer, tracers)
    in_avals = [t.aval for t in tracers]
    axis_name, axis_size = params['axis_name'], params['axis_size']
    reduced_in_avals = [core.mapped_aval(axis_size, in_axis, a)
                        if in_axis is not None else a
                        for a, in_axis in zip(in_avals, params['in_axes'])]
    with core.extend_axis_env_nd([(axis_name, params["global_axis_size"])]):
      olympuspr, reduced_out_avals, consts = trace_to_olympuspr_dynamic(
          f.with_unknown_names(), reduced_in_avals)
      olympuspr, consts = _linearize_of_pmap_hack(f, olympuspr, consts)
      ordered_effects = effects.ordered_effects.filter_in(olympuspr.effects)
      if ordered_effects:
        raise ValueError("Ordered effects not supported for "
                         f"map primitives: {ordered_effects}")
      out_axes = params['out_axes_thunk']()
      out_avals = [core.unmapped_aval(axis_size, out_axis, a)
                  if out_axis is not None else a
                  for a, out_axis in zip(reduced_out_avals, out_axes)]
      const_tracers = map(to_olympuspr_tracer, consts)
      new_in_axes = (None,) * len(consts) + params['in_axes']
      new_params = dict(params, in_axes=new_in_axes, out_axes=out_axes,
                        call_olympuspr=convert_constvars_olympuspr(olympuspr))
      del new_params['out_axes_thunk']
      update_params = call_param_updaters.get(map_primitive)
      if update_params:
        new_params = update_params(new_params, [True] * len(tracers), len(consts))
      effs = core.filter_named_axis_effects(olympuspr.effects, {axis_name})
      out_tracers = self.emit_eqn(
          [*const_tracers, *tracers], out_avals, map_primitive, new_params, effs, source_info=source_info)
    return out_tracers

  def process_custom_jvp_call(self, prim, fun: lu.WrappedFun,
                              jvp: lu.WrappedFun, tracers,
                              symbolic_zeros: bool):
    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      return prim.bind_with_trace(core.eval_trace, (fun, jvp, *tracers),
                                  dict(symbolic_zeros=symbolic_zeros))
    source_info = source_info_util.current()
    to_olympuspr_tracer = partial(self.to_olympuspr_tracer, source_info=source_info)
    tracers = map(to_olympuspr_tracer, tracers)
    in_avals = [t.aval for t in tracers]
    in_tangent_avals = [t.to_tangent_aval() for t in in_avals]
    fun_olympuspr, out_avals, consts = trace_to_olympuspr_dynamic(fun, in_avals)
    closed_fun_olympuspr = core.ClosedOlympuspr(convert_constvars_olympuspr(fun_olympuspr), ())

    @partial(lu.wrap_init, debug_info=jvp.debug_info)
    @_memoize
    def jvp_olympuspr_thunk(*in_zeros):
      for store in jvp.stores: store and store.reset()
      nz_tangent_avals, zero_avals = partition_list(in_zeros, in_tangent_avals)
      jvp_, out_zeros = _jvp_olympuspr_zeros(jvp, in_zeros, tuple(zero_avals))
      in_avals_ = (*in_avals, *nz_tangent_avals)
      olympuspr, _, out_consts = trace_to_olympuspr_dynamic(jvp_.with_unknown_names(),
                                                    in_avals_)
      return olympuspr, out_consts, out_zeros()

    const_tracers = map(to_olympuspr_tracer, consts)
    return self.emit_eqn(
        [*const_tracers, *tracers], out_avals, prim,
        dict(call_olympuspr=closed_fun_olympuspr,
             jvp_olympuspr_fun=jvp_olympuspr_thunk,
             num_consts=len(consts),
             symbolic_zeros=symbolic_zeros),
        fun_olympuspr.effects,
        source_info=source_info)

  def process_custom_vjp_call(self, prim: core.Primitive,
                              fun: lu.WrappedFun,
                              fwd: lu.WrappedFun, bwd: lu.WrappedFun, tracers,
                              out_trees: Callable[[], tuple[PyTreeDef, PyTreeDef, list[int | None]]],
                              symbolic_zeros: bool):
    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      return prim.bind_with_trace(core.eval_trace, (fun, fwd, bwd, *tracers),
                                  dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros))
    source_info = source_info_util.current()
    to_olympuspr_tracer = partial(self.to_olympuspr_tracer, source_info=source_info)
    tracers = map(to_olympuspr_tracer, tracers)
    in_avals = [core.AvalQDD(t.aval, core.cur_qdd(t)) if t.aval.has_qdd else t.aval for t in tracers]
    fun_olympuspr, out_avals, consts = trace_to_olympuspr_dynamic(fun.with_unknown_names(), in_avals)
    num_consts = len(consts)
    closed_fun_olympuspr = core.ClosedOlympuspr(convert_constvars_olympuspr(fun_olympuspr), ())

    @partial(lu.wrap_init, debug_info=fwd.debug_info)
    @_memoize
    def fwd_olympuspr_from_zeros(*zeros):
      for store in fwd.stores: store and store.reset()
      fwd_ = _interleave_fun(fwd.with_unknown_names(), zeros)
      olympuspr, _, consts = trace_to_olympuspr_dynamic(fwd_, in_avals)
      return olympuspr, consts

    def out_trees_():
      out_tree, res_tree, input_fwds = out_trees()
      input_fwds = [f if f is None else f + num_consts for f in input_fwds]
      return out_tree, res_tree, input_fwds

    const_tracers = map(to_olympuspr_tracer, consts)
    return self.emit_eqn(
        [*const_tracers, *tracers], out_avals, prim,
        dict(call_olympuspr=closed_fun_olympuspr,
             fwd_olympuspr_thunk=fwd_olympuspr_from_zeros,
             num_consts=num_consts,
             bwd=bwd, out_trees=out_trees_,
             symbolic_zeros=symbolic_zeros),
        fun_olympuspr.effects,
        source_info=source_info)

  def process_custom_transpose(self, prim: core.Primitive,  # type: ignore[override]
                               call: lu.WrappedFun, tracers, *,
                               transpose: lu.WrappedFun,
                               out_types,
                               lin_tree: PyTreeDef,
                               res_tree: PyTreeDef, out_tree: PyTreeDef):
    source_info = source_info_util.current()
    to_olympuspr_tracer = partial(self.to_olympuspr_tracer, source_info=source_info)
    tracers = map(to_olympuspr_tracer, tracers)
    tracers_res, tracers_lin = split_list(tracers, [res_tree.num_leaves])

    in_avals_p = [t.aval for t in tracers]
    in_avals_t = [*[t.aval for t in tracers_res], *out_types]

    call_olympuspr, out_avals, call_consts = trace_to_olympuspr_dynamic(call, in_avals_p)
    closed_call_olympuspr = core.ClosedOlympuspr(
        convert_constvars_olympuspr(call_olympuspr), ())

    transpose_flat, in_tree2 = api_util.flatten_fun_nokwargs(
        transpose, treedef_tuple((res_tree, out_tree)))

    # the following thunk evaluates to a pair: transpose_olympuspr, transpose_consts
    @_memoize
    def transpose_olympuspr_thunk():
      for store in transpose_flat.stores: store.reset()
      olympuspr, _, consts = trace_to_olympuspr_dynamic(transpose_flat, in_avals_t)
      return olympuspr, consts

    const_tracers = map(to_olympuspr_tracer, call_consts)
    return self.emit_eqn(
        [*const_tracers, *tracers], out_avals, prim,
        dict(call_olympuspr=closed_call_olympuspr,
             transpose_olympuspr_thunk=transpose_olympuspr_thunk,
             out_types=out_types, res_tree=res_tree,
             lin_tree=lin_tree, out_tree=out_tree),
        closed_call_olympuspr.effects,
        source_info=source_info)

  def to_olympuspr(self, out_tracers: Sequence[Tracer],
               debug_info: core.DebugInfo, source_info: SourceInfo):
    return self.frame.to_olympuspr(self, out_tracers, debug_info, source_info)


@lu.cache
def _cached_trace_to_olympuspr(f, in_type):
  olympuspr, out_type, consts = trace_to_olympuspr_dynamic(lu.annotate(f, in_type), in_type)
  return olympuspr, out_type, consts


custom_staging_rules: dict[Primitive, Callable] = {}

@lu.transformation2
def _interleave_fun(f, every_others, *args, **kwargs):
  args_ = [x for pair in zip(args, every_others) for x in pair]
  return f(*args_, **kwargs)

# TODO: consider renaming to "lazy_thunk"
def _memoize(fn):
  cells = {}
  sentinel = object()
  def memoized(*args):
    out = cells.get(args, sentinel)
    if out is sentinel:
      with core.set_current_trace(None):
        out = cells[args] = fn(*args)
    return out
  return memoized

@lu.transformation_with_aux2
def _jvp_olympuspr_zeros(f, store, in_zeros, zero_avals, *primal_tangent_avals):
  in_primals, nz_in_tangents = split_list(primal_tangent_avals, [len(in_zeros)])
  symbolic_zeros = map(ad_util.SymbolicZero, zero_avals)
  tangents = merge_lists(in_zeros, nz_in_tangents, symbolic_zeros)
  out = f(*in_primals, *tangents)
  n, ragged = divmod(len(out), 2)
  assert not ragged
  out_primals, out_tangents = out[:n], out[n:]
  out_zeros = [type(t) is ad_util.SymbolicZero for t in out_tangents]
  out_nz_tangents, _ = partition_list(out_zeros, out_tangents)
  store.store(out_zeros)
  return [*out_primals, *out_nz_tangents]

callsites_with_tracing_cache_miss: set[str] = set()

def explain(keys, fun, in_avals, debug_info, *context):
  func_filename = debug_info.func_filename
  if func_filename and not source_info_util.is_user_filename(func_filename):
   return

  msg: list[str] = []
  p = msg.append

  callsite = source_info_util.summarize(source_info_util.current())
  p(f"TRACING CACHE MISS at {callsite}:")

  src_info = ""
  if func_filename:
    src_info += f" defined at {func_filename}"
  if func_lineno := debug_info.func_lineno:
    src_info += f":{func_lineno}"
  func_name = debug_info.func_name

  # have we seen this function before at all?
  keys = [key for fun_ref, *key in keys if fun_ref() is fun]
  if not keys:
    p(f"  never seen function:\n    {func_name} id={id(fun)}{src_info}")
    if callsite in callsites_with_tracing_cache_miss:
      p("  but seen another function defined on the same line; maybe the function is\n"
        "  being re-defined repeatedly, preventing caching?")
    else:
      callsites_with_tracing_cache_miss.add(callsite)
    return logger.log(logging.WARNING, "\n".join(msg))

  p(f"  for {func_name}{src_info}")

  key = (config.trace_context(), (in_avals, debug_info, *context), {})
  min_diff = min(diff_tracing_cache_keys(key, k) for k in keys)[-1]
  p('  all previously seen cache keys differ. For the closest previous key:')
  p('  ' + min_diff)
  return logger.log(logging.WARNING, "\n".join(msg))

def diff_tracing_cache_keys(new_key, old_key) -> tuple[int, int, str] | None:
  new_ctx, (new_tree, new_dbg, new_qdd, *_), () = new_key
  old_ctx, (old_tree, old_dbg, old_qdd, *_), () = old_key
  return (diff_ctx(new_ctx, old_ctx) or
          diff_trees(new_tree.tree, old_tree.tree) or
          diff_debug(new_dbg, old_dbg) or
          diff_types(new_dbg, new_tree.vals, old_tree.vals) or
          (4, 0, 'cache miss explanation unavailable'))

def diff_ctx(new_ctx, old_ctx):
  msg = "Tracing context doesn't match, e.g. due to config or context manager."
  num_diff = sum(map(op.ne, new_ctx, old_ctx))
  if num_diff: return 0, num_diff, msg

def diff_trees(new_tree, old_tree):
  errs = tree_util.equality_errors_pytreedef(new_tree, old_tree)
  tree_diffs = []
  for path, thing1, thing2, explanation in errs:
    tree_diffs.append(
        f"  * at input path {tree_util.keystr(tuple(path))}, now {thing1} and "
        f"before {thing2}, so {explanation}")
  msg = 'different input pytree:\n' + '\n'.join(tree_diffs)
  if tree_diffs: return 1, len(tree_diffs), msg

def diff_debug(new_dbg, old_dbg):
  msg = "Debug info doesn't match."
  num_diff = sum(map(op.ne, new_dbg, old_dbg))
  if num_diff: return 2, num_diff, msg

def diff_types(dbg, new_leaves, old_leaves):
  if new_leaves == old_leaves: return
  diffs = []
  add_weak_type_hint = False
  for name, new_ty, old_ty in zip(dbg.arg_names, new_leaves, old_leaves):
    if new_ty != old_ty:
      new_str, old_str = new_ty.str_short(True), old_ty.str_short(True)
      if type(new_ty) is type(old_ty) is core.ShapedArray:
        if new_ty.sharding != old_ty.sharding:
          new_str, old_str = new_ty.str_short(True, True), old_ty.str_short(True, True)
        if new_ty.weak_type != old_ty.weak_type:
          add_weak_type_hint = True
          new_str += f'{{weak_type={new_ty.weak_type}}}'
          old_str += f'{{weak_type={old_ty.weak_type}}}'
      diffs.append(f"  * at {name}, now {new_str} and before {old_str}")
  msg = 'different input types:\n' + '\n'.join(diffs)
  if add_weak_type_hint:
    msg += 'https://docs.olympus.dev/en/latest/type_promotion.html#weak-types'
  if diffs: return 3, len(diffs), msg


@weakref_lru_cache(maxsize=None, explain=explain)
def trace_to_olympuspr(
    fun: Callable,
    in_avals: FlatTree,  # (args, kwargs) pair
    debug_info: core.DebugInfo,
    *context_for_cache_key,
) -> tuple[ClosedOlympuspr, FlatTree]:
  if config.no_tracing.value:
    raise RuntimeError(f"re-tracing function {fun} for "
                       "`jit`, but 'no_tracing' is set")
  del context_for_cache_key  # read implicitly, e.g. qdd state
  test_event("trace_to_olympuspr")
  config.enable_checks.value and debug_info.assert_arg_names(len(in_avals))
  parent_trace = core.trace_ctx.trace
  trace = DynamicOlympusprTrace(debug_info, parent_trace=parent_trace)
  # Name stacks are reset because the name stacks on olympuspr equations should be
  # rooted at the enclosing olympuspr.
  with core.ensure_no_leaks(trace), source_info_util.reset_name_stack():
    source_info = source_info_util.current()
    in_tracers = in_avals.map(partial(trace.new_arg, source_info=source_info))
    with core.set_current_trace(trace):
      args, kwargs = in_tracers.unflatten()
      ans_pytree = fun(*args, **kwargs)
      debug_info = debug_info.set_result_paths(ans_pytree)
      ans = FlatTree.flatten(ans_pytree)
      del ans_pytree, args, kwargs

    _check_returned_olympustypes(debug_info, list(ans))
    out_tracers = ans.map(partial(trace.to_olympuspr_tracer, source_info=source_info))
    out_avals = out_tracers.map(lambda t: t.aval)
    _check_no_returned_refs(debug_info, list(out_tracers))
    olympuspr, consts = trace.frame.to_olympuspr(trace, list(out_tracers), debug_info,
                                         source_info)
    del trace, fun, in_tracers, out_tracers, ans

  config.enable_checks.value and core.check_olympuspr(olympuspr)
  return ClosedOlympuspr(olympuspr, consts), out_avals

# TODO(dougalm): remove in favor of `trace_to_olympuspr`
@profiler.annotate_function
def trace_to_olympuspr_dynamic(
    fun: lu.WrappedFun,
    in_avals: Sequence[AbstractValue | core.AvalQDD],
    *,
    keep_inputs: list[bool] | None = None,
    lower: bool = False,
    auto_dce: bool = False,
) -> tuple[Olympuspr, list[AbstractValue], list[Any]]:
  config.enable_checks.value and fun.debug_info.assert_arg_names(len(in_avals))
  keep_inputs = [True] * len(in_avals) if keep_inputs is None else keep_inputs
  parent_trace = core.trace_ctx.trace
  trace = DynamicOlympusprTrace(fun.debug_info, parent_trace=parent_trace,
                            lower=lower, auto_dce=auto_dce)
  # Name stacks are reset because the name stacks on olympuspr equations should be
  # rooted at the enclosing olympuspr.
  with core.ensure_no_leaks(trace), source_info_util.reset_name_stack():
    source_info = source_info_util.current()
    in_tracers = map(partial(trace.new_arg, source_info=source_info), in_avals)
    in_tracers = [t for t, keep in zip(in_tracers, keep_inputs) if keep]

    with core.set_current_trace(trace):
      ans = fun.call_wrapped(*in_tracers)
    _check_returned_olympustypes(fun.debug_info, ans)
    out_tracers = map(partial(trace.to_olympuspr_tracer, source_info=source_info), ans)
    _check_no_returned_refs(fun.debug_info, out_tracers)
    olympuspr, consts = trace.frame.to_olympuspr(trace, out_tracers, fun.debug_info,
                                         source_info)
    del trace, fun, in_tracers, out_tracers, ans

  config.enable_checks.value and core.check_olympuspr(olympuspr)
  return olympuspr, [v.aval for v in olympuspr.outvars], consts

def _check_returned_olympustypes(dbg, out_tracers):
  for i, x in enumerate(out_tracers):
    try: typeof(x)
    except TypeError:
      if (dbg and len(paths := dbg.resolve_result_paths()) > i and
          (p := paths[i].removeprefix('result'))):
        extra = f' at output component {p}'
      else:
        extra = ''
      raise TypeError(
      f"function {dbg.func_src_info} traced for {dbg.traced_for} returned a "
      f"value of type {type(x)}{extra}, which is not a valid OLYMPUS type") from None

def _check_no_returned_refs(
    dbg: core.DebugInfo,
    out_tracers: Sequence[DynamicOlympusprTracer]
) -> None:
  if not config.mutable_array_checks.value: return
  for i, t in enumerate(out_tracers):
    a = t.aval
    if isinstance(a, AbstractRef):
      result_paths = dbg.resolve_result_paths().safe_result_paths(len(out_tracers))
      loc = result_paths[i] and f' at output tree path {result_paths[i]}'
      frame = t._trace.frame
      v = t.val
      eqns = frame.get_eqns()
      # TODO(dougalm): something more efficient
      eqn = next((e for e in eqns if v in e.outvars), None)
      if eqn:
        assert eqn.primitive is core.ref_p
        origin_info = ('\n\nThe returned mutable array was created on line '
                       f'{source_info_util.summarize(eqn.source_info)}.')
      elif v in frame.invars:
        arg_name = dbg.safe_arg_names(len(frame.invars))[frame.invars.index(v)]
        origin_info = ('\n\nThe returned mutable array was passed in as the '
                       f'argument {arg_name}.')
      else:
        origin_info = ''
      raise ValueError(
          f"function {dbg.func_src_info} traced for {dbg.traced_for} returned "
          f"a mutable array reference of type {a.str_short()}{loc}, but "
          f"mutable array references cannot be returned.{origin_info}")

class TracerAsName:
  ref: Any
  def __init__(self, tracer):
    self.ref = core.get_referent(tracer)
  def __eq__(self, other):
    return isinstance(other, TracerAsName) and self.ref is other.ref
  def __hash__(self):
    return id(self.ref)

Const = Any
Val = Any

def instantiate_const_at(trace: OlympusprTrace, instantiate: bool, tracer):
  if instantiate:
    return trace.instantiate_const(tracer)
  else:
    return tracer

def inline_olympuspr_into_trace(
    trace: DynamicOlympusprTrace, src: SourceInfo, olympuspr: Olympuspr,
    consts: Sequence[Any], *arg_tracers: DynamicOlympusprTracer) -> list[Any]:
  # This function is conceptually the same thing as just calling eval_olympuspr,
  const_tracers = map(partial(trace.new_const, source_info=src), consts)
  env: dict[Var, DynamicOlympusprTracer] = dict(
      zip([*olympuspr.constvars, *olympuspr.invars],
          [*const_tracers, *arg_tracers]))

  def inline_atom(src_, x):
    if isinstance(x, Literal):
      return DynamicOlympusprTracer(trace, x.aval, x, src_)
    else:
      return env[x]

  for eqn in olympuspr.eqns:
    src_ = (src if not eqn.source_info.name_stack else
            src.replace(name_stack=src.name_stack + eqn.source_info.name_stack))
    in_tracers = map(partial(inline_atom, src_), eqn.invars)
    out_avals = [v.aval for v in eqn.outvars]

    maybe_consts = try_constant_folding(eqn.primitive, in_tracers, eqn.params, out_avals)
    if maybe_consts is not None:
      out_tracers = [trace.new_const(c, source_info=src_, aval=aval)
                     for c, aval in zip(maybe_consts, out_avals)]
    else:
      out_tracers = trace.emit_eqn(in_tracers, out_avals, eqn.primitive,
                                   eqn.params, eqn.effects, src_, eqn.ctx)
    foreach(env.setdefault, eqn.outvars, out_tracers)

  return map(partial(inline_atom, src), olympuspr.outvars)


def try_constant_folding(primitive, tracers, params, out_avals):
  if primitive in const_fold_rules:
    consts_in = [t.get_const() for t in tracers]
    if any(c is not None for c in consts_in):
      return const_fold_rules[primitive](consts_in, params, out_avals)
  return None

# TODO(mattjj,dougalm): this special handling is to avoid round-tripping the
# olympuspr when we do grad-of-pmap. The tag is set by LinearizeTrace.process_call's
# handling of pmap. Remove when we replace the pmap implementation.
def _linearize_of_pmap_hack(f: lu.WrappedFun, olympuspr, consts) -> tuple[Olympuspr, list]:
  if (not f.transforms and type(f.f) is HashableFunction and
      getattr(f.f, '_pmap_tag', None)):
    _, olympuspr = f.f.closure
    return convert_constvars_olympuspr(olympuspr), []
  return olympuspr, consts


@weakref_lru_cache
def lower_olympuspr(hi_olympuspr: core.ClosedOlympuspr):
  lo_avals = [lo_ty for aval in hi_olympuspr.in_aval_qdds for lo_ty in aval.lo_ty()]
  f = lu.wrap_init(partial(lower_traceable, hi_olympuspr),
                   debug_info=hi_olympuspr.olympuspr.debug_info.with_unknown_names())
  lo_olympuspr, _, lo_consts = trace_to_olympuspr_dynamic(f, lo_avals, lower=True)
  return core.ClosedOlympuspr(lo_olympuspr, lo_consts)

def lower_traceable(olympuspr, *lo_args):
  lo_args_ = iter(lo_args)
  hi_args = [aval.raise_val(*it.islice(lo_args_, len(aval.lo_ty())))
             if not aval.has_qdd else
             aval.new_from_loval(*it.islice(lo_args_, len(aval.lo_ty())))
             for aval in olympuspr.in_aval_qdds]
  assert (problem := next(lo_args_, None)) is None
  hi_outs = core.olympuspr_as_fun(olympuspr)(*hi_args)
  mut_outs = [lo_val for aval, hi_arg in zip(olympuspr.final_aval_qdds, hi_args) if aval.has_qdd
              for lo_val in aval.read_loval(hi_arg)]
  lo_outs = [lo_val for v, hi_val in zip(olympuspr.olympuspr.outvars, hi_outs)
             for lo_val in v.aval.lower_val(hi_val)]
  return mut_outs + lo_outs

@weakref_lru_cache
def convert_const_himutables(olympuspr):
  move = [typeof(c).has_qdd for c in olympuspr.consts]
  constvals, in_mutables = partition_list(move, olympuspr.consts)
  constvars, boxvars = partition_list(move, olympuspr.olympuspr.constvars)
  invars = *boxvars, *olympuspr.olympuspr.invars
  effects = make_olympuspr_effects(constvars, invars, olympuspr.olympuspr.outvars,
                               olympuspr.olympuspr.eqns)
  new_olympuspr = olympuspr.olympuspr.replace(constvars=constvars, invars=invars,
                                  effects=effects)
  return olympuspr.replace(olympuspr=new_olympuspr, consts=constvals), in_mutables

def num_himuts_out(olympuspr):
  return sum(len(a.lo_ty()) for a in olympuspr.final_aval_qdds if a.has_qdd)

def apply_himut(olympuspr: Olympuspr | ClosedOlympuspr, hi_args, out_mut):
  out_mut_ = iter(out_mut)
  for i, v in enumerate(olympuspr.invars):
    if v.final_qdd is not None:
      qdd = v.final_qdd
      lo_vals = it.islice(out_mut_, len(v.aval.lo_ty_qdd(qdd)))
      v.aval.update_from_loval(qdd, hi_args[i], *lo_vals)  # type: ignore
  assert next(out_mut_, None) is None

def raise_lo_outs(avals, lo_outs):
  lo_outs_ = iter(lo_outs)
  hi_outs = [t.raise_val(*it.islice(lo_outs_, len(t.lo_ty()))) for t in avals]
  assert next(lo_outs_, None) is None
  return hi_outs
