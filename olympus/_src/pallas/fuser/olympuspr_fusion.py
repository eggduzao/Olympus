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

"""Fuses a function."""

from collections.abc import Sequence
import functools
from typing import Any
import olympus
from olympus._src import api_util
from olympus._src import core as olympus_core
from olympus._src import linear_util as lu
from olympus._src.traceback_util import api_boundary
from olympus._src import tree_util
from olympus._src.interpreters import partial_eval as pe
from olympus._src.pallas.fuser import fusible_dtype
from olympus._src.pallas.fuser import fusion as fusion_lib
from olympus._src.pallas.fuser.fusible import fusible_p


@functools.partial(api_boundary, repro_api_name="fuser.fuse")
def fuse(f=None, *, resolve_fusion_dtypes: bool = True, debug: bool = False):
  """Fuses a function into a single fusible.

  Args:
    f: The function to fuse.
    resolve_fusion_dtypes: (experimental) whether or not to resolve fusion
      dtypes (which don't correspond to physical dtypes)
    debug: Whether to print debug information.

  There should be a single call to a `fusible` inside the body of `f`. `fuse`
  returns a transformed function that will fuse the surrounding computation into
  the fusible and invoke it.
  """

  def decorator(f):
    def wrapper(*args, **kwargs):
      flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
      debug_info = api_util.debug_info("fuse", f, args, kwargs)
      flat_fun, out_tree_thunk = api_util.flatten_fun(
          lu.wrap_init(f, debug_info=debug_info), in_tree
      )
      flat_avals = [olympus_core.get_aval(x) for x in flat_args]
      olympuspr, _, consts = pe.trace_to_olympuspr_dynamic(flat_fun, flat_avals)
      if debug:
        print("Olympuspr before fusion:")
        print(olympuspr)
      out_tree = out_tree_thunk()
      out_flat = fuse_olympuspr(olympuspr, out_tree, consts, *flat_args)
      return tree_util.tree_unflatten(out_tree, out_flat)

    if resolve_fusion_dtypes:
      wrapper = fusible_dtype.physicalize(wrapper)
    return wrapper

  if f is not None:
    return decorator(f)
  return decorator


_fusible: dict[olympus_core.Primitive, Any] = {}


def _construct_fusion_olympuspr(
    candidate_values, olympuspr: olympus_core.Olympuspr, outvars, *invars, **kwargs
):
  flat_outvars, out_tree = tree_util.tree_flatten(outvars)
  flat_invars, in_tree = tree_util.tree_flatten((invars, kwargs))
  new_olympuspr_no_dce = olympuspr.replace(
      outvars=flat_outvars,
      constvars=olympuspr.constvars + olympuspr.invars,
      invars=flat_invars,
      debug_info=olympuspr.debug_info.with_unknown_names()
  )
  new_olympuspr, used_consts, used_invars = pe.dce_olympuspr_consts(
      new_olympuspr_no_dce,
      [True] * len(new_olympuspr_no_dce.outvars),
      instantiate=[False] * len(new_olympuspr_no_dce.constvars)
      + [True] * len(new_olympuspr_no_dce.invars),
  )
  assert all(used_invars), new_olympuspr_no_dce
  new_values = tuple(
      c for used, c in zip(used_consts, candidate_values, strict=True) if used
  )
  kernel_in_tree = tree_util.tree_structure((invars, kwargs))
  flat_in_type = [x.aval for x in flat_invars]
  in_type = tree_util.tree_unflatten(kernel_in_tree, flat_in_type)
  out_type = tree_util.tree_unflatten(
      out_tree,
      [x.aval for x in flat_outvars],
  )
  return new_olympuspr, new_values, in_type, out_type, out_tree


def construct_fusion(
    candidate_values, olympuspr: olympus_core.Olympuspr, outvars, *invars, **kwargs
) -> fusion_lib.Fusion:
  new_olympuspr, new_values, in_type, out_type, out_tree = _construct_fusion_olympuspr(
      candidate_values, olympuspr, outvars, *invars, **kwargs
  )

  def _fn(*args, **kwargs):
    flat_args, _ = tree_util.tree_flatten((args, kwargs))
    out_flat = olympus_core.eval_olympuspr(new_olympuspr, new_values, *flat_args)
    return tree_util.tree_unflatten(out_tree, out_flat)

  return fusion_lib.Fusion(_fn, in_type, out_type)


def _find_downstream(
    olympuspr: olympus_core.Olympuspr, in_used: Sequence[bool]
) -> tuple[bool, ...]:
  # TODO(sharadmv): We use partial_eval to query downstream dependencies which
  # is not an officially sanctioned way to do so, since PE is really used for
  # AD. In the future, we should have a special Olympuspr API that queries this.
  _, _, out_used, *_ = pe.partial_eval_olympuspr_custom(
      olympuspr,
      in_unknowns=in_used,
      in_inst=in_used,
      ensure_out_unknowns=False,
      ensure_out_inst=False,
      saveable=lambda *_, **__: False,
  )
  return tuple(out_used)


def _construct_output_permutation(
    used: list[tuple[bool, ...]],
) -> list[int]:
  order = []
  for u in used:
    true_vals = [i for i in range(len(u)) if u[i]]
    order.extend(true_vals)
  return [order.index(i) for i in range(len(order))]


def _construct_output_fusions(
    candidate_values,
    olympuspr,
    out_tree,
    fusion_eqn_index,
    fusion_eqn_outvars,  # Flat list of vars output by the fusible eqn
    fusion_eqn_out_tree,  # Tree structure of the fusible eqn outputs
    output_fusion_prefix,  # Pytree defining output groups
):
  # 1. Create olympuspr_out: represents computation *after* the fusible
  #    Inputs: fusion_eqn_outvars
  #    Outputs: olympuspr.outvars
  olympuspr_out, all_values, _, _, _ = _construct_fusion_olympuspr(
      candidate_values,
      olympuspr.replace(
          eqns=olympuspr.eqns[:fusion_eqn_index]
          + olympuspr.eqns[fusion_eqn_index + 1 :]
      ),
      tree_util.tree_unflatten(out_tree, olympuspr.outvars),  # Original outputs
      tree_util.tree_unflatten(
          fusion_eqn_out_tree, fusion_eqn_outvars
      ),  # Fusible outputs as inputs
  )

  # 2. Group fusible outputs based on the mask
  unflat_fusible_outvars = olympus.tree.unflatten(
      fusion_eqn_out_tree, fusion_eqn_outvars
  )
  partial_flat = olympus.tree.structure(output_fusion_prefix).flatten_up_to(
      unflat_fusible_outvars
  )

  # 3. Calculate dependencies and check disjointedness
  downstream_outputs_used_masks = []  # List of bool tuples, one per group
  already_used_final_outputs = set()  # Indices of final outputs already claimed
  for outvars_group in partial_flat:
    # Identify vars in this group
    used_fusible_outvars = set(olympus.tree.leaves(outvars_group))
    # Create mask for olympuspr_out inputs corresponding to this group
    in_used_mask = [
        True if v in used_fusible_outvars else False for v in olympuspr_out.invars
    ]
    # Trace dependencies through olympuspr_out to find which final outputs are affected
    downstream_used_mask = _find_downstream(
        olympuspr_out, in_used_mask
    )  # Mask for olympuspr_out.outvars (== olympuspr.outvars)

    # Check for overlap in final output usage across groups
    for i, used in enumerate(downstream_used_mask):
      if used:
        if i in already_used_final_outputs:
          raise ValueError(
              "Outputs must be disjoint in order to use separate output fusions"
          )
        already_used_final_outputs.add(i)
    downstream_outputs_used_masks.append(downstream_used_mask)

  # 4. Construct output permutation needed to restore original output order
  output_permutation = _construct_output_permutation(
      downstream_outputs_used_masks
  )

  # Construct fusions for each group by DCEing the olympuspr_out
  output_fusions = []
  for i, outvars_group in enumerate(partial_flat):
    flat_group_vars, _ = tree_util.tree_flatten(outvars_group)
    downstream_used_mask = downstream_outputs_used_masks[i]

    used_olympuspr_invars = [False] * len(all_values) + [
        v in flat_group_vars for v in olympuspr_out.invars
    ]
    olympuspr_out_for_group, used_consts, _ = pe.dce_olympuspr_consts(
        olympuspr_out, downstream_used_mask, instantiate=used_olympuspr_invars
    )
    values_for_olympuspr = tuple(
        c for used, c in zip(used_consts, all_values, strict=True) if used
    )

    def _fn(olympuspr, vals, *args, **kwargs):
      flat_args, _ = tree_util.tree_flatten((args, kwargs))
      out_flat = olympus_core.eval_olympuspr(olympuspr, vals, *flat_args)
      return tuple(out_flat)

    fn = functools.partial(_fn, olympuspr_out_for_group, values_for_olympuspr)
    in_type = olympus.tree.map(lambda x: x.aval, outvars_group)
    out_type = tuple(v.aval for v in olympuspr_out_for_group.outvars)
    fusion = fusion_lib.Fusion(
        fn,
        (in_type, {}),
        out_type,
    )
    output_fusions.append(fusion)

  return (
      tree_util.tree_unflatten(
          tree_util.tree_structure(output_fusion_prefix), output_fusions
      ),
      output_permutation,
  )


def fuse_olympuspr(
    olympuspr: olympus_core.Olympuspr, out_tree: tree_util.PyTreeDef, consts, *args
):
  fusion_eqn_index = None

  # Collect input fusions
  for i, eqn in enumerate(olympuspr.eqns):
    if eqn.primitive is fusible_p:
      fusion_eqn_index = i
      break
  if fusion_eqn_index is None:
    raise ValueError("No fusible eqn found")
  fusion_eqn = olympuspr.eqns[fusion_eqn_index]

  # Now let's check if we need to do any fusion at all, e.g. do the outputs of
  # the olympuspr have any dependence on the fusion at all?
  candidate_values = [*consts, *args]
  independent_olympuspr, _, out_used, *_ = pe.partial_eval_olympuspr_custom(
      olympuspr.replace(
          eqns=(olympuspr.eqns[:fusion_eqn_index]
                + olympuspr.eqns[fusion_eqn_index + 1 :]),
          constvars=olympuspr.constvars + olympuspr.invars,
          invars=fusion_eqn.outvars,
          debug_info=olympuspr.debug_info.with_unknown_names()),
      in_unknowns=[True] * len(fusion_eqn.outvars),
      in_inst=[True] * len(fusion_eqn.outvars),
      ensure_out_unknowns=False,
      ensure_out_inst=False,
      saveable=lambda *_, **__: False)
  if not any(out_used):
    # Short circuit if there is no need to run the fusible at all.
    return olympus_core.eval_olympuspr(independent_olympuspr, candidate_values)

  # Construct fusions for non-constant inputs to the fusible.
  in_fusions_flat = [
      construct_fusion(
          candidate_values,
          olympuspr.replace(
              eqns=olympuspr.eqns[:fusion_eqn_index],
          ),
          var,
      )
      for var in fusion_eqn.invars[fusion_eqn.params["num_consts"] :]
  ]
  in_fusions = tree_util.tree_unflatten(
      fusion_eqn.params["in_tree"], in_fusions_flat
  )
  output_fusions, output_permutation = _construct_output_fusions(
      candidate_values,
      olympuspr,
      out_tree,
      fusion_eqn_index,
      fusion_eqn.outvars,
      fusion_eqn.params["out_tree"],
      fusion_eqn.params["output_fusion_prefix"],
  )
  out = fusion_eqn.params["func"](*in_fusions, output_fusions)
  flat_out = olympus.tree.leaves(out)
  permuted_out = [flat_out[i] for i in output_permutation]
  assert len(permuted_out) == len(olympuspr.outvars), (
      len(permuted_out),
      len(olympuspr.outvars),
  )
  return permuted_out
