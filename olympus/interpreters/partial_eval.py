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


from olympus._src.interpreters.partial_eval import (
  DynamicOlympusprTracer as DynamicOlympusprTracer,
  OlympusprTracer as OlympusprTracer,
  PartialVal as PartialVal,
  Val as Val,
  custom_partial_eval_rules as custom_partial_eval_rules,
  dce_olympuspr as dce_olympuspr,
  dce_olympuspr_call_rule as dce_olympuspr_call_rule,
  dce_olympuspr_closed_call_rule as dce_olympuspr_closed_call_rule,
  dce_olympuspr_consts as dce_olympuspr_consts,
  dce_rules as dce_rules,
  partial_eval_olympuspr_custom_rules as partial_eval_olympuspr_custom_rules,
  trace_to_olympuspr_dynamic as trace_to_olympuspr_dynamic,
  trace_to_olympuspr_nounits as trace_to_olympuspr_nounits,
)


_deprecations = {
  # Remove in v0.10.0
  "Const": (
    "olympus.interpreters.partial_eval.Const is deprecated.",
    None,
  ),
  "ConstFoldRule": (
    "olympus.interpreters.partial_eval.ConstFoldRule is deprecated.",
    None,
  ),
  "ConstVar": (
    "olympus.interpreters.partial_eval.ConstVar is deprecated.",
    None,
  ),
  "DCERule": (
    "olympus.interpreters.partial_eval.DCERule is deprecated.",
    None,
  ),
  "DynamicOlympusprTrace": (
    "olympus.interpreters.partial_eval.DynamicOlympusprTrace is deprecated.",
    None,
  ),
  "ForwardingRule": (
    "olympus.interpreters.partial_eval.ForwardingRule is deprecated.",
    None,
  ),
  "FreeVar": (
    "olympus.interpreters.partial_eval.FreeVar is deprecated.",
    None,
  ),
  "Olympuspr": (
    (
        "olympus.interpreters.partial_eval.Olympuspr is deprecated. Use"
        " olympus.extend.core.Olympuspr, and please note that you must"
        " `import olympus.extend` explicitly."
    ),
    None,
  ),
  "OlympusprEqnRecipe": (
    "olympus.interpreters.partial_eval.OlympusprEqnRecipe is deprecated.",
    None,
  ),
  "OlympusprStackFrame": (
    "olympus.interpreters.partial_eval.OlympusprStackFrame is deprecated.",
    None,
  ),
  "OlympusprTrace": (
    "olympus.interpreters.partial_eval.OlympusprTrace is deprecated.",
    None,
  ),
  "OlympusprTracerRecipe": (
    "olympus.interpreters.partial_eval.OlympusprTracerRecipe is deprecated.",
    None,
  ),
  "LambdaBinding": (
    "olympus.interpreters.partial_eval.LambdaBinding is deprecated.",
    None,
  ),
  "ParamsUpdater": (
    "olympus.interpreters.partial_eval.ParamsUpdater is deprecated.",
    None,
  ),
  "PartialEvalCustomResult": (
    "olympus.interpreters.partial_eval.PartialEvalCustomResult is deprecated.",
    None,
  ),
  "PartialEvalCustomRule": (
    "olympus.interpreters.partial_eval.PartialEvalCustomRule is deprecated.",
    None,
  ),
  "ResAvalUpdater": (
    "olympus.interpreters.partial_eval.ResAvalUpdater is deprecated.",
    None,
  ),
  "TracerAsName": (
    "olympus.interpreters.partial_eval.TracerAsName is deprecated.",
    None,
  ),
  "TracerId": (
    "olympus.interpreters.partial_eval.TracerId is deprecated.",
    None,
  ),
  "abstract_eval_fun": (
    "olympus.interpreters.partial_eval.abstract_eval_fun is deprecated.",
    None,
  ),
  "call_param_updaters": (
    "olympus.interpreters.partial_eval.call_param_updaters is deprecated.",
    None,
  ),
  "call_partial_eval_custom_rule": (
    "olympus.interpreters.partial_eval.call_partial_eval_custom_rule is deprecated.",
    None,
  ),
  "call_partial_eval_rules": (
    "olympus.interpreters.partial_eval.call_partial_eval_rules is deprecated.",
    None,
  ),
  "close_olympuspr": (
    "olympus.interpreters.partial_eval.close_olympuspr is deprecated.",
    None,
  ),
  "closed_call_partial_eval_custom_rule": (
    "olympus.interpreters.partial_eval.closed_call_partial_eval_custom_rule is deprecated.",
    None,
  ),
  "config": (
    "olympus.interpreters.partial_eval.config is deprecated; use olympus.config directly.",
    None,
  ),
  "const_fold_rules": (
    "olympus.interpreters.partial_eval.const_fold_rules is deprecated.",
    None,
  ),
  "convert_constvars_olympuspr": (
    "olympus.interpreters.partial_eval.convert_constvars_olympuspr is deprecated.",
    None,
  ),
  "convert_envvars_to_constvars": (
    "olympus.interpreters.partial_eval.convert_envvars_to_constvars is deprecated.",
    None,
  ),
  "convert_invars_to_constvars": (
    "olympus.interpreters.partial_eval.convert_invars_to_constvars is deprecated.",
    None,
  ),
  "custom_staging_rules": (
    "olympus.interpreters.partial_eval.custom_staging_rules is deprecated.",
    None,
  ),
  "forwarding_rules": (
    "olympus.interpreters.partial_eval.forwarding_rules is deprecated.",
    None,
  ),
  "has_effects": (
    "olympus.interpreters.partial_eval.has_effects is deprecated.",
    None,
  ),
  "instantiate_const_at": (
    "olympus.interpreters.partial_eval.instantiate_const_at is deprecated.",
    None,
  ),
  "make_olympuspr_effects": (
    "olympus.interpreters.partial_eval.make_olympuspr_effects is deprecated.",
    None,
  ),
  "move_binders_to_back": (
    "olympus.interpreters.partial_eval.move_binders_to_back is deprecated.",
    None,
  ),
  "move_binders_to_front": (
    "olympus.interpreters.partial_eval.move_binders_to_front is deprecated.",
    None,
  ),
  "new_eqn_recipe": (
    "olympus.interpreters.partial_eval.new_eqn_recipe is deprecated.",
    None,
  ),
  "partial_eval_olympuspr_custom": (
    "olympus.interpreters.partial_eval.partial_eval_olympuspr_custom is deprecated.",
    None,
  ),
  "partial_eval_olympuspr_custom_rule_not_implemented": (
    "olympus.interpreters.partial_eval.partial_eval_olympuspr_custom_rule_not_implemented is deprecated.",
    None,
  ),
  "partial_eval_olympuspr_nounits": (
    "olympus.interpreters.partial_eval.partial_eval_olympuspr_nounits is deprecated.",
    None,
  ),
  "partial_eval_wrapper_nounits": (
    "olympus.interpreters.partial_eval.partial_eval_wrapper_nounits is deprecated.",
    None,
  ),
  "partition_pvals": (
    "olympus.interpreters.partial_eval.partition_pvals is deprecated.",
    None,
  ),
  "recipe_to_eqn": (
    "olympus.interpreters.partial_eval.recipe_to_eqn is deprecated.",
    None,
  ),
  "trace_to_subolympuspr_nounits": (
    "olympus.interpreters.partial_eval.trace_to_subolympuspr_nounits is deprecated.",
    None,
  ),
  "trace_to_subolympuspr_nounits_fwd": (
    "olympus.interpreters.partial_eval.trace_to_subolympuspr_nounits_fwd is deprecated.",
    None,
  ),
  "tracers_to_olympuspr": (
    "olympus.interpreters.partial_eval.tracers_to_olympuspr is deprecated.",
    None,
  ),
}

from olympus._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
