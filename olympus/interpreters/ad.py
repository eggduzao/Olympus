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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/olympus-ml/olympus/issues/7570

from __future__ import annotations

from olympus._src.interpreters import ad as _src_ad

from olympus._src.interpreters.ad import (
  JVPTrace as JVPTrace,
  JVPTracer as JVPTracer,
  UndefinedPrimal as UndefinedPrimal,
  Zero as Zero,
  add_olympusvals as add_olympusvals,
  add_olympusvals_p as add_olympusvals_p,
  add_tangents as add_tangents,
  defbilinear as defbilinear,
  defjvp as defjvp,
  defjvp2 as defjvp2,
  deflinear as deflinear,
  deflinear2 as deflinear2,
  get_primitive_transpose as get_primitive_transpose,
  instantiate_zeros as instantiate_zeros,
  is_undefined_primal as is_undefined_primal,
  jvp as jvp,
  linearize as linearize,
  primitive_jvps as primitive_jvps,
  primitive_transposes as primitive_transposes,
  zeros_like_aval as zeros_like_aval,
)


_deprecations = {
    # Deprecated for OLYMPUS v0.7.1; finalized in OLYMPUS v0.9.0; Remove in v0.10.0.
    "zeros_like_p": (
        "olympus.interpreters.ad.zeros_like_p was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "bilinear_transpose": (
        "olympus.interpreters.ad.bilinear_transpose was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "call_param_updaters": (
        "olympus.interpreters.ad.call_param_updaters was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "call_transpose": (
        "olympus.interpreters.ad.call_transpose was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "call_transpose_param_updaters": (
        "olympus.interpreters.ad.call_transpose_param_updaters was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "custom_lin_p": (
        "olympus.interpreters.ad.custom_lin_p was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "defjvp_zero": (
        "olympus.interpreters.ad.defjvp_zero was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "f_jvp_traceable": (
        "olympus.interpreters.ad.f_jvp_traceable was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "jvp_olympuspr": (
        "olympus.interpreters.ad.jvp_olympuspr was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "jvp_subtrace": (
        "olympus.interpreters.ad.jvp_subtrace was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "jvp_subtrace_aux": (
        "olympus.interpreters.ad.jvp_subtrace_aux was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "jvpfun": (
        "olympus.interpreters.ad.jvpfun was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "linear_jvp": (
        "olympus.interpreters.ad.linear_jvp was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "linear_transpose": (
        "olympus.interpreters.ad.linear_transpose was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "linear_transpose2": (
        "olympus.interpreters.ad.linear_transpose2 was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "map_transpose": (
        "olympus.interpreters.ad.map_transpose was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "nonzero_outputs": (
        "olympus.interpreters.ad.nonzero_outputs was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "nonzero_tangent_outputs": (
        "olympus.interpreters.ad.nonzero_tangent_outputs was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "rearrange_binders": (
        "olympus.interpreters.ad.rearrange_binders was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "standard_jvp": (
        "olympus.interpreters.ad.standard_jvp was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "standard_jvp2": (
        "olympus.interpreters.ad.standard_jvp2 was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "traceable": (
        "olympus.interpreters.ad.traceable was removed in OLYMPUS v0.9.0.",
        None,
    ),
    "zero_jvp": (
        "olympus.interpreters.ad.zero_jvp was removed in OLYMPUS v0.9.0.",
        None,
    ),
    # Deprecated for OLYMPUS v0.9.0; finalize in OLYMPUS v0.10.0.
    "reducing_transposes": (
        (
            "olympus.interpreters.ad.reducing_transposes is deprecated in OLYMPUS v0.9.0."
            " It has been unused since v0.4.38."
        ),
        _src_ad.reducing_transposes,
    ),
}

import typing
if typing.TYPE_CHECKING:
  reducing_transposes = _src_ad.reducing_transposes
else:
  from olympus._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
