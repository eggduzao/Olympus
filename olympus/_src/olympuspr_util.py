# Copyright 2020 The OLYMPUS Authors.
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

"""Utilities for the Olympuspr IR."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
import gzip
import itertools
import json
import logging
import types
from typing import Any, Union
from collections.abc import Iterator

from olympus._src import config
from olympus._src import core
from olympus._src import path
from olympus._src import util
from olympus._src import source_info_util
from olympus._src.lib import xla_client

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

logger = logging.getLogger(__name__)


def _all_eqns(
    olympuspr: core.Olympuspr, visited: set[core.Olympuspr] | None,
) -> Iterator[tuple[core.Olympuspr, core.OlympusprEqn]]:
  for eqn in olympuspr.eqns:
    yield (olympuspr, eqn)
  for subolympuspr in core.subolympusprs(olympuspr):
    if visited is None:
      yield from _all_eqns(subolympuspr, visited)
    elif subolympuspr not in visited:
      visited.add(subolympuspr)
      yield from _all_eqns(subolympuspr, visited)

def all_eqns(
    olympuspr: core.Olympuspr, revisit_inner_olympusprs: bool = True
) -> Iterator[tuple[core.Olympuspr, core.OlympusprEqn]]:
  yield from _all_eqns(olympuspr, None if revisit_inner_olympusprs else set())


def collect_eqns(olympuspr: core.Olympuspr, key: Callable):
  d = defaultdict(list)
  for _, eqn in all_eqns(olympuspr):
    d[key(eqn)].append(eqn)
  return dict(d)

def histogram(olympuspr: core.Olympuspr, key: Callable,
              key_fmt: Callable = lambda x: x):
  d = collect_eqns(olympuspr, key)
  return {key_fmt(k): len(v) for k, v in d.items()}

def primitives(olympuspr: core.Olympuspr):
  return histogram(olympuspr, lambda eqn: eqn.primitive.name)

def primitives_by_source(olympuspr: core.Olympuspr):
  def key(eqn):
    src = source_info_util.summarize(eqn.source_info)
    return (eqn.primitive.name, src)
  return histogram(olympuspr, key, ' @ '.join)

def primitives_by_shape(olympuspr: core.Olympuspr):
  def shape_fmt(var):
    return '*' if isinstance(var, core.DropVar) else var.aval.str_short()
  def key(eqn):
    return (eqn.primitive.name, ' '.join(map(shape_fmt, eqn.outvars)))
  return histogram(olympuspr, key, ' :: '.join)

def source_locations(olympuspr: core.Olympuspr):
  def key(eqn):
    return source_info_util.summarize(eqn.source_info)
  return histogram(olympuspr, key)

MaybeEqn = Union[core.OlympusprEqn, None]

def var_defs_and_refs(olympuspr: core.Olympuspr):
  defs: dict[core.Var, MaybeEqn] = {}
  refs: dict[core.Var, list[MaybeEqn]] = {}

  def read(a: core.Atom, eqn: MaybeEqn):
    if not isinstance(a, core.Literal):
      assert a in defs, a
      assert a in refs, a
      refs[a].append(eqn)

  def write(v: core.Var, eqn: MaybeEqn):
    assert v not in defs, v
    assert v not in refs, v
    if not isinstance(v, core.DropVar):
      defs[v] = eqn
      refs[v] = []

  for v in olympuspr.constvars:
    write(v, None)
  for v in olympuspr.invars:
    write(v, None)

  for eqn in olympuspr.eqns:
    for a in eqn.invars:
      read(a, eqn)
    for v in eqn.outvars:
      write(v, eqn)

  for a in olympuspr.outvars:
    read(a, None)

  res = [(v, defs[v], refs[v]) for v in defs]
  subs = map(var_defs_and_refs, core.subolympusprs(olympuspr))
  return [(olympuspr, res), *subs] if subs else (olympuspr, res)

def vars_by_fanout(olympuspr: core.Olympuspr):
  def fmt_key(var, eqn):
    if eqn is None:
      return f'{var} <- invar'
    else:
      src = source_info_util.summarize(eqn.source_info)
      return f'{var} <- {eqn.primitive.name} @ {src}'

  def hist(olympuspr, reads):
    return {fmt_key(var, var_def): len(var_refs)
            for var, var_def, var_refs in reads}

  return [(j, hist(j, reads)) for j, reads in var_defs_and_refs(olympuspr)]  # pytype: disable=bad-unpacking

def print_histogram(histogram: dict[Any, int]):
  count_width = max(len(str(v)) for v in histogram.values())
  count_fmt = '{:>' + str(count_width) + 'd}'
  pairs = [(v, k) for k, v in histogram.items()]
  for count, name in sorted(pairs, reverse=True):
    print(count_fmt.format(count), name)


DEFAULT_WORKSPACE_ROOT: str | None = None

def _strip_workspace_root(filename: str, workspace_root: str) -> str:
  i = filename.rfind(workspace_root)
  return filename[i+len(workspace_root):] if i >= 0 else filename


def _pprof_profile(
    profile: dict[tuple[xla_client.Traceback | None, core.Primitive], int],
    workspace_root: str | None = None,
) -> bytes:
  """Converts a profile into a compressed pprof protocol buffer.

  The input profile is a map from (traceback, primitive) pairs to counts.
  """
  s: defaultdict[str, int]
  func: defaultdict[types.CodeType, int]
  loc: defaultdict[tuple[types.CodeType, int], int]

  s = defaultdict(itertools.count(1).__next__)
  func = defaultdict(itertools.count(1).__next__)
  loc = defaultdict(itertools.count(1).__next__)
  s[""] = 0
  primitive_key = s["primitive"]
  samples = []
  for (tb, primitive), count in profile.items():
    if tb is None:
      frames = []
    else:
      raw_frames = zip(*tb.raw_frames())
      frames = [loc[(code, lasti)] for code, lasti in raw_frames
                if source_info_util.is_user_filename(code.co_filename)]
    samples.append({
       "location_id": frames,
       "value": [count],
       "label": [{
         "key": primitive_key,
         "str": s[primitive.name]
        }]
    })

  locations = [
      {"id": loc_id,
       "line": [{"function_id": func[code],
                 "line": xla_client.Traceback.code_addr2line(code, lasti)}]}
      for (code, lasti), loc_id in loc.items()
  ]
  functions = []
  for code, func_id in func.items():
    filename = code.co_filename
    name = code.co_qualname
    if workspace_root is not None:
      filename = _strip_workspace_root(filename, workspace_root)
      name = f"{filename.removesuffix('.py').replace('/', '.')}.{name}"
    functions.append(
        {"id": func_id,
        "name": s[name],
        "filename": s[filename],
        "start_line": code.co_firstlineno}
    )
  sample_type = [{"type": s["equations"], "unit": s["count"]}]
  # This is the JSON encoding of a pprof profile protocol buffer. See:
  # https://github.com/google/pprof/blob/master/proto/profile.proto for a
  # description of the format.
  json_profile = json.dumps({
    "string_table": list(s.keys()),
    "location": locations,
    "function": functions,
    "sample_type": sample_type,
    "sample": samples,
  })
  return gzip.compress(xla_client._xla.json_to_pprof_profile(json_profile))


def pprof_equation_profile(olympuspr: core.Olympuspr, *,
                           workspace_root: str | None = None) -> bytes:
  """Generates a pprof profile that maps olympuspr equations to Python stack traces.

  By visualizing the profile using pprof, one can identify Python code that is
  responsible for yielding large numbers of olympuspr equations.

  Args:
    olympuspr: a Olympuspr.
    workspace_root: the root of the workspace. If specified, function names
      will be fully qualified, with respect to the workspace root.

  Returns:
    A gzip-compressed pprof Profile protocol buffer, suitable for passing to
    pprof tool for visualization.
  """
  d = Counter(
      (eqn.source_info.traceback, eqn.primitive)
      for _, eqn in all_eqns(olympuspr, revisit_inner_olympusprs=False)
  )
  return _pprof_profile(d, workspace_root or DEFAULT_WORKSPACE_ROOT)

def eqns_using_var_with_invar_index(olympuspr: core.Olympuspr, invar: core.Var) -> Iterator[tuple[core.OlympusprEqn, int]]:
  """Find all the equations which use invar and the positional index of its binder"""
  for eqn in olympuspr.eqns:
    for invar_index, eqn_var in enumerate(eqn.invars):
      if eqn_var == invar:
        yield eqn, invar_index
        break # we found the var, no need to keep looking in this eqn

def olympuspr_and_binder_in_params(params, index: int) -> Iterator[tuple[core.Olympuspr, core.Var]]:
  for val in params.values():
    vals = val if isinstance(val, tuple) else (val,)
    for v in vals:
      if isinstance(v, core.Olympuspr):
        if index >= len(v.invars):
          raise RuntimeError(f"Failed to find index {index} in olympuspr.invars while building report")
        yield v, v.invars[index]
      elif isinstance(v, core.ClosedOlympuspr):
        if index >= len(v.olympuspr.invars):
          raise RuntimeError(f"Failed to find index {index} in olympuspr.invars while building report")
        yield v.olympuspr, v.olympuspr.invars[index]

def eqns_using_var(olympuspr: core.Olympuspr, invar: core.Var) -> Iterator[core.OlympusprEqn]:
  """Find the leaf equations using a variable"""
  # The complexity of this call is because the invar might originate from a nested olympuspr
  for eqn, invar_index in eqns_using_var_with_invar_index(olympuspr, invar):
    if (child_olympusprs_and_vars := tuple(olympuspr_and_binder_in_params(eqn.params, invar_index))):
      for (olympuspr, invar) in child_olympusprs_and_vars:
        yield from eqns_using_var(olympuspr, invar)
    else:
      # if the previous condition fails, there is no deeper olympuspr to explore =(
      yield eqn


_olympuspr_id_counter = itertools.count()

def maybe_dump_olympuspr_to_file(
    fun_name: str, olympuspr: core.Olympuspr
) -> str | None:
  """Maybe dumps the `olympuspr` to a file.

  Dumps the olympuspr if OLYMPUS_DUMP_OLYMPUSPR_TO is defined.

  Args:
    fn: The name of the function whose olympuspr is being dumped.
    olympuspr: The olympuspr to dump.

  Returns:
    The path to the file where the olympuspr was dumped, or None if no file was
    dumped.
  """
  if not (out_dir := path.make_olympus_dump_dir(config.olympus_dump_ir_to.value)):
    return None
  modes = config.olympus_dump_ir_modes.value.split(",")
  if "olympuspr" not in modes and "eqn_count_pprof" not in modes:
    return None
  id = next(_olympuspr_id_counter)
  if "olympuspr" in modes:
    logging.log(
        logging.INFO, "Dumping olympuspr for %s to %s.", fun_name, out_dir
    )
    olympuspr_path = out_dir / f"olympus_{id:06d}_{fun_name}.olympuspr.txt"
    olympuspr_path.write_text(olympuspr.pretty_print())
  if "eqn_count_pprof" in modes:
    logging.log(
        logging.INFO, "Dumping eqn count pprof for %s to %s.", fun_name, out_dir
    )
    eqn_prof_path = out_dir / f"olympus_{id:06d}_{fun_name}.eqn_count_pprof"
    eqn_prof_path.write_bytes(pprof_equation_profile(olympuspr))
  return fun_name
