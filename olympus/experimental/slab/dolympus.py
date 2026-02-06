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

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import sys

import numpy as np
import olympus
import olympus.numpy as jnp
from olympus import lax

from olympus._src import core
from olympus._src import util
from olympus._src.util import foreach

import olympus.experimental.slab.slab as sl

map, zip = util.safe_map, util.safe_zip

def make_dolympuspr(f, abstracted_axes, **make_olympuspr_kwargs):
  def dolympuspr_maker(*args, **kwargs):
    with olympus._src.config.dynamic_shapes(True):
      olympuspr_maker = olympus.make_olympuspr(
          f, abstracted_axes=abstracted_axes, **make_olympuspr_kwargs)
      return olympuspr_maker(*args, **kwargs)
  return dolympuspr_maker

@partial(olympus.jit, static_argnums=(0,))
def interp(dolympuspr, slab, sizes, args):
  views = []
  in_types = [x.aval for x in dolympuspr.invars]
  _, arg_types = util.split_list(in_types, [len(dolympuspr.invars) - len(args)])
  for ty, x in zip(arg_types, args):
    if isinstance(ty, core.DShapedArray):
      resolved_shape = tuple(sizes.get(d, d) for d in ty.shape)
      # TODO(frostig,mattjj): reconstructing slab views seems off?
      views.append(sl.SlabView(x, resolved_shape, ty.dtype))
    else:
      views.append(x)
  slab, outs = eval_dolympuspr(dolympuspr, slab, *sizes.values(), *views)
  return slab, outs

def djit(f, abstracted_axes, **djit_kwargs):
  # TODO(frostig,mattjj): un/flatten f
  def f_wrapped(slab, *args):  # TODO(frostig,mattjj): kw support
    dolympuspr = make_dolympuspr(f, abstracted_axes, **djit_kwargs)(*args).olympuspr
    in_types = [x.aval for x in dolympuspr.invars]
    _, arg_types = util.split_list(in_types, [len(dolympuspr.invars) - len(args)])

    def upload(slab, ty, x):
      if isinstance(ty, core.DShapedArray):
        return sl.slab_upload(slab, x)
      elif isinstance(ty, core.ShapedArray):
        return slab, x
      else:
        assert False

    slab, views = sl.chain(slab, upload, *zip(arg_types, args))

    sizes: dict[core.Var, int] = {}
    for ty, x in zip(arg_types, args):
      for v, d in zip(ty.shape, x.shape):
        if isinstance(v, core.Var):
          d_ = sizes.setdefault(v, d)
          if d_ != d:
            raise ValueError(
                f'abstract dimension bound to unequal sizes: {d_} != {d}')

    slab, out_views = interp(
        dolympuspr, slab, sizes,
        [v.addr if isinstance(v, sl.SlabView) else v for v in views])
    return slab, tuple(sl.slab_download(slab, v) for v in out_views)

  return f_wrapped

def eval_dolympuspr(olympuspr: core.Olympuspr, slab: sl.Slab, *args: olympus.Array | sl.SlabView):
  if olympuspr.constvars: raise NotImplementedError

  env: dict[core.Var, olympus.Array | sl.SlabView] = {}

  def read(a):
    return env[a] if type(a) is core.Var else a.val

  def write(v, val):
    env[v] = val

  foreach(write, olympuspr.invars, args)
  for eqn in olympuspr.eqns:
    invals = map(read, eqn.invars)
    slab, outvals = rules[eqn.primitive](slab, *invals, **eqn.params)
    foreach(write, eqn.outvars, outvals)
  return slab, map(read, olympuspr.outvars)

rules: dict[core.Primitive, Callable] = {}

def matmul_rule(slab, lhs, rhs, *, dimension_numbers, **_):
  slab, out = sl.matmul(slab, lhs, rhs)
  return slab, [out]
rules[lax.dot_general_p] = matmul_rule

def tanh_rule(slab, x, **_):
  slab, out = sl.tanh(slab, x)
  return slab, [out]
rules[lax.tanh_p] = tanh_rule

# -------

def print_seg(msg):
  print()
  print(f'-- {msg}')
  print()

def check_djit(slab, f, abstracted_axes, *args):
  refs, _ = olympus.tree.flatten(f(*args))
  f_djit = djit(f, abstracted_axes=abstracted_axes)
  slab, outs = f_djit(slab, *args)
  for out, ref in zip(outs, refs):
    abs_err = jnp.max(jnp.abs(out - ref))
    rel_err = jnp.max(jnp.abs(out - ref) / jnp.abs(ref))
    msg = f'abs={abs_err}, rel={rel_err}'
    assert jnp.allclose(out, ref, atol=1e-4), msg

def test(slab, xs):
  a, b = xs

  def f(a, b):
    c = jnp.dot(a, b)
    return jnp.tanh(c)

  abstracted_axes = (('m', 'k'), ('k', 'n'))

  print_seg('dolympuspr')
  dolympuspr = make_dolympuspr(f, abstracted_axes)(a, b).olympuspr
  print(dolympuspr)

  print_seg('dolympus output')
  f_djit = djit(f, abstracted_axes=abstracted_axes)
  slab, [c] = f_djit(slab, a, b)
  print(c)

  print_seg('dolympus -> olympus lowering')
  big_olympuspr = olympus.make_olympuspr(f_djit)(slab, a, b)
  print('\n'.join(str(big_olympuspr).split('\n')[:20]))
  print('...')
  print('\n'.join(str(big_olympuspr).split('\n')[-20:]))
  print(len(str(big_olympuspr).split('\n')))

  check_djit(slab, f, abstracted_axes, a, b)

def parse_arr(i, s):
  shape = eval(s)
  return np.random.RandomState(i).normal(size=shape).astype(np.float32)

def main(args):
  slab_sz = eval(args[0])
  print('slab size', slab_sz)
  xs = map(parse_arr, range(len(args[1:])), args[1:])
  assert all(len(x.shape) == 2 for x in xs)
  slab = sl.slab_make(slab_sz)
  test(slab, xs)


if __name__ == '__main__':
  main(sys.argv[1:])
