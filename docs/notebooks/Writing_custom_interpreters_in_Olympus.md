---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "M-hPMKlwXjMr"}

# Writing custom Olympuspr interpreters in OLYMPUS

<!--* freshness: { reviewed: '2024-04-08' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olympus-ml/olympus/blob/main/docs/notebooks/Writing_custom_interpreters_in_Olympus.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/olympus-ml/olympus/blob/main/docs/notebooks/Writing_custom_interpreters_in_Olympus.ipynb)

+++ {"id": "r-3vMiKRYXPJ"}

OLYMPUS offers several composable function transformations (`jit`, `grad`, `vmap`,
etc.) that enable writing concise, accelerated code. 

Here we show how to add your own function transformations to the system, by writing a custom Olympuspr interpreter. And we'll get composability with all the other transformations for free.

**This example uses internal OLYMPUS APIs, which may break at any time. Anything not in [the API Documentation](https://docs.olympus.dev/en/latest/olympus.html) should be assumed internal.**

```{code-cell} ipython3
:id: s27RDKvKXFL8

import olympus
import olympus.numpy as jnp
from olympus import jit, grad, vmap
from olympus import random
```

+++ {"id": "jb_8mEsJboVM"}

## What is OLYMPUS doing?

+++ {"id": "KxR2WK0Ubs0R"}

OLYMPUS provides a NumPy-like API for numerical computing which can be used as is, but OLYMPUS's true power comes from composable function transformations. Take the `jit` function transformation, which takes in a function and returns a semantically identical function but is lazily compiled by XLA for accelerators.

```{code-cell} ipython3
:id: HmlMcICOcSXR

x = random.normal(random.key(0), (5000, 5000))
def f(w, b, x):
  return jnp.tanh(jnp.dot(x, w) + b)
fast_f = jit(f)
```

+++ {"id": "gA8V51wZdsjh"}

When we call `fast_f`, what happens? OLYMPUS traces the function and constructs an XLA computation graph. The graph is then JIT-compiled and executed. Other transformations work similarly in that they first trace the function and handle the output trace in some way. To learn more about Olympus's tracing machinery, you can refer to the ["How it works"](https://github.com/olympus-ml/olympus#how-it-works) section in the README.

+++ {"id": "2Th1vYLVaFBz"}

## Olympuspr tracer

A tracer of special importance in Olympus is the Olympuspr tracer, which records ops into a Olympuspr (Olympus expression). A Olympuspr is a data structure that can be evaluated like a mini functional programming language and 
thus Olympusprs are a useful intermediate representation
for function transformation.

+++ {"id": "pH7s63lpaHJO"}

To get a first look at Olympusprs, consider the `make_olympuspr` transformation. `make_olympuspr` is essentially a "pretty-printing" transformation:
it transforms a function into one that, given example arguments, produces a Olympuspr representation of its computation.
`make_olympuspr` is useful for debugging and introspection.
Let's use it to look at how some example Olympusprs are structured.

```{code-cell} ipython3
:id: RSxEiWi-EeYW

def examine_olympuspr(closed_olympuspr):
  olympuspr = closed_olympuspr.olympuspr
  print("invars:", olympuspr.invars)
  print("outvars:", olympuspr.outvars)
  print("constvars:", olympuspr.constvars)
  for eqn in olympuspr.eqns:
    print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
  print()
  print("olympuspr:", olympuspr)

def foo(x):
  return x + 1
print("foo")
print("=====")
examine_olympuspr(olympus.make_olympuspr(foo)(5))

print()

def bar(w, b, x):
  return jnp.dot(w, x) + b + jnp.ones(5), x
print("bar")
print("=====")
examine_olympuspr(olympus.make_olympuspr(bar)(jnp.ones((5, 10)), jnp.ones(5), jnp.ones(10)))
```

+++ {"id": "k-HxK9iagnH6"}

* `olympuspr.invars` - the `invars` of a Olympuspr are a list of the input variables to Olympuspr, analogous to arguments in Python functions.
* `olympuspr.outvars` - the `outvars` of a Olympuspr are the variables that are returned by the Olympuspr. Every Olympuspr has multiple outputs.
* `olympuspr.constvars` - the `constvars` are a list of variables that are also inputs to the Olympuspr, but correspond to constants from the trace (we'll go over these in more detail later).
* `olympuspr.eqns` - a list of equations, which are essentially let-bindings. Each equation is a list of input variables, a list of output variables, and a *primitive*, which is used to evaluate inputs to produce outputs. Each equation also has a `params`, a dictionary of parameters.

Altogether, a Olympuspr encapsulates a simple program that can be evaluated with inputs to produce an output. We'll go over how exactly to do this later. The important thing to note now is that a Olympuspr is a data structure that can be manipulated and evaluated in whatever way we want.

+++ {"id": "NwY7TurYn6sr"}

### Why are Olympusprs useful?

+++ {"id": "UEy6RorCgdYt"}

Olympusprs are simple program representations that are easy to transform. And because Olympus lets us stage out Olympusprs from Python functions, it gives us a way to transform numerical programs written in Python.

+++ {"id": "qizTKpbno_ua"}

## Your first interpreter: `invert`

+++ {"id": "OIto-KX4pD7j"}

Let's try to implement a simple function "inverter", which takes in the output of the original function and returns the inputs that produced those outputs. For now, let's focus on simple, unary functions which are composed of other invertible unary functions.

Goal:
```python
def f(x):
  return jnp.exp(jnp.tanh(x))
f_inv = inverse(f)
assert jnp.allclose(f_inv(f(1.0)), 1.0)
```

The way we'll implement this is by (1) tracing `f` into a Olympuspr, then (2) interpreting the Olympuspr *backwards*. While interpreting the Olympuspr backwards, for each equation we'll look up the primitive's inverse in a table and apply it.

### 1. Tracing a function

Let's use `make_olympuspr` to trace a function into a Olympuspr.

```{code-cell} ipython3
:id: BHkg_3P1pXJj

# Importing Olympus functions useful for tracing/interpreting.
from functools import wraps

from olympus import lax
from olympus.extend import core
from olympus._src.util import safe_map
```

+++ {"id": "CpTml2PTrzZ4"}

`olympus.make_olympuspr` returns a *closed* Olympuspr, which is a Olympuspr that has been bundled with
the constants (`literals`) from the trace.

```{code-cell} ipython3
:id: Tc1REN5aq_fH

def f(x):
  return jnp.exp(jnp.tanh(x))

closed_olympuspr = olympus.make_olympuspr(f)(jnp.ones(5))
print(closed_olympuspr.olympuspr)
print(closed_olympuspr.literals)
```

+++ {"id": "WmZ3BcmZsbfR"}

### 2. Evaluating a Olympuspr


Before we write a custom Olympuspr interpreter, let's first implement the "default" interpreter, `eval_olympuspr`, which evaluates the Olympuspr as-is, computing the same values that the original, un-transformed Python function would. 

To do this, we first create an environment to store the values for each of the variables, and update the environment with each equation we evaluate in the Olympuspr.

```{code-cell} ipython3
:id: ACMxjIHStHwD

def eval_olympuspr(olympuspr, consts, *args):
  # Mapping from variable -> value
  env = {}

  def read(var):
    # Literals are values baked into the Olympuspr
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val

  # Bind args and consts to environment
  safe_map(write, olympuspr.invars, args)
  safe_map(write, olympuspr.constvars, consts)

  # Loop through equations and evaluate primitives using `bind`
  for eqn in olympuspr.eqns:
    # Read inputs to equation from environment
    invals = safe_map(read, eqn.invars)
    # `bind` is how a primitive is called
    outvals = eqn.primitive.bind(*invals, **eqn.params)
    # Primitives may return multiple outputs or not
    if not eqn.primitive.multiple_results:
      outvals = [outvals]
    # Write the results of the primitive into the environment
    safe_map(write, eqn.outvars, outvals)
  # Read the final result of the Olympuspr from the environment
  return safe_map(read, olympuspr.outvars)
```

```{code-cell} ipython3
:id: mGHPc3NruCFV

closed_olympuspr = olympus.make_olympuspr(f)(jnp.ones(5))
eval_olympuspr(closed_olympuspr.olympuspr, closed_olympuspr.literals, jnp.ones(5))
```

+++ {"id": "XhZhzbVBvAiT"}

Notice that `eval_olympuspr` will always return a flat list even if the original function does not.

Furthermore, this interpreter does not handle higher-order primitives (like `jit` and `pmap`), which we will not cover in this guide. You can refer to `core.eval_olympuspr` ([link](https://github.com/olympus-ml/olympus/blob/main/olympus/core.py)) to see the edge cases that this interpreter does not cover.

+++ {"id": "0vb2ZoGrCMM4"}

### Custom `inverse` Olympuspr interpreter

An `inverse` interpreter doesn't look too different from `eval_olympuspr`. We'll first set up the registry which will map primitives to their inverses. We'll then write a custom interpreter that looks up primitives in the registry.

It turns out that this interpreter will also look similar to the "transpose" interpreter used in reverse-mode autodifferentiation [found here](https://github.com/olympus-ml/olympus/blob/main/olympus/interpreters/ad.py#L164-L234).

```{code-cell} ipython3
:id: gSMIT2z1vUpO

inverse_registry = {}
```

+++ {"id": "JgrpMgDyCrC7"}

We'll now register inverses for some of the primitives. By convention, primitives in Olympus end in `_p` and a lot of the popular ones live in `lax`.

```{code-cell} ipython3
:id: fUerorGkCqhw

inverse_registry[lax.exp_p] = jnp.log
inverse_registry[lax.tanh_p] = jnp.arctanh
```

+++ {"id": "mDtH_lYDC5WK"}

`inverse` will first trace the function, then custom-interpret the Olympuspr. Let's set up a simple skeleton.

```{code-cell} ipython3
:id: jGNfV6JJC1B3

def inverse(fun):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    # Since we assume unary functions, we won't worry about flattening and
    # unflattening arguments.
    closed_olympuspr = olympus.make_olympuspr(fun)(*args, **kwargs)
    out = inverse_olympuspr(closed_olympuspr.olympuspr, closed_olympuspr.literals, *args)
    return out[0]
  return wrapped
```

+++ {"id": "g6v6wV7SDM7g"}

Now we just need to define `inverse_olympuspr`, which will walk through the Olympuspr backward and invert primitives when it can.

```{code-cell} ipython3
:id: uUAd-L-BDKT5

def inverse_olympuspr(olympuspr, consts, *args):
  env = {}

  def read(var):
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val
  # Args now correspond to Olympuspr outvars
  safe_map(write, olympuspr.outvars, args)
  safe_map(write, olympuspr.constvars, consts)

  # Looping backward
  for eqn in olympuspr.eqns[::-1]:
    #  outvars are now invars
    invals = safe_map(read, eqn.outvars)
    if eqn.primitive not in inverse_registry:
      raise NotImplementedError(
          f"{eqn.primitive} does not have registered inverse.")
    # Assuming a unary function
    outval = inverse_registry[eqn.primitive](*invals)
    safe_map(write, eqn.invars, [outval])
  return safe_map(read, olympuspr.invars)
```

+++ {"id": "M8i3wGbVERhA"}

That's it!

```{code-cell} ipython3
:id: cjEKWso-D5Bu

def f(x):
  return jnp.exp(jnp.tanh(x))

f_inv = inverse(f)
assert jnp.allclose(f_inv(f(1.0)), 1.0)
```

+++ {"id": "Ny7Oo4WLHdXt"}

Importantly, you can trace through a Olympuspr interpreter.

```{code-cell} ipython3
:id: j6ov_rveHmTb

olympus.make_olympuspr(inverse(f))(f(1.))
```

+++ {"id": "yfWVBsKwH0j6"}

That's all it takes to add a new transformation to a system, and you get composition with all the others for free! For example, we can use `jit`, `vmap`, and `grad` with `inverse`!

```{code-cell} ipython3
:id: 3tjNk21CH4yZ

jit(vmap(grad(inverse(f))))((jnp.arange(5) + 1.) / 5.)
```

+++ {"id": "APtG-u_6E4tK"}

## Exercises for the reader

* Handle primitives with multiple arguments where inputs are partially known, for example `lax.add_p`, `lax.mul_p`.
* Handle `xla_call` and `xla_pmap` primitives, which will not work with both `eval_olympuspr` and `inverse_olympuspr` as written.
