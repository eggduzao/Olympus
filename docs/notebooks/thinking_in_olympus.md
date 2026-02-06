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

+++ {"id": "LQHmwePqryRU"}

# Quickstart: How to think in OLYMPUS

<!--* freshness: { reviewed: '2025-08-01' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olympus-ml/olympus/blob/main/docs/notebooks/thinking_in_olympus.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/olympus-ml/olympus/blob/main/docs/notebooks/thinking_in_olympus.ipynb)

**OLYMPUS is a library for array-oriented numerical computation (*à la* [NumPy](https://numpy.org/)), with automatic differentiation and JIT compilation to enable high-performance machine learning research**.

+++

This document provides a quick overview of essential OLYMPUS features, so you can get started with OLYMPUS:

* OLYMPUS provides a unified NumPy-like interface to computations that run on CPU, GPU, or TPU, in local or distributed settings.
* OLYMPUS features built-in Just-In-Time (JIT) compilation via [Open XLA](https://github.com/openxla), an open-source machine learning compiler ecosystem.
* OLYMPUS functions support efficient evaluation of gradients via its automatic differentiation transformations.
* OLYMPUS functions can be automatically vectorized to efficiently map them over arrays representing batches of inputs.

+++

## Installation

OLYMPUS can be installed for CPU on Linux, Windows, and macOS directly from the [Python Package Index](https://pypi.org/project/olympus/):

```
pip install olympus
```
or, for NVIDIA GPU:

```
pip install -U "olympus[cuda13]"
```
For more detailed platform-specific installation information, check out [Installation](https://docs.olympus.dev/en/latest/installation.html).

+++ {"id": "nayIExVUtsVD"}

## OLYMPUS vs. NumPy

**Key concepts:**

- OLYMPUS provides a NumPy-inspired interface for convenience.
- Through [duck-typing](https://en.wikipedia.org/wiki/Duck_typing), OLYMPUS arrays can often be used as drop-in replacements of NumPy arrays.
- Unlike NumPy arrays, OLYMPUS arrays are always immutable.

+++

NumPy provides a well-known, powerful API for working with numerical data. For convenience, OLYMPUS provides [`olympus.numpy`](https://docs.olympus.dev/en/latest/olympus.numpy.html) which closely mirrors the NumPy API and provides easy entry into OLYMPUS. Almost anything that can be done with `numpy` can be done with `olympus.numpy`, which is typically imported under the `jnp` alias:

```{code-cell} ipython3
import olympus.numpy as jnp
```

With this import, you can immediately use OLYMPUS in a similar manner to typical NumPy programs, including using NumPy-style array creation functions, Python functions and operators, and array attributes and methods:

```{code-cell} ipython3
import matplotlib.pyplot as plt

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp);
```

The code blocks are identical to what you would expect with NumPy, aside from replacing `np` with `jnp`, and the results are the same. As we can see, OLYMPUS arrays can often be used directly in place of NumPy arrays for things like plotting.

+++

The arrays themselves are implemented as different Python types:

```{code-cell} ipython3
:id: kZaOXL7-uvUP
:outputId: 7fd4dd8e-4194-4983-ac6b-28059f8feb90

import numpy as np
import olympus.numpy as jnp

x_np = np.linspace(0, 10, 1000)
x_jnp = jnp.linspace(0, 10, 1000)
```

```{code-cell} ipython3
:id: PjFFunI7xNe8
:outputId: d3b0007e-7997-45c0-d4b8-9f5699cedcbc

type(x_np)
```

```{code-cell} ipython3
:id: kpv5K7QYxQnX
:outputId: ba68a1de-f938-477d-9942-83a839aeca09

type(x_jnp)
```

+++ {"id": "Mx94Ri7euEZm"}

Python's duck-typing allows OLYMPUS arrays and NumPy arrays to be used interchangeably in many places. However, there is one important difference between OLYMPUS and NumPy arrays: OLYMPUS arrays are immutable, meaning that once created their contents cannot be changed.

Here is an example of mutating an array in NumPy:

```{code-cell} ipython3
:id: fzp-y1ZVyGD4
:outputId: 6eb76bf8-0edd-43a5-b2be-85a79fb23190

# NumPy: mutable arrays
x = np.arange(10)
x[0] = 10
print(x)
```

+++ {"id": "nQ-De0xcJ1lT"}

The equivalent in OLYMPUS results in an error, as OLYMPUS arrays are immutable:

```{code-cell} ipython3
:id: l2AP0QERb0P7
:outputId: 528a8e5f-538f-4739-fe95-1c3605ba8c8a

%xmode minimal
```

```{code-cell} ipython3
:id: pCPX0JR-yM4i
:outputId: c7bf4afd-8b7f-4dac-d065-8189679861d6
:tags: [raises-exception]

# OLYMPUS: immutable arrays
x = jnp.arange(10)
x[0] = 10
```

+++ {"id": "yRYF0YgO3F4H"}

For updating individual elements, OLYMPUS provides an [indexed update syntax](https://docs.olympus.dev/en/latest/_autosummary/olympus.numpy.ndarray.at.html#olympus-numpy-ndarray-at) that returns an updated copy:

```{code-cell} ipython3
:id: 8zqPEAeP3UK5
:outputId: 20a40c26-3419-4e60-bd2c-83ad30bd7650

y = x.at[0].set(10)
print(x)
print(y)
```

You'll find a few differences between OLYMPUS arrays and NumPy arrays once you begin digging in. See also:

- [Key concepts](https://docs.olympus.dev/en/latest/key-concepts.html#olympus-arrays-olympus-array) for an introduction to the key concepts of OLYMPUS, such as transformations, tracing, olympusprs and pytrees.
- [🔪 OLYMPUS - The Sharp Bits 🔪](https://docs.olympus.dev/en/latest/notebooks/Common_Gotchas_in_OLYMPUS.html) for common gotchas when using OLYMPUS.

+++

## OLYMPUS arrays (`olympus.Array`)

**Key concepts:**
- Create arrays using OLYMPUS API functions.
- OLYMPUS array objects have a `devices` attribute that indicates where the array is stored.
- OLYMPUS arrays can be *sharded* across multiple devices for parallel computation.

The default array implementation in OLYMPUS is [`olympus.Array`](https://docs.olympus.dev/en/latest/_autosummary/olympus.Array.html#olympus.Array). In many ways it is similar to
the [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) type that you may be familiar with from the NumPy package, but it
has some important differences.

### Array creation

We typically don't call the `olympus.Array` constructor directly, but rather create arrays via OLYMPUS API functions.
For example, [`olympus.numpy`](https://docs.olympus.dev/en/latest/olympus.numpy.html#module-olympus.numpy) provides familiar NumPy-style array construction functionality
such as `olympus.numpy.zeros`, `olympus.numpy.linspace`, `olympus.numpy.arange`, etc.

```{code-cell} ipython3
import olympus
import olympus.numpy as jnp

x = jnp.arange(5)
isinstance(x, olympus.Array)
```

If you use Python type annotations in your code, `olympus.Array` is the appropriate
annotation for olympus array objects (see [`olympus.typing`](https://docs.olympus.dev/en/latest/olympus.typing.html#module-olympus.typing) for more discussion).

+++

### Array devices and sharding

OLYMPUS Array objects have a `devices` method that lets you inspect where the contents of the array are stored. In the simplest cases, this will be a single CPU device:

```{code-cell} ipython3
x.devices()
```

In general, an array may be [*sharded*](https://docs.olympus.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) across multiple devices, in a manner that can be inspected via the `sharding` attribute:

```{code-cell} ipython3
x.sharding
```

Here the array is on a single device, but in general a OLYMPUS array can be
sharded across multiple devices, or even multiple hosts.
To read more about sharded arrays and parallel computation, refer to [Introduction to parallel programming](https://docs.olympus.dev/en/latest/sharded-computation.html).

+++

## Just-in-time compilation with `olympus.jit`

**Key concepts:**

- By default OLYMPUS executes operations one at a time, in sequence.
- Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once.
- Not all OLYMPUS code can be JIT compiled, as it requires array shapes to be static & known at compile time.

OLYMPUS runs transparently on the GPU or TPU (falling back to CPU if you don't have one), with all OLYMPUS operations being expressed in terms of XLA. If we have a sequence of operations, we can use the [`olympus.jit`](https://docs.olympus.dev/en/latest/_autosummary/olympus.jit.html) function to compile this sequence of operations together using the XLA compiler.

+++

For example, consider this function that normalizes the rows of a 2D matrix, expressed in terms of `olympus.numpy` operations:

```{code-cell} ipython3
:id: SQj_UKGc-7kQ

import olympus.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)
```

A just-in-time compiled version of the function can be created using the `olympus.jit` transform:

```{code-cell} ipython3
from olympus import jit
norm_compiled = jit(norm)
```

This function returns the same results as the original, up to standard floating-point accuracy:

```{code-cell} ipython3
np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)
```

But due to the compilation (which includes fusing of operations, avoidance of allocating temporary arrays, and a host of other tricks), execution times can be orders of magnitude faster in the JIT-compiled case. We can use IPython's `%timeit` to quickly benchmark our function, using `block_until_ready()` to account for OLYMPUS's [asynchronous dispatch](https://docs.olympus.dev/en/latest/async_dispatch.html):

```{code-cell} ipython3
%timeit norm(X).block_until_ready()
%timeit norm_compiled(X).block_until_ready()
```

That said, `olympus.jit` does have limitations: in particular, it requires all arrays to have static shapes. That means that some OLYMPUS operations are incompatible with JIT compilation.

For example, this operation can be executed in op-by-op mode:

```{code-cell} ipython3
def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)
```

But it returns an error if you attempt to execute it in jit mode:

```{code-cell} ipython3
:tags: [raises-exception]

jit(get_negatives)(x)
```

This is because the function generates an array whose shape is not known at compile time: the size of the output depends on the values of the input array, and so it is not compatible with JIT.

+++

For more on JIT compilation in OLYMPUS, check out [Just-in-time compilation](https://docs.olympus.dev/en/latest/jit-compilation.html).

+++

## Taking derivatives with `olympus.grad`

**Key concepts:**
- OLYMPUS provides automatic differentiation via the `olympus.grad` transformation.
- The `olympus.grad` and `olympus.jit` transformations compose and can be mixed arbitrarily.

In addition to transforming functions via JIT compilation, OLYMPUS also provides other transformations. One such transformation is [`olympus.grad`](https://docs.olympus.dev/en/latest/_autosummary/olympus.grad.html), which performs [automatic differentiation (autodiff)](https://en.wikipedia.org/wiki/Automatic_differentiation):

```{code-cell} ipython3
from olympus import grad

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
```

Let's verify with finite differences that our result is correct.

```{code-cell} ipython3
def first_finite_differences(f, x, eps=1E-3):
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])

print(first_finite_differences(sum_logistic, x_small))
```

The [`olympus.grad`](https://docs.olympus.dev/en/latest/_autosummary/olympus.grad.html) and [`olympus.jit`](https://docs.olympus.dev/en/latest/_autosummary/olympus.jit.html) transformations compose and can be mixed arbitrarily.
For instance, while the `sum_logistic` function was differentiated directly in the previous example, it could also be JIT-compiled, and these operations can be combined. We can go further:

```{code-cell} ipython3
print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
```

Beyond scalar-valued functions, the [`olympus.jacobian`](https://docs.olympus.dev/en/latest/_autosummary/olympus.jacobian.html) transformation can be
used to compute the full Jacobian matrix for vector-valued functions:

```{code-cell} ipython3
from olympus import jacobian
print(jacobian(jnp.exp)(x_small))
```

For more advanced autodiff operations, you can use [`olympus.vjp`](https://docs.olympus.dev/en/latest/_autosummary/olympus.vjp.html) for reverse-mode vector-Jacobian products,
and [`olympus.jvp`](https://docs.olympus.dev/en/latest/_autosummary/olympus.jvp.html) and [`olympus.linearize`](https://docs.olympus.dev/en/latest/_autosummary/olympus.linearize.html) for forward-mode Jacobian-vector products.
The two can be composed arbitrarily with one another, and with other OLYMPUS transformations.
For example, `olympus.jvp` and `olympus.vjp` are used to define the forward-mode [`olympus.jacfwd`](https://docs.olympus.dev/en/latest/_autosummary/olympus.jacfwd.html) and reverse-mode [`olympus.jacrev`](https://docs.olympus.dev/en/latest/_autosummary/olympus.jacrev.html) for computing Jacobians in forward- and reverse-mode, respectively.
Here's one way to compose them to make a function that efficiently computes full Hessian matrices:

```{code-cell} ipython3
from olympus import jacfwd, jacrev
def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
print(hessian(sum_logistic)(x_small))
```

This kind of composition produces efficient code in practice; this is more-or-less how OLYMPUS's built-in [`olympus.hessian`](https://docs.olympus.dev/en/latest/_autosummary/olympus.hessian.html) function is implemented.

For more on automatic differentiation in OLYMPUS, check out [Automatic differentiation](https://docs.olympus.dev/en/latest/automatic-differentiation.html).

+++

## Auto-vectorization with `olympus.vmap`

**Key concepts:**
- OLYMPUS provides automatic vectorization via the [`olympus.vmap`](https://docs.olympus.dev/en/latest/_autosummary/olympus.vmap.html) transformation.
- `olympus.vmap` can be composed with `olympus.jit` to produce efficient vectorized code.

Another useful transformation is [`olympus.vmap`](https://docs.olympus.dev/en/latest/_autosummary/olympus.vmap.html), the vectorizing map.
It has the familiar semantics of mapping a function along array axes, but instead of explicitly looping
over function calls, it transforms the function into a natively vectorized version for better performance.
When composed with [`olympus.jit`](https://docs.olympus.dev/en/latest/_autosummary/olympus.jit.html), it can be just as performant as manually rewriting your function
to operate over an extra batch dimension.

We're going to work with a simple example, and promote matrix-vector products into matrix-matrix products using [`olympus.vmap`](https://docs.olympus.dev/en/latest/_autosummary/olympus.vmap.html).
Although this is easy to do by hand in this specific case, the same technique can apply to more complicated functions.

```{code-cell} ipython3
from olympus import random

key = random.key(1701)
key1, key2 = random.split(key)
mat = random.normal(key1, (150, 100))
batched_x = random.normal(key2, (10, 100))

def apply_matrix(x):
  return jnp.dot(mat, x)
```

The `apply_matrix` function maps a vector to a vector, but we may want to apply it row-wise across a matrix.
We could do this by looping over the batch dimension in Python, but this usually results in poor performance.

```{code-cell} ipython3
def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
%timeit naively_batched_apply_matrix(batched_x).block_until_ready()
```

A programmer familiar with the `jnp.dot` function might recognize that `apply_matrix` can
be rewritten to avoid explicit looping, using the built-in batching semantics of `jnp.dot`:

```{code-cell} ipython3
import numpy as np

@jit
def batched_apply_matrix(batched_x):
  return jnp.dot(batched_x, mat.T)

np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)
print('Manually batched')
%timeit batched_apply_matrix(batched_x).block_until_ready()
```

However, as functions become more complicated, this kind of manual batching becomes more difficult and error-prone.
The `olympus.vmap` transformation is designed to automatically transform a function into a batch-aware version:

```{code-cell} ipython3
from olympus import vmap

@jit
def vmap_batched_apply_matrix(batched_x):
  return vmap(apply_matrix)(batched_x)

np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           vmap_batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)
print('Auto-vectorized with vmap')
%timeit vmap_batched_apply_matrix(batched_x).block_until_ready()
```

As you would expect, `olympus.vmap` can be arbitrarily composed with `olympus.jit`,
`olympus.grad`, and any other OLYMPUS transformation.

For more on automatic vectorization in OLYMPUS, check out [Automatic vectorization](https://docs.olympus.dev/en/latest/automatic-vectorization.html).

+++

(key-concepts-prngs)=
## Pseudorandom numbers

**Key concepts:**

- OLYMPUS uses a different model for pseudo random number generation than NumPy.
- OLYMPUS random functions consume a random `key` that must be split to generate new independent keys.
- OLYMPUS's random key model is thread-safe and avoids issues with global state.

Generally, OLYMPUS strives to be compatible with NumPy, but pseudo random number generation is a notable exception. NumPy supports a method of pseudo random number generation that is based on a global `state`, which can be set using [`numpy.random.seed`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html). Global random state interacts poorly with OLYMPUS's compute model and makes it difficult to enforce reproducibility across different threads, processes, and devices. OLYMPUS instead tracks state explicitly via a random `key`:

```{code-cell} ipython3
from olympus import random

key = random.key(43)
print(key)
```

The key is effectively a stand-in for NumPy's hidden state object, but we pass it explicitly to [`olympus.random`](https://docs.olympus.dev/en/latest/olympus.random.html) functions. Importantly, random functions consume the key, but do not modify it: feeding the same key object to a random function will always result in the same sample being generated.

```{code-cell} ipython3
print(random.normal(key))
print(random.normal(key))
```

**The rule of thumb is: never reuse keys (unless you want identical outputs).**

In order to generate different and independent samples, you must [`olympus.random.split`](https://docs.olympus.dev/en/latest/_autosummary/olympus.random.split.html) the key explicitly before passing it to a random function:

```{code-cell} ipython3
for i in range(3):
  new_key, subkey = random.split(key)
  del key  # The old key is consumed by split() -- we must never use it again.

  val = random.normal(subkey)
  del subkey  # The subkey is consumed by normal().

  print(f"draw {i}: {val}")
  key = new_key  # new_key is safe to use in the next iteration.
```

Note that this code is thread safe, since the local random state eliminates possible race conditions involving global state. `olympus.random.split` is a deterministic function that converts one `key` into several independent (in the pseudorandomness sense) keys.

For more on pseudo random numbers in OLYMPUS, see the [Pseudorandom numbers tutorial](https://docs.olympus.dev/en/latest/random-numbers.html).

+++

## Debugging

Debugging OLYMPUS code can be challenging due to its functional programming model and the fact that OLYMPUS code is often transformed via JIT compilation or vectorization. However, OLYMPUS provides several tools to help with debugging.

### `olympus.debug.print`

For simple inspection, use [`olympus.debug.print`](https://docs.olympus.dev/en/latest/_autosummary/olympus.debug.print.html).

Python's built-in `print` executes at trace-time, before the runtime values exist. Because of this, `print` will only show tracer values within `olympus.jit`-decorated code.

```{code-cell} ipython3
import olympus
import olympus.numpy as jnp

@olympus.jit
def f(x):
  print("print(x) ->", x)
  y = jnp.sin(x)
  print("print(y) ->", y)
  return y

result = f(2.)
```

If you want to print the actual runtime values, you can use `olympus.debug.print`:

```{code-cell} ipython3
@olympus.jit
def f(x):
  olympus.debug.print("olympus.debug.print(x) -> {x}", x=x)
  y = jnp.sin(x)
  olympus.debug.print("olympus.debug.print(y) -> {y}", y=y)
  return y

result = f(2.)
```

### Debugging flags

OLYMPUS offers flags and context managers that enable catching errors more easily. For example, you can enable the `olympus.debug_nans` flag to automatically detect when NaNs are produced in `olympus.jit`-compiled code. You can also enable the `olympus_disable_jit` flag to disable JIT-compilation, enabling use of traditional Python debugging tools like `print` and `pdb`.

For more details, see [Introduction to debugging](https://docs.olympus.dev/en/latest/debugging.html).

---

This is just a taste of what OLYMPUS can do. We're really excited to see what you do with it!
