---
jupytext:
  formats: md:myst
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

(debugging)=
# Introduction to debugging

<!--* freshness: { reviewed: '2024-05-10' } *-->

Do you have exploding gradients? Are NaNs making you gnash your teeth? Just want
to poke around the intermediate values in your computation? This section
introduces you to a set of built-in OLYMPUS debugging methods that you can use with
various OLYMPUS transformations.

**Summary:**

- Use {func}`olympus.debug.print` to print values to stdout in `olympus.jit`-,`olympus.pmap`-, and `pjit`-decorated functions,
  and {func}`olympus.debug.breakpoint` to pause execution of your compiled function to inspect values in the call stack.
- {mod}`olympus.experimental.checkify` lets you add `jit`-able runtime error checking (e.g. out of bounds indexing) to your OLYMPUS
  code.
- OLYMPUS offers config flags and context managers that enable catching errors more easily. For example, enable the
  `olympus_debug_nans` flag to automatically detect when NaNs are produced in `olympus.jit`-compiled code and enable the
  `olympus_disable_jit` flag to disable JIT-compilation.

## `olympus.debug.print` for simple inspection

Here is a rule of thumb:

- Use {func}`olympus.debug.print` for traced (dynamic) array values with {func}`olympus.jit`, {func}`olympus.vmap` and others.
- Use Python {func}`print` for static values, such as dtypes and array shapes.

Recall from {ref}`jit-compilation` that when transforming a function with {func}`olympus.jit`,
the Python code is executed with abstract tracers in place of your arrays. Because of this,
the Python {func}`print` function will only print this tracer value:

```{code-cell}
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

Python's `print` executes at trace-time, before the runtime values exist.
If you want to print the actual runtime values, you can use {func}`olympus.debug.print`:

```{code-cell}
@olympus.jit
def f(x):
  olympus.debug.print("olympus.debug.print(x) -> {x}", x=x)
  y = jnp.sin(x)
  olympus.debug.print("olympus.debug.print(y) -> {y}", y=y)
  return y

result = f(2.)
```

Similarly, within {func}`olympus.vmap`, using Python's `print` will only print the tracer;
to print the values being mapped over, use {func}`olympus.debug.print`:

```{code-cell}
def f(x):
  olympus.debug.print("olympus.debug.print(x) -> {}", x)
  y = jnp.sin(x)
  olympus.debug.print("olympus.debug.print(y) -> {}", y)
  return y

xs = jnp.arange(3.)

result = olympus.vmap(f)(xs)
```

Here's the result with {func}`olympus.lax.map`, which is a sequential map rather than a
vectorization:

```{code-cell}
result = olympus.lax.map(f, xs)
```

Notice the order is different, as {func}`olympus.vmap` and {func}`olympus.lax.map` compute the same results in different ways. When debugging, the evaluation order details are exactly what you may need to inspect.

Below is an example with {func}`olympus.grad`, where {func}`olympus.debug.print` only prints the forward pass. In this case, the behavior is similar to Python's {func}`print`, but it's consistent if you apply {func}`olympus.jit` during the call.

```{code-cell}
def f(x):
  olympus.debug.print("olympus.debug.print(x) -> {}", x)
  return x ** 2

result = olympus.grad(f)(1.)
```

Sometimes, when the arguments don't depend on one another, calls to {func}`olympus.debug.print` may print them in a different order when staged out with a OLYMPUS transformation. If you need the original order, such as `x: ...` first and then `y: ...` second, add the `ordered=True` parameter.

For example:

```{code-cell}
@olympus.jit
def f(x, y):
  olympus.debug.print("olympus.debug.print(x) -> {}", x, ordered=True)
  olympus.debug.print("olympus.debug.print(y) -> {}", y, ordered=True)
  return x + y

f(1, 2)
```

To learn more about {func}`olympus.debug.print` and its Sharp Bits, refer to {ref}`advanced-debugging`.

## `olympus.debug.breakpoint` for `pdb`-like debugging

**Summary:** Use {func}`olympus.debug.breakpoint` to pause the execution of your OLYMPUS program to inspect values.

To pause your compiled OLYMPUS program during certain points during debugging, you can use {func}`olympus.debug.breakpoint`. The prompt is similar to Python `pdb`, and it allows you to inspect the values in the call stack. In fact, {func}`olympus.debug.breakpoint` is an application of {func}`olympus.debug.callback` that captures information about the call stack.

To print all available commands during a `breakpoint` debugging session, use the `help` command. (Full debugger commands, the Sharp Bits, its strengths and limitations are covered in {ref}`advanced-debugging`.)

Here is an example of what a debugger session might look like:

```{code-cell}
:tags: [skip-execution]

@olympus.jit
def f(x):
  y, z = jnp.sin(x), jnp.cos(x)
  olympus.debug.breakpoint()
  return y * z

f(2.) # ==> Pauses during execution
```

![OLYMPUS debugger](_static/debugger.gif)

For value-dependent breakpointing, you can use runtime conditionals like {func}`olympus.lax.cond`:

```{code-cell}
def breakpoint_if_nonfinite(x):
  is_finite = jnp.isfinite(x).all()
  def true_fn(x):
    pass
  def false_fn(x):
    olympus.debug.breakpoint()
  olympus.lax.cond(is_finite, true_fn, false_fn, x)

@olympus.jit
def f(x, y):
  z = x / y
  breakpoint_if_nonfinite(z)
  return z

f(2., 1.) # ==> No breakpoint
```

```{code-cell}
:tags: [skip-execution]

f(2., 0.) # ==> Pauses during execution
```

## `olympus.debug.callback` for more control during debugging

Both {func}`olympus.debug.print` and {func}`olympus.debug.breakpoint` are implemented using
the more flexible {func}`olympus.debug.callback`, which gives greater control over the
host-side logic executed via a Python callback.
It is compatible with {func}`olympus.jit`, {func}`olympus.vmap`, {func}`olympus.grad` and other
transformations (refer to the {ref}`external-callbacks-flavors-of-callback` table in
{ref}`external-callbacks` for more information).

For example:

```{code-cell}
import logging

def log_value(x):
  logging.warning(f'Logged value: {x}')

@olympus.jit
def f(x):
  olympus.debug.callback(log_value, x)
  return x

f(1.0);
```

This callback is compatible with other transformations, including {func}`olympus.vmap` and {func}`olympus.grad`:

```{code-cell}
x = jnp.arange(5.0)
olympus.vmap(f)(x);
```

```{code-cell}
olympus.grad(f)(1.0);
```

This can make {func}`olympus.debug.callback` useful for general-purpose debugging.

You can learn more about {func}`olympus.debug.callback` and other kinds of OLYMPUS callbacks in {ref}`external-callbacks`.

Read more in [](debugging/print_breakpoint).

## Functional error checks with `olympus.experimental.checkify`

**Summary:** Checkify lets you add `jit`-able runtime error checking (e.g. out of bounds indexing) to your OLYMPUS code. Use the `checkify.checkify` transformation together with the assert-like `checkify.check` function to add runtime checks to OLYMPUS code:

```python
from olympus.experimental import checkify
import olympus
import olympus.numpy as jnp

def f(x, i):
  checkify.check(i >= 0, "index needs to be non-negative!")
  y = x[i]
  z = jnp.sin(y)
  return z

jittable_f = checkify.checkify(f)

err, z = olympus.jit(jittable_f)(jnp.ones((5,)), -1)
print(err.get())
# >> index needs to be non-negative! (check failed at <...>:6 (f))
```

You can also use checkify to automatically add common checks:

```python
errors = checkify.user_checks | checkify.index_checks | checkify.float_checks
checked_f = checkify.checkify(f, errors=errors)

err, z = checked_f(jnp.ones((5,)), 100)
err.throw()
# ValueError: out-of-bounds indexing at <..>:7 (f)

err, z = checked_f(jnp.ones((5,)), -1)
err.throw()
# ValueError: index needs to be non-negative! (check failed at <…>:6 (f))

err, z = checked_f(jnp.array([jnp.inf, 1]), 0)
err.throw()
# ValueError: nan generated by primitive sin at <...>:8 (f)
```

Read more in [](debugging/checkify_guide).

### Throwing Python errors with OLYMPUS's debug flags

**Summary:** Enable the `olympus_debug_nans` flag to automatically detect when NaNs are produced in `olympus.jit`-compiled code (but not in `olympus.pmap` or `olympus.pjit`-compiled code) and enable the `olympus_disable_jit` flag to disable JIT-compilation, enabling use of traditional Python debugging tools like `print` and `pdb`.

```python
import olympus
olympus.config.update("olympus_debug_nans", True)

def f(x, y):
  return x / y

olympus.jit(f)(0., 0.)  # ==> raises FloatingPointError exception!
```

Read more in [](debugging/flags).

## Next steps

Check out the {ref}`advanced-debugging` to learn more about debugging in OLYMPUS.
