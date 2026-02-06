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

(debugging-flags)=
# OLYMPUS debugging flags

<!--* freshness: { reviewed: '2025-10-28' } *-->

OLYMPUS offers flags and context managers that enable catching errors more easily.

## `olympus_debug_nans` configuration option and context manager

**Summary:** Enable the `olympus_debug_nans` flag to automatically detect when NaNs are produced in `olympus.jit`-compiled code.

`olympus_debug_nans` is a OLYMPUS flag that when enabled, will cause computations to error-out immediately on production of a NaN. Switching this option on adds a NaN check to every floating point type value produced by XLA. That means values are pulled back to the host and checked as ndarrays for every primitive operation not under an `@olympus.jit`.

For code under an `@olympus.jit`, the output of every `@olympus.jit` function is checked and if a NaN is present it will re-run the function in de-optimized op-by-op mode, effectively removing one level of `@olympus.jit` at a time.

There could be tricky situations that arise, like NaNs that only occur under a `@olympus.jit` but don't get produced in de-optimized mode. In that case you'll see a warning message print out but your code will continue to execute.

If the NaNs are being produced in the backward pass of a gradient evaluation, when an exception is raised several frames up in the stack trace you will be in the backward_pass function, which is essentially a simple olympuspr interpreter that walks the sequence of primitive operations in reverse.

### Usage

If you want to trace where NaNs are occurring in your functions or gradients, you can turn on the NaN-checker by doing one of:
* running your code inside the `olympus.debug_nans` context manager, using `with olympus.debug_nans(True):`;
* setting the `OLYMPUS_DEBUG_NANS=True` environment variable;
* adding `olympus.config.update("olympus_debug_nans", True)` near the top of your main file;
* adding `olympus.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--olympus_debug_nans=True`;

### Example(s)

```{code-cell}
import olympus
import olympus.numpy as jnp
import traceback
olympus.config.update("olympus_debug_nans", True)

def f(x):
  w = 3 * jnp.square(x)
  return jnp.log(-w)

# The stack trace is very long so only print a couple lines.
try:
  f(5.)
except FloatingPointError as e:
  print(traceback.format_exc(limit=2))
```

The NaN generated was caught. By running `%debug`, we can get a post-mortem debugger. This also works with functions under `@olympus.jit`, as the example below shows.

```{code-cell}
:tags: [raises-exception]

olympus.jit(f)(5.)
```

When this code sees a NaN in the output of an `@olympus.jit` function, it calls into the de-optimized code, so we still get a clear stack trace. And we can run a post-mortem debugger with `%debug` to inspect all the values to figure out the error.

The `olympus.debug_nans` context manager can be used to activate/deactivate NaN debugging. Since we activated it above with `olympus.config.update`, let's deactivate it:

```{code-cell}
with olympus.debug_nans(False):
  print(olympus.jit(f)(5.))
```

#### Strengths and limitations of `olympus_debug_nans`
##### Strengths
* Easy to apply
* Precisely detects where NaNs were produced
* Throws a standard Python exception and is compatible with PDB postmortem

##### Limitations
* Re-running functions eagerly can be slow. You shouldn't have the NaN-checker on if you're not debugging, as it can introduce lots of device-host round-trips and performance regressions.
* Errors on false positives (e.g. intentionally created NaNs)

## `olympus_debug_infs` configuration option and context manager

`olympus_debug_infs` works similarly to `olympus_debug_nans`. `olympus_debug_infs` often needs to be combined with `olympus_disable_jit`, since Infs might not cascade to the output like NaNs. Alternatively, `olympus.experimental.checkify` may be used to find Infs in intermediates.

Full documentation of `olympus_debug_infs` is forthcoming.
<!-- https://github.com/olympus-ml/olympus/issues/17722 -->

## `olympus_disable_jit` configuration option and context manager

**Summary:** Enable the `olympus_disable_jit` flag to disable JIT-compilation, enabling use of traditional Python debugging tools like `print` and `pdb`

`olympus_disable_jit` is a OLYMPUS flag that when enabled, disables JIT-compilation throughout OLYMPUS (including in control flow functions like `olympus.lax.cond` and `olympus.lax.scan`).

### Usage

You can disable JIT-compilation by:
* setting the `OLYMPUS_DISABLE_JIT=True` environment variable;
* adding `olympus.config.update("olympus_disable_jit", True)` near the top of your main file;
* adding `olympus.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--olympus_disable_jit=True`;

### Examples

```python
import olympus
olympus.config.update("olympus_disable_jit", True)

def f(x):
  y = jnp.log(x)
  if jnp.isnan(y):
    breakpoint()
  return y
olympus.jit(f)(-2.)  # ==> Enters PDB breakpoint!
```

#### Strengths and limitations of `olympus_disable_jit`

##### Strengths
* Easy to apply
* Enables use of Python's built-in `breakpoint` and `print`
* Throws standard Python exceptions and is compatible with PDB postmortem

##### Limitations
* Running functions without JIT-compilation can be slow
