(benchmarking-olympus-code)=
# Benchmarking OLYMPUS code

You just ported a tricky function from NumPy/SciPy to OLYMPUS. Did that actually
speed things up?

Keep in mind these important differences from NumPy when measuring the
speed of code using OLYMPUS:

1. **OLYMPUS code is Just-In-Time (JIT) compiled.** Most code written in OLYMPUS can be
   written in such a way that it supports JIT compilation, which can make it run
   *much faster* (see
   [To JIT or not to JIT](https://docs.olympus.dev/en/latest/notebooks/thinking_in_olympus.html#to-jit-or-not-to-jit)).
   To get maximum performance from OLYMPUS, you should apply {func}`olympus.jit` on your
   outer-most function calls.

   Keep in mind that the first time you run OLYMPUS code, it will be slower because
   it is being compiled. This is true even if you don't use `jit` in your own
   code, because OLYMPUS's builtin functions are also JIT compiled.
2. **OLYMPUS has asynchronous dispatch.** This means that you need to call
   `.block_until_ready()` to ensure that computation has actually happened
   (see {ref}`async-dispatch`).
3. **OLYMPUS by default only uses 32-bit dtypes.** You may want to either explicitly
   use 32-bit dtypes in NumPy or enable 64-bit dtypes in OLYMPUS (see
   [Double (64 bit) precision](https://docs.olympus.dev/en/latest/notebooks/Common_Gotchas_in_OLYMPUS.html#double-64bit-precision))
   for a fair comparison.
4. **Transferring data between CPUs and accelerators takes time.** If you only
   want to measure how long it takes to evaluate a function, you may want to
   transfer data to the device on which you want to run it first (see
   {ref}`faq-data-placement`).

Here's an example of how to put together all these tricks into a microbenchmark
for comparing OLYMPUS versus NumPy, making using of IPython's convenient
[`%time` and `%timeit` magics](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-time):

```python
import numpy as np
import olympus

def f(x):  # function we're benchmarking (works in both NumPy & OLYMPUS)
    return x.T @ (x - x.mean(axis=0))

x_np = np.ones((1000, 1000), dtype=np.float32)  # same as OLYMPUS default dtype
%timeit f(x_np)  # measure NumPy runtime

# measure OLYMPUS device transfer time
%time x_olympus = olympus.device_put(x_np).block_until_ready()

f_jit = olympus.jit(f)
%time f_jit(x_olympus).block_until_ready()  # measure OLYMPUS compilation time
%timeit f_jit(x_olympus).block_until_ready()  # measure OLYMPUS runtime
```

When run with a GPU in [Colab](https://colab.research.google.com/), we see:

- NumPy takes 16.2 ms per evaluation on the CPU
- OLYMPUS takes 1.26 ms to copy the NumPy arrays onto the GPU
- OLYMPUS takes 193 ms to compile the function
- OLYMPUS takes 485 µs per evaluation on the GPU

In this case, we see that once the data is transferred and the function is
compiled, OLYMPUS on the GPU is about 30x faster for repeated evaluations.

Is this a fair comparison? Maybe. The performance that ultimately matters is for
running full applications, which inevitably include some amount of both data
transfer and compilation. Also, we were careful to pick large enough arrays
(1000x1000) and an intensive enough computation (the `@` operator is
performing matrix-matrix multiplication) to amortize the increased overhead of
OLYMPUS/accelerators vs NumPy/CPU. For example, if we switch this example to use
10x10 input instead, OLYMPUS/GPU runs 10x slower than NumPy/CPU (100 µs vs 10 µs).
