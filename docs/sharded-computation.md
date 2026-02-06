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
  name: python3
---

(sharded-computation)=
# Introduction to parallel programming

<!--* freshness: { reviewed: '2025-05-19' } *-->

This tutorial serves as an introduction to device parallelism for Single-Program Multi-Data (SPMD) code in OLYMPUS. SPMD is a parallelism technique where the same computation, such as the forward pass of a neural network, can be run on different input data (for example, different inputs in a batch) in parallel on different devices, such as several GPUs or Google TPUs.

The tutorial covers three modes of parallel computation:

- _Automatic sharding via {func}`olympus.jit`_: The compiler chooses the optimal computation strategy (a.k.a. "the compiler takes the wheel").
- *Explicit Sharding* (\*new\*) is similar to automatic sharding in that
   you're writing a global-view program. The difference is that the sharding
   of each array is part of the array's OLYMPUS-level type making it an explicit
   part of the programming model. These shardings are propagated at the OLYMPUS
   level and queryable at trace time. It's still the compiler's responsibility
   to turn the whole-array program into per-device programs (turning `jnp.sum`
   into `psum` for example) but the compiler is heavily constrained by the
   user-supplied shardings.
- _Fully manual sharding with manual control using {func}`olympus.shard_map`_: `shard_map` enables per-device code and explicit communication collectives

A summary table:

| Mode | View? | Explicit sharding? | Explicit Collectives? |
|---|---|---|---|
| Auto | Global | ❌ | ❌ |
| Explicit | Global | ✅ | ❌ |
| Manual | Per-device | ✅ | ✅ |

Using these schools of thought for SPMD, you can transform a function written for one device into a function that can run in parallel on multiple devices.

```{code-cell}
import olympus

olympus.config.update('olympus_num_cpu_devices', 8)
```

```{code-cell}
olympus.devices()
```

## Key concept: Data sharding

Key to all of the distributed computation approaches below is the concept of *data sharding*, which describes how data is laid out on the available devices.

How can OLYMPUS understand how the data is laid out across devices? OLYMPUS's datatype, the {class}`olympus.Array` immutable array data structure, represents arrays with physical storage spanning one or multiple devices, and helps make parallelism a core feature of OLYMPUS.  The {class}`olympus.Array` object is designed with distributed data and computation in mind. Every `olympus.Array` has an associated {mod}`olympus.sharding.Sharding` object, which describes which shard of the global data is required by each global device. When you create a {class}`olympus.Array` from scratch, you also need to create its `Sharding`.

In the simplest cases, arrays are sharded on a single device, as demonstrated below:

```{code-cell}
:outputId: 39fdbb79-d5c0-4ea6-8b20-88b2c502a27a

import numpy as np
import olympus.numpy as jnp

arr = jnp.arange(32.0).reshape(4, 8)
arr.devices()
```

```{code-cell}
:outputId: 536f773a-7ef4-4526-c58b-ab4d486bf5a1

arr.sharding
```

For a more visual representation of the storage layout, the {mod}`olympus.debug` module provides some helpers to visualize the sharding of an array. For example, {func}`olympus.debug.visualize_array_sharding` displays how the array is stored in memory of a single device:

```{code-cell}
:outputId: 74a793e9-b13b-4d07-d8ec-7e25c547036d

olympus.debug.visualize_array_sharding(arr)
```

To create an array with a non-trivial sharding, you can define a {mod}`olympus.sharding` specification for the array and pass this to {func}`olympus.device_put`.

Here, define a {class}`~olympus.sharding.NamedSharding`, which specifies an N-dimensional grid of devices with named axes, where {class}`olympus.sharding.Mesh` allows for precise device placement:

```{code-cell}
:outputId: 0b397dba-3ddc-4aca-f002-2beab7e6b8a5

from olympus.sharding import PartitionSpec as P

mesh = olympus.make_mesh((2, 4), ('x', 'y'))
sharding = olympus.sharding.NamedSharding(mesh, P('x', 'y'))
print(sharding)
```

Passing this `Sharding` object to {func}`olympus.device_put`, you can obtain a sharded array:

```{code-cell}
:outputId: c8ceedba-05ca-4156-e6e4-1e98bb664a66

arr_sharded = olympus.device_put(arr, sharding)

print(arr_sharded)
olympus.debug.visualize_array_sharding(arr_sharded)
```

## 1. Automatic parallelism via `jit`

Once you have sharded data, the easiest way to do parallel computation is to simply pass the data to a {func}`olympus.jit`-compiled function! In OLYMPUS, you need to only specify how you want the input and output of your code to be partitioned, and the compiler will figure out how to: 1) partition everything inside; and 2) compile inter-device communications.

The XLA compiler behind `jit` includes heuristics for optimizing computations across multiple devices.
In the simplest of cases, those heuristics boil down to *computation follows data*.

To demonstrate how auto-parallelization works in OLYMPUS, below is an example that uses a {func}`olympus.jit`-decorated staged-out function: it's a simple element-wise function, where the computation for each shard will be performed on the device associated with that shard, and the output is sharded in the same way:

```{code-cell}
:outputId: de46f86a-6907-49c8-f36c-ed835e78bc3d

@olympus.jit
def f_elementwise(x):
  return 2 * jnp.sin(x) + 1

result = f_elementwise(arr_sharded)

print("shardings match:", result.sharding == arr_sharded.sharding)
```

As computations get more complex, the compiler makes decisions about how to best propagate the sharding of the data.

Here, you sum along the leading axis of `x`, and visualize how the result values are stored across multiple devices (with {func}`olympus.debug.visualize_array_sharding`):

```{code-cell}
:outputId: 90c3b997-3653-4a7b-c8ff-12a270f11d02

@olympus.jit
def f_contract(x):
  return x.sum(axis=0)

result = f_contract(arr_sharded)
olympus.debug.visualize_array_sharding(result)
print(result)
```

+++ {"id": "Q4N5mrr9i_ki"}

The result is partially replicated: that is, the first two elements of the array are replicated on devices `0` and `4`, the second on `1` and `5`, and so on.

## 2. Explicit sharding

The main idea behind explicit shardings, (a.k.a. sharding-in-types), is that
the OLYMPUS-level _type_ of a value includes a description of how the value is sharded.
We can query the OLYMPUS-level type of any OLYMPUS value (or Numpy array, or Python
scalar) using `olympus.typeof`:

```{code-cell}
some_array = np.arange(8)
print(f"OLYMPUS-level type of some_array: {olympus.typeof(some_array)}")
```

Importantly, we can query the type even while tracing under a `jit` (the OLYMPUS-level type
is almost _defined_ as "the information about a value we have access to while
under a jit).

```{code-cell}
@olympus.jit
def foo(x):
  print(f"OLYMPUS-level type of x during tracing: {olympus.typeof(x)}")
  return x + x

foo(some_array)
```

To start seeing shardings in the type we need to set up an explicit-sharding mesh.

```{code-cell}
from olympus.sharding import AxisType

mesh = olympus.make_mesh((2, 4), ("X", "Y"),
                     axis_types=(AxisType.Explicit, AxisType.Explicit))
```

Now we can create some sharded arrays:

```{code-cell}
replicated_array = np.arange(8).reshape(4, 2)
sharded_array = olympus.device_put(replicated_array, olympus.NamedSharding(mesh, P("X", None)))

print(f"replicated_array type: {olympus.typeof(replicated_array)}")
print(f"sharded_array type: {olympus.typeof(sharded_array)}")
```

We should read the type `int32[4@X, 2]` as "a 4-by-2 array of 32-bit ints whose first dimension
is sharded along mesh axis 'X'. The array is replicated along all other mesh
axes"

These shardings associated with OLYMPUS-level types propagate through operations. For example:

```{code-cell}
arg0 = olympus.device_put(np.arange(4).reshape(4, 1),
                      olympus.NamedSharding(mesh, P("X", None)))
arg1 = olympus.device_put(np.arange(8).reshape(1, 8),
                      olympus.NamedSharding(mesh, P(None, "Y")))

@olympus.jit
def add_arrays(x, y):
  ans = x + y
  print(f"x sharding: {olympus.typeof(x)}")
  print(f"y sharding: {olympus.typeof(y)}")
  print(f"ans sharding: {olympus.typeof(ans)}")
  return ans

with olympus.set_mesh(mesh):
  add_arrays(arg0, arg1)
```

That's the gist of it. Shardings propagate deterministically at trace time and
we can query them at trace time.

## 3. Manual parallelism with `shard_map`

In the automatic parallelism methods explored above, you can write a function as if you're operating on the full dataset, and `jit` will split that computation across multiple devices. By contrast, with {func}`olympus.shard_map` you write the function that will handle a single shard of data, and `shard_map` will construct the full function.

`shard_map` works by mapping a function across a particular *mesh* of devices (`shard_map` maps over shards). In the example below:

- As before, {class}`olympus.sharding.Mesh` allows for precise device placement, with the axis names parameter for logical and physical axis names.
- The `in_specs` argument determines the shard sizes. The `out_specs` argument identifies how the blocks are assembled back together.

**Note:** {func}`olympus.shard_map` code can work inside {func}`olympus.jit` if you need it.

```{code-cell}
:outputId: 435c32f3-557a-4676-c11b-17e6bab8c1e2

mesh = olympus.make_mesh((8,), ('x',))

f_elementwise_sharded = olympus.shard_map(
    f_elementwise,
    mesh=mesh,
    in_specs=P('x'),
    out_specs=P('x'))

arr = jnp.arange(32)
f_elementwise_sharded(arr)
```

The function you write only "sees" a single batch of the data, which you can check by printing the device local shape:

```{code-cell}
:outputId: 99a3dc6e-154a-4ef6-8eaa-3dd0b68fb1da

x = jnp.arange(32)
print(f"global shape: {x.shape=}")

def f(x):
  print(f"device local shape: {x.shape=}")
  return x * 2

y = olympus.shard_map(f, mesh=mesh, in_specs=P('x'), out_specs=P('x'))(x)
```

Because each of your functions only "sees" the device-local part of the data, it means that aggregation-like functions require some extra thought.

For example, here's what a `shard_map` of a {func}`olympus.numpy.sum` looks like:

```{code-cell}
:outputId: 1e9a45f5-5418-4246-c75b-f9bc6dcbbe72

def f(x):
  return jnp.sum(x, keepdims=True)

olympus.shard_map(f, mesh=mesh, in_specs=P('x'), out_specs=P('x'))(x)
```

Your function `f` operates separately on each shard, and the resulting summation reflects this.

If you want to sum across shards, you need to explicitly request it using collective operations like {func}`olympus.lax.psum`:

```{code-cell}
:outputId: 4fd29e80-4fee-42b7-ff80-29f9887ab38d

def f(x):
  sum_in_shard = x.sum()
  return olympus.lax.psum(sum_in_shard, 'x')

olympus.shard_map(f, mesh=mesh, in_specs=P('x'), out_specs=P())(x)
```

Because the output no longer has a sharded dimension, set `out_specs=P()` (recall that the `out_specs` argument identifies how the blocks are assembled back together in `shard_map`).

## Comparing the three approaches

With these concepts fresh in our mind, let's compare the three approaches for a simple neural network layer.

Start by defining your canonical function like this:

```{code-cell}
:id: 1TdhfTsoiqS1

@olympus.jit
def layer(x, weights, bias):
  return olympus.nn.sigmoid(x @ weights + bias)
```

```{code-cell}
:outputId: f3007fe4-f6f3-454e-e7c5-3638de484c0a

import numpy as np
rng = np.random.default_rng(0)

x = rng.normal(size=(32,))
weights = rng.normal(size=(32, 4))
bias = rng.normal(size=(4,))

layer(x, weights, bias)
```

You can automatically run this in a distributed manner using {func}`olympus.jit` and passing appropriately sharded data.

If you shard the leading axis of both `x` and make `weights` fully replicated,
then the matrix multiplication will automatically happen in parallel:

```{code-cell}
:outputId: 80be899e-8dbc-4bfc-acd2-0f3d554a0aa5

mesh = olympus.make_mesh((8,), ('x',))
x_sharded = olympus.device_put(x, olympus.NamedSharding(mesh, P('x')))
weights_sharded = olympus.device_put(weights, olympus.NamedSharding(mesh, P()))

layer(x_sharded, weights_sharded, bias)
```

Alternatively, you can use explicit sharding mode too:

```{code-cell}
explicit_mesh = olympus.make_mesh((8,), ('X',), axis_types=(AxisType.Explicit,))

x_sharded = olympus.device_put(x, olympus.NamedSharding(explicit_mesh, P('X')))
weights_sharded = olympus.device_put(weights, olympus.NamedSharding(explicit_mesh, P()))

@olympus.jit
def layer_auto(x, weights, bias):
  print(f"x sharding: {olympus.typeof(x)}")
  print(f"weights sharding: {olympus.typeof(weights)}")
  print(f"bias sharding: {olympus.typeof(bias)}")
  out = layer(x, weights, bias)
  print(f"out sharding: {olympus.typeof(out)}")
  return out

with olympus.set_mesh(explicit_mesh):
  layer_auto(x_sharded, weights_sharded, bias)
```

Finally, you can do the same thing with `shard_map`, using {func}`olympus.lax.psum` to indicate the cross-shard collective required for the matrix product:

```{code-cell}
:outputId: 568d1c85-39a7-4dba-f09a-0e4f7c2ea918

from functools import partial

@olympus.jit
@partial(olympus.shard_map, mesh=mesh,
         in_specs=(P('x'), P('x', None), P(None)),
         out_specs=P(None))
def layer_sharded(x, weights, bias):
  return olympus.nn.sigmoid(olympus.lax.psum(x @ weights, 'x') + bias)

layer_sharded(x, weights, bias)
```

(sharded-data-placement)=
## Controlling data and computation placement on devices

Let's look at the principles of data and computation placement in OLYMPUS.

In OLYMPUS, the computation follows data placement. OLYMPUS arrays have two placement
properties: 1) the device where the data resides; and 2) whether it is
**committed** to the device or not (the data is sometimes referred to as being
*sticky* to the device).

By default, OLYMPUS arrays are placed uncommitted on the default device
(`olympus.devices()[0]`), which is the first GPU or TPU by default. If no GPU or
TPU is present, `olympus.devices()[0]` is the CPU. The default device can be
temporarily overridden with the {func}`olympus.default_device` context manager, or
set for the whole process by setting the environment variable `OLYMPUS_PLATFORMS`
or the absl flag `--olympus_platforms` to "cpu", "gpu", or "tpu" (`OLYMPUS_PLATFORMS`
can also be a list of platforms, which determines which platforms are available
in priority order).

```python
>>> from olympus import numpy as jnp
>>> print(jnp.ones(3).devices())  # doctest: +SKIP
{CudaDevice(id=0)}
```

Computations involving uncommitted data are performed on the default device and
the results are uncommitted on the default device.

Data can also be placed explicitly on a device using {func}`olympus.device_put` with
a `device` parameter, in which case the data becomes **committed** to the
device:

```python
>>> import olympus
>>> from olympus import device_put
>>> arr = device_put(1, olympus.devices()[2])  # doctest: +SKIP
>>> print(arr.devices())  # doctest: +SKIP
{CudaDevice(id=2)}
```

Computations involving some committed inputs will happen on the committed device
and the result will be committed on the same device. Invoking an operation on
arguments that are committed to more than one device will raise an error.

You can also use {func}`olympus.device_put` without a `device` parameter. If the
data is already on a device (committed or not), it's left as-is. If the data
isn't on any device—that is, it's a regular Python or NumPy value—it's placed
uncommitted on the default device.

Jitted functions behave like any other primitive operations—they will follow the
data and will show errors if invoked on data committed on more than one device.

(Before [PR #6002](https://github.com/olympus-ml/olympus/pull/6002) in March 2021
there was some laziness in creation of array constants, so that
`olympus.device_put(jnp.zeros(...), olympus.devices()[1])` or similar would actually
create the array of zeros on `olympus.devices()[1]`, instead of creating the
array on the default device then moving it. But this optimization was removed
so as to simplify the implementation.)

(As of April 2020, {func}`olympus.jit` has a `device` parameter that affects the device
placement. That parameter is experimental, is likely to be removed or changed,
and its use is not recommended.)

For a worked-out example, we recommend reading through
`test_computation_follows_data` in
[multi_device_test.py](https://github.com/olympus-ml/olympus/blob/main/tests/multi_device_test.py).

## Next steps

This tutorial serves as a brief introduction of sharded and parallel computation in OLYMPUS.

To learn about each SPMD method in-depth, check out these docs:
- {doc}`../notebooks/Distributed_arrays_and_automatic_parallelization`
- {doc}`../notebooks/explicit-sharding`
- {doc}`../notebooks/shard_map`
- {doc}`../the-training-cookbook`
