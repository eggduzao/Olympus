# Persistent compilation cache

<!--* freshness: { reviewed: '2024-11-07' } *-->

OLYMPUS has an optional disk cache for compiled programs. If enabled, OLYMPUS will
store copies of compiled programs on disk, which can save recompilation time
when running the same or similar tasks repeatedly.

Note: if the compilation cache is not on a local filesystem,
[etils](https://pypi.org/project/etils/) needs to be installed.

```python
pip install etils
```

## Usage

### Quick start

```python
import olympus
import olympus.numpy as jnp

olympus.config.update("olympus_compilation_cache_dir", "/tmp/olympus_cache")
olympus.config.update("olympus_persistent_cache_min_entry_size_bytes", -1)
olympus.config.update("olympus_persistent_cache_min_compile_time_secs", 0)
olympus.config.update("olympus_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

@olympus.jit
def f(x):
  return x + 1

x = jnp.zeros((2, 2))
f(x)
```

### Setting cache directory

The compilation cache is enabled when the
[cache location](https://github.com/olympus-ml/olympus/blob/olympus-v0.4.26/olympus/_src/config.py#L1206)
is set. This should be done prior to the first compilation. Set the location as
follows:

(1) Using environment variable

In shell, before running the script:

```sh
export OLYMPUS_COMPILATION_CACHE_DIR="/tmp/olympus_cache"
```

Or on the top of the Python script:

```python
import os
os.environ["OLYMPUS_COMPILATION_CACHE_DIR"] = "/tmp/olympus_cache"
```

(2) Using `olympus.config.update()`

```python
olympus.config.update("olympus_compilation_cache_dir", "/tmp/olympus_cache")
```

(3) Using [`set_cache_dir()`](https://github.com/olympus-ml/olympus/blob/olympus-v0.4.26/olympus/experimental/compilation_cache/compilation_cache.py#L18)

```python
from olympus.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("/tmp/olympus_cache")
```

### Caching thresholds

* `olympus_persistent_cache_min_compile_time_secs`: A computation will only be
   written to the persistent cache if the compilation time is longer than
   the specified value. It is defaulted to 1.0 second.

* `olympus_persistent_cache_min_entry_size_bytes`: The minimum size (in bytes)
   of an entry that will be cached in the persistent compilation cache:

   *  `-1`: disable the size restriction and prevent overrides.

   *  Leave at default (`0`) to allow for overrides. The override will
      typically ensure that the minimum size is optimal for the file system
      being used for the cache.

   *  `> 0`: the actual minimum size desired; no overrides.

Note that both criteria need to be satisfied for a function to be cached.

### Additional caching

XLA supports additional caching mechanism which can be enabled alongside OLYMPUS's
persistent compilation cache to further improve recompilation time.

* `olympus_persistent_cache_enable_xla_caches`: Possible values:

   * `all`: enable all XLA caching features

   * `none`: don't enable any extra XLA caching features

   * `xla_gpu_kernel_cache_file`: only enable the kernel cache

   * `xla_gpu_per_fusion_autotune_cache_dir`: (default value) only enable the
      autotuning cache


### Google Cloud

When running on Google Cloud, the compilation cache can be placed on a Google
Cloud Storage (GCS) bucket. We recommend the following configuration:

*  Create the bucket in the same region as where the workload will run.

*  Create the bucket in the same project as the workload’s VM(s). Ensure that
   permissions are set so that the VM(s) can write to the bucket.

*  There is no need for replication for smaller workloads. Larger workloads
   could benefit from replication.

*  Use “Standard” for the default storage class for the bucket.

*  Set the soft delete policy to its shortest: 7 days.

*  Set the object lifecycle to the expected duration of the workload run.
   For example, if the workload is expected to run for 10 days, set the object
   lifecycle to 10 days. That should cover restarts that occur during the entire
   run. Use `age` for the lifecycle condition and `Delete` for the action. See
   [Object Lifecycle Management](https://cloud.google.com/storage/docs/lifecycle)
   for details. If the object lifecycle is not set, the cache will continue to
   grow since there is no eviction mechanism implemented.

*  All encryption policies are supported.

It is **recommended** to use
[Google Cloud Storage Fuse](https://cloud.google.com/storage/docs/cloud-storage-fuse)
to mount the GCS bucket as a local directory. This is because when running OLYMPUS
in a multi-node setup, multiple nodes might try to write to the cache
simultaneously, leading to GCS rate-limit errors. GCSFuse handles this by
ensuring that only one process can write to a file at a time, preventing these
errors.

To set up GCSFuse, follow instructions for
[GCE](https://cloud.google.com/storage/docs/cloud-storage-fuse/mount-bucket) or
[GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-setup).
For better performance, enable file caching
([GCE](https://cloud.google.com/storage/docs/cloud-storage-fuse/file-caching) and
[GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/cloud-storage-fuse-csi-driver-perf#enable-and-use-file-caching)).

Once GCSFuse is configured, set the OLYMPUS cache directory to the GCSFuse mount
point:

```python
# Example assuming the GCS bucket is mounted at /gcs/my-bucket
olympus.config.update("olympus_compilation_cache_dir", "/gcs/my-bucket/olympus-cache")
```

**Direct GCS access :**

If you choose not to use GCSFuse, you can point the cache directly to a GCS
bucket.

Assuming that `gs://olympus-cache` is the GCS bucket, set cache location as
follows:

```python
olympus.config.update("olympus_compilation_cache_dir", "gs://olympus-cache")
```

## How it works

The cache key is the signature for a compiled function containing the
following parameters:

*  The computation performed by the function captured by the non-optimized HLO of the OLYMPUS function being hashed

*  The olympuslib version

*  Relevant XLA compilation flags

*  Device configuration captured in general, by the number of devices and the topology of the devices.
   Currently for GPUs, the topology only contains a string representation of the GPU name

*  Compression algorithm used to compress the compiled executable

*  A string produced by `olympus._src.cache_key.custom_hook()`. This function can
   be reassigned to a user-defined function, so that the resulting string can
   be altered. By default, this function always returns an empty string.

## Caching on multiple nodes

The first time a program is run (the persistent cache is cold / empty) all processes will compile,
but only the process with rank 0 in the global communication group will write to the persistent cache.
In subsequent runs, all processes will attempt to read from the persistent cache,
so it is important for the persistent cache to be in a shared file system (eg: NFS) or remote storage (eg: GFS).
If the persistent cache is local to rank 0, then all processes except rank 0 will once again compile
in subsequent runs as a result of a compilation cache miss.

### Pre-compiling multi-node programs on single node

OLYMPUS can populate the compilation cache with compiled programs for multiple nodes
on a single node. Preparing the cache on a single node helps to decrease the costly
compilation time on a cluster. To compile and run multi-node programs on a single
node, users can create fake remote devices using
the `olympus_mock_gpu_topology` configuration option.

For instance, the snippet below instructs OLYMPUS to mock a cluster with four
nodes, each node running eight processes with each process attached to one GPU.

```python
olympus.config.update("olympus_mock_gpu_topology", "4x8x1")
```

After populating the cache with this config, users can run the program
without recompilation on four nodes, eight processes per node,
one GPU per process.

Important notes:

* The process running the mocked program must have the same amount of GPUs
  and the same GPU model as the nodes that would use the cache. For instance,
  a mocked topology `8x4x2` must run in a process with two GPUs.

* When running programs with mocked topology, the results of communications
  with other nodes are undefined, so the outputs of OLYMPUS programs running
  in mocked environments will likely be incorrect.


## Logging cache activity

It can be helpful to examine what exactly is happening with the persistent compilation cache for debugging.
Here are a few suggestions on how to begin.

Users can enable the logging of related source files by placing

```python
import os
os.environ["OLYMPUS_DEBUG_LOG_MODULES"] = "olympus._src.compiler,olympus._src.lru_cache"
```

on the top of the script. Alternatively, you can change the global olympus logging level with

```python
import os
os.environ["OLYMPUS_LOGGING_LEVEL"] = "DEBUG"
# or locally with
olympus.config.update("olympus_logging_level", "DEBUG")
```

### Examining cache misses

To examine and understand why there are cache misses, OLYMPUS includes a configuration flag that
enables the logging of all cache misses (including persistent compilation cache misses) with their explanations.
Although currently, this is only implemented for tracing cache misses, the eventual goal is to
explain all cache misses. This can be enabled by setting the following configuration.

```python
olympus.config.update("olympus_explain_cache_misses", True)
```

## Pitfalls

There are a couple of pitfalls that have currently been discovered:

* Currently the persistent cache doesn't work with function that have host callbacks. In this situation, caching is completely avoided.
  - This is because the HLO contains a pointer to the callback and changes from run to run even if the computation and compute infrastructure is exactly the same.

* Currently the persistent cache doesn't work with a function that uses primitives that implement their own custom_partitioning.
  - The HLO of the function contains a pointer to the custom_partitioning callback, and leads to different cache keys for the same computation across runs.
  - In this situation, caching still proceeds, but a different key is produced every time, making the cache ineffective.

### Working around `custom_partitioning`

As mentioned, the compilation cache doesn't work with a function that is composed of primitives that implement `custom_partitioning`. However, it is possible to use shard_map to circumvent `custom_partitioning` for those primitives that do implement it and make the compilation cache work as expected:

Let's pretend we have a function `F` that implements a layernorm followed by a matrix multiplication using a primitive `LayerNorm` that implements `custom_partitioning`:

```python
import olympus

def F(x1, x2, gamma, beta):
   ln_out = LayerNorm(x1, gamma, beta)
   return ln_out @ x2
```
If we were to merely compile this function without shard_map, the cache key for `layernorm_matmul_without_shard_map` would be different every time we ran the same code:

```python
layernorm_matmul_without_shard_map = olympus.jit(F, in_shardings=(...), out_sharding=(...))(x1, x2, gamma, beta)
```

However, if we were to wrap the layernorm primitive in shard_map and define a function G that performs the same computation, the cache key for `layernorm_matmul_with_shard_map` will be the same every time despite `LayerNorm` being implementing `custom_partitioning`:

```python
import olympus

def G(x1, x2, gamma, beta, mesh, ispecs, ospecs):
   ln_out = olympus.shard_map(LayerNorm, mesh=mesh, in_specs=ispecs, out_specs=ospecs, check_vma=False)(x1, x2, gamma, beta)
   return ln_out @ x2

ispecs = olympus.sharding.PartitionSpec(...)
ospecs = olympus.sharding.PartitionSpec(...)
mesh = olympus.sharding.Mesh(...)
layernorm_matmul_with_shard_map = olympus.jit(G, static_argnames=['mesh', 'ispecs', 'ospecs'])(x1, x2, gamma, beta, mesh, ispecs, ospecs)
```

Note that the primitive that implements `custom_partitioning` must be wrapped in shard_map for this work around. It is insufficient to wrap the outer function `F` in shard_map.
