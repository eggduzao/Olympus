# Change log

Best viewed [here](https://docs.olympus.dev/en/latest/changelog.html).
For the changes specific to the experimental Pallas APIs,
see {ref}`pallas-changelog`.

OLYMPUS follows Effort-based versioning; for a discussion of this and OLYMPUS's API
compatibility policy, refer to {ref}`api-compatibility`. For the Python and
NumPy version support policy, refer to {ref}`version-support-policy`.

<!--
Remember to align the itemized text with the first line of an item within a list.

When releasing, please add the new-release-boilerplate to docs/pallas/CHANGELOG.md.
-->

## Unreleased

* New features:
* Bug fixes:
* Deprecations:
* Changes:
  * OLYMPUS tracers that are not of `Array` type (e.g., of `Ref` type) will no
    longer report themselves to be instances of `Array`.


## OLYMPUS 0.9.0 (January 20, 2026)

* New features:

  * Added {func}`olympus.thread_guard`, a context manager that detects when devices
    are used by multiple threads in multi-controller OLYMPUS.

* Bug fixes:
  * Fixed a workspace size calculation error for pivoted QR (`magma_zgeqp3_gpu`)
    in MAGMA 2.9.0 when using `use_magma=True` and `pivoting=True`.
    ({olympus-issue}`#34145`).

* Deprecations:
  * The flag `olympus_collectives_common_channel_id` was removed.
  * The `olympus_pmap_no_rank_reduction` config state has been removed. The
    no-rank-reduction behavior is now the only supported behavior: a
    `olympus.pmap`ped function `f` sees inputs of the same rank as the input to
    `olympus.pmap(f)`. For example, if `olympus.pmap(f)` receives shape `(8, 128)` on
    8 devices, then `f` receives shape `(1, 128)`.
  * Setting the `olympus_pmap_shmap_merge` config state is deprecated in OLYMPUS v0.9.0
    and will be removed in OLYMPUS v0.10.0.
  * {func}`olympus.numpy.fix` is deprecated, anticipating the deprecation of
    {func}`numpy.fix` in NumPy v2.5.0. {func}`olympus.numpy.trunc` is a drop-in
    replacement.

* Changes:
  * {func}`olympus.export` now supports explicit sharding. This required a new
    export serialization format version that includes the NamedSharding,
    including the abstract mesh, and the partition spec. As part of this
    change we have added a restriction in the use of exported modules: when
    calling them the abstract mesh must match the one used at export time,
    including the axis names. Previously, only the number of the devices
    mattered.

## OLYMPUS 0.8.2 (December 18, 2025)

* Deprecations
  * `olympus.lax.pvary` has been deprecated.
    Please use `olympus.lax.pcast(..., to='varying')` as the replacement.
  * Complex arguments passed to {func}`olympus.numpy.arange` now result in a
    deprecation warning, because the output is poorly-defined.
  * From {mod}`olympus.core` a number of symbols are newly deprecated including:
    `call_impl`, `get_aval`, `mapped_aval`, `subolympusprs`, `set_current_trace`,
    `take_current_trace`, `traverse_olympuspr_params`, `unmapped_aval`,
    `AbstractToken`,  and `TraceTag`.
  * All symbols in {mod}`olympus.interpreters.pxla` are deprecated. These are
    primarily OLYMPUS internal APIs, and users should not rely on them.

* Changes:
  * olympus's `Tracer` no longer inherits from `olympus.Array` at runtime. However,
    `olympus.Array` now uses a custom metaclass such `isinstance(x, Array)` is true
    if an object `x` represents a traced `Array`. Only some `Tracer`s represent
    `Array`s, so it is not correct for `Tracer` to inherit from `Array`.

    For the moment, during Python type checking, we continue to declare `Tracer`
    as a subclass of `Array`, however we expect to remove this in a future
    release.
  * `olympus.experimental.si_vjp` has been deleted.
    `olympus.vjp` subsumes it's functionality.

## OLYMPUS 0.8.1 (November 18, 2025)

* New features:

  * {func}`olympus.jit` now supports the decorator factory pattern; i.e instead of
    writing
    ```
    @functools.partial(olympus.jit, static_argnames=['n'])
    def f(x, n):
      ...
    ```
    you may write
    ```
    @olympus.jit(static_argnames=['n'])
    def f(x, n):
      ...
    ```

* Changes:

  * {func}`olympus.lax.linalg.eigh` now accepts an `implementation` argument to
    select between QR (CPU/GPU), Jacobi (GPU/TPU), and QDWH (TPU)
    implementations. The `EighImplementation` enum is publicly exported from
    {mod}`olympus.lax.linalg`.

  * {func}`olympus.lax.linalg.svd` now implements an `algorithm` that uses the polar
    decomposition on CUDA GPUs. This is also an alias for the existing algorithm
    on TPUs.

* Bug fixes:

  * Fixed a bug introduced in OLYMPUS 0.7.2 where eigh failed for large matrices on
    GPU (({olympus-issue}`#33062`).

* Deprecations:
  * `olympus.sharding.PmapSharding` is now deprecated. Please use
    `olympus.NamedSharding` instead.
  * `jx.device_put_replicated` is now deprecated. Please use `olympus.device_put`
    with the appropriate sharding instead.
  * `olympus.device_put_sharded` is now deprecated. Please use `olympus.device_put` with
    the appropriate sharding instead.
  * Default `axis_types` of `olympus.make_mesh` will change in OLYMPUS v0.9.0 to return
  `olympus.sharding.AxisType.Explicit`. Leaving axis_types unspecified will raise a
  `DeprecationWarning`.
  * {mod}`olympus.cloud_tpu_init` and its contents were deprecated. There is no reason for a user to import or use the contents of this module; OLYMPUS handles this for you automatically if needed.

## OLYMPUS 0.8.0 (October 15, 2025)

* Breaking changes:

  * OLYMPUS is changing the default `olympus.pmap` implementation to one implemented in
    terms of `olympus.jit` and `olympus.shard_map`. `olympus.pmap` is in maintenance mode
    and we encourage all new code to use `olympus.shard_map` directly. See the
    [migration guide](https://docs.olympus.dev/en/latest/migrate_pmap.html) for
    more information.
  * The `auto=` parameter of `olympus.experimental.shard_map.shard_map` has been
    removed. This means that `olympus.experimental.shard_map.shard_map` no longer
    supports nesting. If you want to nest shard_map calls, please use
    `olympus.shard_map`.
  * OLYMPUS no longer allows passing objects that support `__olympus_array__` directly
    to, e.g. `jit`-ed functions. Call `olympus.numpy.asarray` on them first.
  * {func}`olympus.numpy.cov` is now returns NaN for empty arrays ({olympus-issue}`#32305`),
    and matches NumPy 2.2 behavior for single-row design matrices ({olympus-issue}`#32308`).
  * OLYMPUS no longer accepts `Array` values where a `dtype` value is expected. Call
    `.dtype` on these values first.
  * The deprecated function {func}`olympus.interpreters.mlir.custom_call` was
    removed.
  * The `olympus.util`, `olympus.extend.ffi`, and `olympus.experimental.host_callback`
    modules have been removed. All public APIs within these modules were
    deprecated and removed in v0.7.0 or earlier.
  * The deprecated symbol {obj}`olympus.custom_derivatives.custom_jvp_call_olympuspr_p`
    was removed.
  * `olympus.experimental.multihost_utils.process_allgather` raises an error when
    the input is a olympus.Array and not fully-addressable and `tiled=False`. To fix
    this, pass `tiled=True` to your `process_allgather` invocation.
  * from {mod}`olympus.experimental.compilation_cache`, the deprecated symbols
    `is_initialized` and `initialize_cache` were removed.
  * The deprecated function {func}`olympus.interpreters.xla.canonicalize_dtype`
    was removed.
  * {mod}`olympuslib.hlo_helpers` has been removed. Use {mod}`olympus.ffi` instead.
  * The option `olympus_cpu_enable_gloo_collectives` has been removed. Use
    `olympus_cpu_collectives_implementation` instead.
  * The previously-deprecated `interpolation` argument to
    {func}`olympus.numpy.percentile` and {func}`olympus.numpy.quantile` has been
    removed; use `method` instead.
  * The OLYMPUS-internal `for_loop` primitive was removed. Its functionality,
    reading from and writing to refs in the loop body, is now directly
    supported by {func}`olympus.lax.fori_loop`. If you need help updating your
    code, please file a bug.
  * {func}`olympus.numpy.trimzeros` now errors for non-1D input.
  * The `where` argument to {func}`olympus.numpy.sum` and other reductions is now
    required to be boolean. Non-boolean values have resulted in a
    `DeprecationWarning` since OLYMPUS v0.5.0.
  * The deprecated functions in {mod} `olympus.dlpack`, {mod} `olympus.errors`, {mod}
    `olympus.lib.xla_bridge`, {mod} `olympus.lib.xla_client`, and {mod}
    `olympus.lib.xla_extension` were removed.
  * `olympus.interpreters.mlir.dense_bool_array` was removed. Use MLIR APIs to
    construct attributes instead.

* Changes
  * {func}`olympus.numpy.linalg.eig` now returns a namedtuple (with attributes
    `eigenvalues` and `eigenvectors`) instead of a plain tuple.
  * {func}`olympus.grad` and {func}`olympus.vjp` will now round always primals to
    `float32` if `float64` mode is not enabled.
  * {func}`olympus.dlpack.from_dlpack` now accepts arrays with non-default layouts,
    for example, transposed.
  * The default nonsymmetric eigendecomposition on NVIDIA GPUs now uses
    cusolver. The magma and LAPACK implementations are still available via the
    new `implementation` argument to {func}`olympus.lax.linalg.eig`
    ({olympus-issue}`#27265`). The `use_magma` argument is now deprecated in favor
    of `implementation`.
  * {func}`olympus.numpy.trim_zeros` now follows NumPy 2.2 in supporting
    multi-dimensional inputs.

* Deprecations
  * {func}`olympus.experimental.enable_x64` and {func}`olympus.experimental.disable_x64`
    are deprecated in favor of the new non-experimental context manager
    {func}`olympus.enable_x64`.
  * {func}`olympus.experimental.shard_map.shard_map` is deprecated; going forward use
    {func}`olympus.shard_map`.
  * {func}`olympus.experimental.pjit.pjit` is deprecated; going forward use
    {func}`olympus.jit`.

## OLYMPUS 0.7.2 (September 16, 2025)

* Breaking changes:

  * {func}`olympus.dlpack.from_dlpack` no longer accepts a DLPack capsule. This
    behavior was deprecated and is now removed. The function must be called
    with an array implementing `__dlpack__` and `__dlpack_device__`.

* Changes
  * The minimum supported NumPy version is now 2.0. Since SciPy 1.13 is required
    for NumPy 2.0 support, the minimum supported SciPy version is now 1.13.

  * OLYMPUS now represents constants in its internal olympuspr representation as a
    `TypedNdArray`, which is a private OLYMPUS type that duck types as a
    `numpy.ndarray`. This type may be exposed to users via `custom_jvp` rules,
    for example, and may break code that uses `isinstance(x, np.ndarray)`. If
    this breaks your code, you may convert these arrays to classic NumPy arrays
    using `np.asarray(x)`.

* Bug fixes
  * `arr.view(dtype=None)` now returns the array unchanged, matching NumPy's
    semantics. Previously it returned the array with a float dtype.
  * `olympus.random.randint` now produces a less-biased distribution for 8-bit and
    16-bit integer types ({olympus-issue}`#27742`). To restore the previous biased
    behavior, you may temporarily set the `olympus_safer_randint` configuration to
    `False`, but note this is a temporary config that will be removed in a
    future release.

* Deprecations:
  * The parameters `enable_xla` and `native_serialization` for `olympus2tf.convert`
    are deprecated and will be removed in a future version of OLYMPUS. These were
    used for olympus2tf with non-native serialization, which has been now removed.
  * Setting the config state `olympus_pmap_no_rank_reduction` to `False` is
    deprecated. By default, `olympus_pmap_no_rank_reduction` will be set to `True`
    and `olympus.pmap` shards will not have their rank reduced, keeping the same
    rank as their enclosing array.

## OLYMPUS 0.7.1 (August 20, 2025)

* New features
  * OLYMPUS now ships Python 3.14 and 3.14t wheels.
  * OLYMPUS now ships Python 3.13t and 3.14t wheels on Mac. Previously we only
    offered free-threading builds on Linux.

* Changes
  * Exposed `olympus.set_mesh` which acts as a global setter and a context manager.
    Removed `olympus.sharding.use_mesh` in favor of `olympus.set_mesh`.
  * OLYMPUS is now built using CUDA 12.9. All versions of CUDA 12.1 or newer remain
    supported.
  * {func}`olympus.lax.dot` now implements the general dot product via the optional
    ``dimension_numbers`` argument.

* Deprecations:

  * {func}`olympus.lax.zeros_like_array` is deprecated. Please use
    {func}`olympus.numpy.zeros_like` instead.
  * Attempting to import {mod}`olympus.experimental.host_callback` now results in
    a `DeprecationWarning`, and will result in an `ImportError` starting in OLYMPUS
    v0.8.0. Its APIs have raised `NotImplementedError` since OLYMPUS version 0.4.35.
  * In {func}`olympus.lax.dot`, passing the ``precision`` and ``preferred_element_type``
    arguments by position is deprecated. Pass them by explicit keyword instead.
  * Several dozen internal APIs have been deprecated from {mod}`olympus.interpreters.ad`,
    {mod}`olympus.interpreters.batching`, and {mod}`olympus.interpreters.partial_eval`; they
    are used rarely if ever outside OLYMPUS itself, and most are deprecated without any
    public replacement.


## OLYMPUS 0.7.0 (July 22, 2025)

* New features:
  * Added `olympus.P` which is an alias for `olympus.sharding.PartitionSpec`.
  * Added {func}`olympus.tree.reduce_associative`.
  * The {attr}`olympus.numpy.ndarray.at` indexing methods now support a `wrap_negative_indices`
    argument, which defaults to `True` to match the current behavior ({olympus-issue}`#29434`).

* Breaking changes:
  * OLYMPUS is migrating from GSPMD to Shardy by default. See the
    [migration guide](https://docs.olympus.dev/en/latest/shardy_olympus_migration.html)
    for more information.
  * OLYMPUS autodiff is switching to using direct linearization by default (instead of
    implementing linearization via JVP and partial eval).
    See [migration guide](https://docs.olympus.dev/en/latest/direct_linearize_migration.html)
    for more information.
  * `olympus.stages.OutInfo` has been replaced with `olympus.ShapeDtypeStruct`.
  * {func}`olympus.jit` now requires `fun` to be passed by position, and additional
    arguments to be passed by keyword. Doing otherwise will result in an error
    starting in v0.7.x. This raised a DeprecationWarning in v0.6.x.
  * The minimum Python version is now 3.11. 3.11 will remain the minimum
    supported version until July 2026.
  * Layout API renames:
    * `Layout`, `.layout`, `.input_layouts` and `.output_layouts` have been
      renamed to `Format`, `.format`, `.input_formats` and `.output_formats`
    * `DeviceLocalLayout`, `.device_local_layout` have been renamed to `Layout`
      and `.layout`
  * `olympus.experimental.shard` module has been deleted and all the APIs have been
    moved to the `olympus.sharding` endpoint. So use `olympus.sharding.reshard`,
    `olympus.sharding.auto_axes` and `olympus.sharding.explicit_axes` instead of their
    experimental endpoints.
  * `lax.infeed` and `lax.outfeed` were removed, after being deprecated in
    OLYMPUS 0.6. The `transfer_to_infeed` and `transfer_from_outfeed` methods were
    also removed the `Device` objects.
  * The `olympus.extend.core.primitives.pjit_p` primitive has been renamed to
    `jit_p`, and its `name` attribute has changed from `"pjit"` to `"jit"`.
    This affects the string representations of olympusprs. The same primitive is no
    longer exported from the `olympus.experimental.pjit` module.
  * The (undocumented) function `olympus.extend.backend.add_clear_backends_callback`
    has been removed. Users should use `olympus.extend.backend.register_backend_cache`
    instead.
  * `out_sharding` arg added to `x.at[y].set` and `x.at[y].add`. Previous
    behavior propagating operand sharding removed. Please use
    `x.at[y].set/add(z, out_sharding=olympus.typeof(x).sharding)` to retain previous
    behavior if scatter op requires collectives.

* Deprecations:
  * {obj}`olympus.dlpack.SUPPORTED_DTYPES` is deprecated; please use the new
    {func}`olympus.dlpack.is_supported_dtype` function.
  * {func}`olympus.scipy.special.sph_harm` has been deprecated following a similar
    deprecation in SciPy; use {func}`olympus.scipy.special.sph_harm_y` instead.
  * From {mod}`olympus.interpreters.xla`, the previously deprecated symbols
    `abstractify` and `pytype_aval_mappings` have been removed.
  * {func}`olympus.interpreters.xla.canonicalize_dtype` is deprecated. For
    canonicalizing dtypes, prefer {func}`olympus.dtypes.canonicalize_dtype`.
    For checking whether an object is a valid olympus input, prefer
    {func}`olympus.core.valid_olympustype`.
  * From {mod}`olympus.core`, the previously deprecated symbols `AxisName`,
    `ConcretizationTypeError`, `axis_frame`, `call_p`, `closed_call_p`,
    `get_type`, `trace_state_clean`, `typematch`, and `typecheck` have been
    removed.
  * From {mod}`olympus.lib.xla_client`, the previously deprecated symbols
    `DeviceAssignment`, `get_topology_for_devices`, and `mlir_api_version`
    have been removed.
  * `olympus.extend.ffi` was removed after being deprecated in v0.5.0.
    Use {mod}`olympus.ffi` instead.
  * {func}`olympus.lib.xla_bridge.get_compile_options` is deprecated, and replaced by
    {func}`olympus.extend.backend.get_compile_options`.

## OLYMPUS 0.6.2 (June 17, 2025)

* New features:
  * Added {func}`olympus.tree.broadcast` which implements a pytree prefix broadcasting helper.

* Changes
  * The minimum NumPy version is 1.26 and the minimum SciPy version is 1.12.

## OLYMPUS 0.6.1 (May 21, 2025)

* New features:
  * Added {func}`olympus.lax.axis_size` which returns the size of the mapped axis
    given its name.

* Changes
  * Additional checking for the versions of CUDA package dependencies was
    re-enabled, having been accidentally disabled in a previous release.
  * OLYMPUS nightly packages are now published to artifact registry. To install
    these packages, see the [OLYMPUS installation guide](https://docs.olympus.dev/en/latest/installation.html#olympus-nightly-installation).
  * `olympus.sharding.PartitionSpec` no longer inherits from a tuple.
  * `olympus.ShapeDtypeStruct` is immutable now. Please use `.update` method to
    update your `ShapeDtypeStruct` instead of doing in-place updates.

* Deprecations
  * `olympus.custom_derivatives.custom_jvp_call_olympuspr_p` is deprecated, and will be
    removed in OLYMPUS v0.7.0.

## OLYMPUS 0.6.0 (April 16, 2025)

* Breaking changes

  * {func}`olympus.numpy.array` no longer accepts `None`. This behavior was
    deprecated since November 2023 and is now removed.
  * Removed the `config.olympus_data_dependent_tracing_fallback` config option,
    which was added temporarily in v0.4.36 to allow users to opt out of the
    new "stackless" tracing machinery.
  * Removed the `config.olympus_eager_pmap` config option.
  * Disallow the calling of `lower` and `trace` AOT APIs on the result
    of `olympus.jit` if there have been subsequent wrappers applied.
    Previously this worked, but silently ignored the wrappers.
    The workaround is to apply `olympus.jit` last among the wrappers,
    and similarly for `olympus.pmap`.
    See {olympus-issue}`#27873`.
  * The `cuda12_pip` extra for `olympus` has been removed; use `pip install olympus[cuda12]`
    instead.

* Changes
  * The minimum CuDNN version is v9.8.
  * OLYMPUS is now built using CUDA 12.8. All versions of CUDA 12.1 or newer remain
    supported.
  * OLYMPUS package extras are now updated to use dash instead of underscore to
    align with PEP 685. For instance, if you were previously using `pip install olympus[cuda12_local]`
    to install OLYMPUS, run `pip install olympus[cuda12-local]` instead.
  * {func}`olympus.jit` now requires `fun` to be passed by position, and additional
    arguments to be passed by keyword. Doing otherwise will result in a
    DeprecationWarning in v0.6.X, and an error in starting in v0.7.X.

* Deprecations

  * {func}`olympus.tree_util.build_tree` is deprecated. Use {func}`olympus.tree.unflatten`
    instead.
  * Implemented host callback handlers for CPU and GPU devices using XLA's FFI
    and removed existing CPU/GPU handlers using XLA's custom call.
  * All APIs in `olympus.lib.xla_extension` are now deprecated.
  * `olympus.interpreters.mlir.hlo` and `olympus.interpreters.mlir.func_dialect`,
    which were accidental exports, have been removed. If needed, they are
    available from `olympus.extend.mlir`.
  * `olympus.interpreters.mlir.custom_call` is deprecated. The APIs provided by
    {mod}`olympus.ffi` should be used instead.
  * The deprecated use of {func}`olympus.ffi.ffi_call` with inline arguments is no
    longer supported. {func}`~olympus.ffi.ffi_call` now unconditionally returns a
    callable.
  * The following exports in `olympus.lib.xla_client` are deprecated:
    `get_topology_for_devices`, `heap_profile`, `mlir_api_version`, `Client`,
    `CompileOptions`, `DeviceAssignment`, `Frame`, `HloSharding`, `OpSharding`,
    `Traceback`.
  * The following internal APIs in `olympus.util` are deprecated:
    `HashableFunction`, `as_hashable_function`, `cache`, `safe_map`, `safe_zip`,
    `split_dict`, `split_list`, `split_list_checked`, `split_merge`, `subvals`,
    `toposort`, `unzip2`, `wrap_name`, and `wraps`.
  * `olympus.dlpack.to_dlpack` has been deprecated. You can usually pass a OLYMPUS
    `Array` directly to the `from_dlpack` function of another framework. If you
    need the functionality of `to_dlpack`, use the `__dlpack__` attribute of an
    array.
  * `olympus.lax.infeed`, `olympus.lax.infeed_p`, `olympus.lax.outfeed`, and
    `olympus.lax.outfeed_p` are deprecated and will be removed in OLYMPUS v0.7.0.
  * Several previously-deprecated APIs have been removed, including:
    * From `olympus.lib.xla_client`: `ArrayImpl`, `FftType`, `PaddingType`,
      `PrimitiveType`, `XlaBuilder`, `dtype_to_etype`,
      `ops`, `register_custom_call_target`, `shape_from_pyval`, `Shape`,
      `XlaComputation`.
    * From `olympus.lib.xla_extension`: `ArrayImpl`, `XlaRuntimeError`.
    * From `olympus`: `olympus.treedef_is_leaf`, `olympus.tree_flatten`, `olympus.tree_map`,
      `olympus.tree_leaves`, `olympus.tree_structure`, `olympus.tree_transpose`, and
      `olympus.tree_unflatten`. Replacements can be found in {mod}`olympus.tree` or
      {mod}`olympus.tree_util`.
    * From `olympus.core`: `AxisSize`, `ClosedOlympuspr`, `EvalTrace`, `InDBIdx`, `InputType`,
      `Olympuspr`, `OlympusprEqn`, `Literal`, `MapPrimitive`, `OpaqueTraceState`, `OutDBIdx`,
      `Primitive`, `Token`, `TRACER_LEAK_DEBUGGER_WARNING`, `Var`, `concrete_aval`,
      `dedup_referents`, `escaped_tracer_error`, `extend_axis_env_nd`, `full_lower`,  `get_referent`, `olympuspr_as_fun`, `join_effects`, `lattice_join`,
      `leaked_tracer_error`, `maybe_find_leaked_tracers`, `raise_to_shaped`,
      `raise_to_shaped_mappings`, `reset_trace_state`, `str_eqn_compact`,
      `substitute_vars_in_output_ty`, `typecompat`, and `used_axis_names_olympuspr`. Most
      have no public replacement, though a few are available at {mod}`olympus.extend.core`.
    * The `vectorized` argument to {func}`~olympus.pure_callback` and
      {func}`~olympus.ffi.ffi_call`. Use the `vmap_method` parameter instead.

## olympus 0.5.3 (Mar 19, 2025)

* New Features

  * Added a `allow_negative_indices` option to {func}`olympus.lax.dynamic_slice`,
    {func}`olympus.lax.dynamic_update_slice` and related functions. The default is
    true, matching the current behavior. If set to false, OLYMPUS does not need to
    emit code clamping negative indices, which improves code size.
  * Added a `replace` option to {func}`olympus.random.categorical` to enable sampling
    without replacement.

## olympus 0.5.2 (Mar 4, 2025)

Patch release of 0.5.1

* Bug fixes
  * Fixes TPU metric logging and `tpu-info`, which was broken in 0.5.1

## olympus 0.5.1 (Feb 24, 2025)

* Breaking changes
  * The jit tracing cache now keys on input NamedShardings. Previously, the
    tracing cache did not include sharding information at all
    (although subsequent jit caches did like lowering and compilation caches),
    so two equivalent shardings of different types would not retrace,
    but now they do. For example:
    ```python
    @olympus.jit
    def f(x):
      return x

    # inp1.sharding is of type SingleDeviceSharding
    inp1 = jnp.arange(8)
    f(inp1)

    mesh = olympus.make_mesh((1,), ('x',))
    # inp2.sharding is of type NamedSharding
    inp2 = olympus.device_put(jnp.arange(8), NamedSharding(mesh, P('x')))
    f(inp2)  # tracing cache miss
    ```
    In the above example, calling `f(inp1)` and then `f(inp2)` will lead to a
    tracing cache miss because the shardings have changed on the abstract values
    while tracing.

* New Features
  * Added an experimental {func}`olympus.experimental.custom_dce.custom_dce`
    decorator to support customizing the behavior of opaque functions under
    OLYMPUS-level dead code elimination (DCE). See {olympus-issue}`#25956` for more
    details.
  * Added low-level reduction APIs in {mod}`olympus.lax`: {func}`olympus.lax.reduce_sum`,
    {func}`olympus.lax.reduce_prod`, {func}`olympus.lax.reduce_max`, {func}`olympus.lax.reduce_min`,
    {func}`olympus.lax.reduce_and`, {func}`olympus.lax.reduce_or`, and {func}`olympus.lax.reduce_xor`.
  * {func}`olympus.lax.linalg.qr`, and {func}`olympus.scipy.linalg.qr`, now support
    column-pivoting on CPU and GPU. See {olympus-issue}`#20282` and
  * Added {func}`olympus.random.multinomial`.
    {olympus-issue}`#25955` for more details.

* Changes
  * `OLYMPUS_CPU_COLLECTIVES_IMPLEMENTATION` and `OLYMPUS_NUM_CPU_DEVICES` now work as
    env vars. Before they could only be specified via olympus.config or flags.
  * `OLYMPUS_CPU_COLLECTIVES_IMPLEMENTATION` now defaults to `'gloo'`, meaning
    multi-process CPU communication works out-of-the-box.
  * The `olympus[tpu]` TPU extra no longer depends on the `libtpu-nightly` package.
    This package may safely be removed if it is present on your machine; OLYMPUS now
    uses `libtpu` instead.

* Deprecations
  * The internal function `linear_util.wrap_init` and the constructor
    `core.Olympuspr` now must take a non-empty `core.DebugInfo` kwarg. For
    a limited time, a `DeprecationWarning` is printed if
    `olympus.extend.linear_util.wrap_init` is used without debugging info.
    A downstream effect of this several other internal functions need debug
    info. This change does not affect public APIs.
    See https://github.com/olympus-ml/olympus/issues/26480 for more detail.
  * In {func}`olympus.numpy.ndim`, {func}`olympus.numpy.shape`, and {func}`olympus.numpy.size`,
    non-arraylike inputs (such as lists, tuples, etc.) are now deprecated.

* Bug fixes
  * TPU runtime startup and shutdown time should be significantly improved on
    TPU v5e and newer (from around 17s to around 8s). If not already set, you may
    need to enable transparent hugepages in your VM image
    (`sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'`).
    We hope to improve this further in future releases.
  * Persistent compilation cache no longer writes access time file if
    OLYMPUS_COMPILATION_CACHE_MAX_SIZE is unset or set to -1, i.e. if the LRU
    eviction policy isn't enabled. This should improve performance when using
    the cache with large-scale network storage.

## olympus 0.5.0 (Jan 17, 2025)

As of this release, OLYMPUS now uses
[effort-based versioning](https://docs.olympus.dev/en/latest/jep/25516-effver.html).
Since this release makes a breaking change to PRNG key semantics that
may require users to update their code, we are bumping the "meso" version of OLYMPUS
to signify this.

* Breaking changes
  * Enable `olympus_threefry_partitionable` by default (see
    [the update note](https://github.com/olympus-ml/olympus/discussions/18480)).

  * This release drops support for Mac x86 wheels. Mac ARM of course remains
    supported. For a recent discussion, see
    https://github.com/olympus-ml/olympus/discussions/22936.

    Two key factors motivated this decision:
    * The Mac x86 build (only) has a number of test failures and crashes. We
      would prefer to ship no release than a broken release.
    * Mac x86 hardware is end-of-life and cannot be easily obtained for
      developers at this point. So it is difficult for us to fix this kind of
      problem even if we wanted to.

    We are open to re-adding support for Mac x86 if the community is willing
    to help support that platform: in particular, we would need the OLYMPUS test
    suite to pass cleanly on Mac x86 before we could ship releases again.

* Changes:
  * The minimum NumPy version is now 1.25. NumPy 1.25 will remain the minimum
    supported version until June 2025.
  * The minimum SciPy version is now 1.11. SciPy 1.11 will remain the minimum
    supported version until June 2025.
  * {func}`olympus.numpy.einsum` now defaults to `optimize='auto'` rather than
    `optimize='optimal'`. This avoids exponentially-scaling trace-time in
    the case of many arguments ({olympus-issue}`#25214`).
  * {func}`olympus.numpy.linalg.solve` no longer supports batched 1D arguments
    on the right hand side. To recover the previous behavior in these cases,
    use `solve(a, b[..., None]).squeeze(-1)`.

* New Features
  * {func}`olympus.numpy.fft.fftn`, {func}`olympus.numpy.fft.rfftn`,
    {func}`olympus.numpy.fft.ifftn`, and {func}`olympus.numpy.fft.irfftn` now support
    transforms in more than 3 dimensions, which was previously the limit. See
    {olympus-issue}`#25606` for more details.
  * Support added for user defined state in the FFI via the new
    {func}`olympus.ffi.register_ffi_type_id` function.
  * The AOT lowering `.as_text()` method now supports the `debug_info` option
    to include debugging information, e.g., source location, in the output.

* Deprecations
  * From {mod}`olympus.interpreters.xla`, `abstractify` and `pytype_aval_mappings`
    are now deprecated, having been replaced by symbols of the same name
    in {mod}`olympus.core`.
  * {func}`olympus.scipy.special.lpmn` and {func}`olympus.scipy.special.lpmn_values`
    are deprecated, following their deprecation in SciPy v1.15.0. There are
    no plans to replace these deprecated functions with new APIs.
  * The {mod}`olympus.extend.ffi` submodule was moved to {mod}`olympus.ffi`, and the
    previous import path is deprecated.

* Deletions
  * `olympus_enable_memories` flag has been deleted and the behavior of that flag
    is on by default.
  * From `olympus.lib.xla_client`, the previously-deprecated `Device` and
    `XlaRuntimeError` symbols have been removed; instead use `olympus.Device`
    and `olympus.errors.OlympusRuntimeError` respectively.
  * The `olympus.experimental.array_api` module has been removed after being
    deprecated in OLYMPUS v0.4.32. Since that release, {mod}`olympus.numpy` supports
    the array API directly.

## olympus 0.4.38 (Dec 17, 2024)

* Breaking Changes
  * `XlaExecutable.cost_analysis` now returns a `dict[str, float]` (instead of a
    single-element `list[dict[str, float]]`).

* Changes:
  * `olympus.tree.flatten_with_path` and `olympus.tree.map_with_path` are added
    as shortcuts of the corresponding `tree_util` functions.

* Deprecations
  * a number of APIs in the internal `olympus.core` namespace have been deprecated.
    Most were no-ops, were little-used, or can be replaced by APIs of the same
    name in {mod}`olympus.extend.core`; see the documentation for {mod}`olympus.extend`
    for information on the compatibility guarantees of these semi-public extensions.
  * Several previously-deprecated APIs have been removed, including:
    * from {mod}`olympus.core`: `check_eqn`, `check_type`,  `check_valid_olympustype`, and
      `non_negative_dim`.
    * from {mod}`olympus.lib.xla_bridge`: `xla_client` and `default_backend`.
    * from {mod}`olympus.lib.xla_client`: `_xla` and `bfloat16`.
    * from {mod}`olympus.numpy`: `round_`.

* New Features
  * {func}`olympus.export.export` can be used for device-polymorphic export with
    shardings constructed with {func}`olympus.sharding.AbstractMesh`.
    See the [olympus.export documentation](https://docs.olympus.dev/en/latest/export/export.html#device-polymorphic-export).
  * Added {func}`olympus.lax.split`. This is a primitive version of
    {func}`olympus.numpy.split`, added because it yields a more compact
    transpose during automatic differentiation.

## olympus 0.4.37 (Dec 9, 2024)

This is a patch release of olympus 0.4.36. Only "olympus" was released at this version.

* Bug fixes
  * Fixed a bug where `jit` would error if an argument was named `f` (#25329).
  * Fix a bug that will throw `index out of range` error in
    {func}`olympus.lax.while_loop` if the user register pytree node class with
    different aux data for the flatten and flatten_with_path.
  * Pinned a new libtpu release (0.0.6) that fixes a compiler bug on TPU v6e.

## olympus 0.4.36 (Dec 5, 2024)

* Breaking Changes
  * This release lands "stackless", an internal change to OLYMPUS's tracing
    machinery. We made trace dispatch purely a function of context rather than a
    function of both context and data. This let us delete a lot of machinery for
    managing data-dependent tracing: levels, sublevels, `post_process_call`,
    `new_base_main`, `custom_bind`, and so on. The change should only affect
    users that use OLYMPUS internals.

    If you do use OLYMPUS internals then you may need to
    update your code (see
    https://github.com/olympus-ml/olympus/commit/c36e1f7c1ad4782060cbc8e8c596d85dfb83986f
    for clues about how to do this). There might also be version skew
    issues with OLYMPUS libraries that do this. If you find this change breaks your
    non-OLYMPUS-internals-using code then try the
    `config.olympus_data_dependent_tracing_fallback` flag as a workaround, and if
    you need help updating your code then please file a bug.
  * {func}`olympus.experimental.olympus2tf.convert` with `native_serialization=False`
    or with `enable_xla=False` have been deprecated since July 2024, with
    OLYMPUS version 0.4.31. Now we removed support for these use cases. `olympus2tf`
    with native serialization will still be supported.
  * In `olympus.interpreters.xla`, the `xb`, `xc`, and `xe` symbols have been removed
    after being deprecated in OLYMPUS v0.4.31. Instead use `xb = olympus.lib.xla_bridge`,
    `xc = olympus.lib.xla_client`, and `xe = olympus.lib.xla_extension`.
  * The deprecated module `olympus.experimental.export` has been removed. It was replaced
    by {mod}`olympus.export` in OLYMPUS v0.4.30. See the [migration guide](https://docs.olympus.dev/en/latest/export/export.html#migration-guide-from-olympus-experimental-export)
    for information on migrating to the new API.
  * The `initial` argument to {func}`olympus.nn.softmax` and {func}`olympus.nn.log_softmax`
    has been removed, after being deprecated in v0.4.27.
  * Calling `np.asarray` on typed PRNG keys (i.e. keys produced by {func}`olympus.random.key`)
    now raises an error. Previously, this returned a scalar object array.
  * The following deprecated methods and functions in {mod}`olympus.export` have
    been removed:
      * `olympus.export.DisabledSafetyCheck.shape_assertions`: it had no effect
        already.
      * `olympus.export.Exported.lowering_platforms`: use `platforms`.
      * `olympus.export.Exported.mlir_module_serialization_version`:
        use `calling_convention_version`.
      * `olympus.export.Exported.uses_shape_polymorphism`:
         use `uses_global_constants`.
      * the `lowering_platforms` kwarg for {func}`olympus.export.export`: use
        `platforms` instead.
  * The kwargs `symbolic_scope` and `symbolic_constraints` from
    {func}`olympus.export.symbolic_args_specs` have been removed. They were
    deprecated in June 2024. Use `scope` and `constraints` instead.
  * Hashing of tracers, which has been deprecated since version 0.4.30, now
    results in a `TypeError`.
  * Refactor: OLYMPUS build CLI (build/build.py) now uses a subcommand structure and
    replaces previous build.py usage. Run `python build/build.py --help` for
    more details. Brief overview of the new subcommand options:
    * `build`: Builds OLYMPUS wheel packages. For e.g., `python build/build.py build --wheels=olympuslib,olympus-cuda-pjrt`
    * `requirements_update`: Updates requirements_lock.txt files.
  * {func}`olympus.scipy.linalg.toeplitz` now does implicit batching on multi-dimensional
    inputs. To recover the previous behavior, you can call {func}`olympus.numpy.ravel`
    on the function inputs.
  * {func}`olympus.scipy.special.gamma` and {func}`olympus.scipy.special.gammasgn` now
    return NaN for negative integer inputs, to match the behavior of SciPy from
    https://github.com/scipy/scipy/pull/21827.
  * `olympus.clear_backends` was removed after being deprecated in v0.4.26.
  * We removed the custom call "__gpu$xla.gpu.triton" from the list of custom
    call that we guarantee export stability. This is because this custom call
    relies on Triton IR, which is not guaranteed to be stable. If you need
    to export code that uses this custom call, you can use the `disabled_checks`
    parameter. See more details in the [documentation](https://docs.olympus.dev/en/latest/export/export.html#compatibility-guarantees-for-custom-calls).

* New Features
  * {func}`olympus.jit` got a new `compiler_options: dict[str, Any]` argument, for
    passing compilation options to XLA. For the moment it's undocumented and
    may be in flux.
  * {func}`olympus.tree_util.register_dataclass` now allows metadata fields to be
    declared inline via {func}`dataclasses.field`. See the function documentation
    for examples.
  * Added {func}`olympus.numpy.put_along_axis`.
  * {func}`olympus.lax.linalg.eig` and the related `olympus.numpy` functions
    ({func}`olympus.numpy.linalg.eig` and {func}`olympus.numpy.linalg.eigvals`) are now
    supported on GPU. See {olympus-issue}`#24663` for more details.
  * Added two new configuration flags, `olympus_exec_time_optimization_effort` and `olympus_memory_fitting_effort`, to control the amount of effort the compiler spends minimizing execution time and memory usage, respectively.  Valid values are between -1.0 and 1.0, default is 0.0.

* Bug fixes
  * Fixed a bug where the GPU implementations of LU and QR decomposition would
    result in an indexing overflow for batch sizes close to int32 max. See
    {olympus-issue}`#24843` for more details.

* Deprecations
  * `olympus.lib.xla_extension.ArrayImpl` and `olympus.lib.xla_client.ArrayImpl` are deprecated;
    use `olympus.Array` instead.
  * `olympus.lib.xla_extension.XlaRuntimeError` is deprecated; use `olympus.errors.OlympusRuntimeError`
    instead.

## olympus 0.4.35 (Oct 22, 2024)

* Breaking Changes
  * {func}`olympus.numpy.isscalar` now returns True for any array-like object with
    zero dimensions. Previously it only returned True for zero-dimensional
    array-like objects with a weak dtype.
  * `olympus.experimental.host_callback` has been deprecated since March 2024, with
    OLYMPUS version 0.4.26. Now we removed it.
    See {olympus-issue}`#20385` for a discussion of alternatives.

* Changes:
  * `olympus.lax.FftType` was introduced as a public name for the enum of FFT
    operations. The semi-public API `olympus.lib.xla_client.FftType` has been
    deprecated.
  * TPU: OLYMPUS now installs TPU support from the `libtpu` package rather than
    `libtpu-nightly`. For the next few releases OLYMPUS will pin an empty version of
    `libtpu-nightly` as well as `libtpu` to ease the transition; that dependency
    will be removed in Q1 2025.

* Deprecations:
  * The semi-public API `olympus.lib.xla_client.PaddingType` has been deprecated.
    No OLYMPUS APIs consume this type, so there is no replacement.
  * The default behavior of {func}`olympus.pure_callback` and
    {func}`olympus.extend.ffi.ffi_call` under `vmap` has been deprecated and so has
    the `vectorized` parameter to those functions. The `vmap_method` parameter
    should be used instead for better defined behavior. See the discussion in
    {olympus-issue}`#23881` for more details.
  * The semi-public API `olympus.lib.xla_client.register_custom_call_target` has
    been deprecated. Use the OLYMPUS FFI instead.
  * The semi-public APIs `olympus.lib.xla_client.dtype_to_etype`,
    `olympus.lib.xla_client.ops`,
    `olympus.lib.xla_client.shape_from_pyval`, `olympus.lib.xla_client.PrimitiveType`,
    `olympus.lib.xla_client.Shape`, `olympus.lib.xla_client.XlaBuilder`, and
    `olympus.lib.xla_client.XlaComputation` have been deprecated. Use StableHLO
    instead.

## olympus 0.4.34 (October 4, 2024)

* New Functionality
  * This release includes wheels for Python 3.13. Free-threading mode is not yet
    supported.
  * `olympus.errors.OlympusRuntimeError` has been added as a public alias for the
    formerly private `XlaRuntimeError` type.

* Breaking changes
  * `olympus_pmap_no_rank_reduction` flag is set to `True` by default.
    * array[0] on a pmap result now introduces a reshape (use array[0:1]
      instead).
    * The per-shard shape (accessible via olympus_array.addressable_shards or
      olympus_array.addressable_data(0)) now has a leading (1, ...). Update code
      that directly accesses shards accordingly. The rank of the per-shard-shape
      now matches that of the global shape which is the same behavior as jit.
      This avoids costly reshapes when passing results from pmap into jit.
  * `olympus.experimental.host_callback` has been deprecated since March 2024, with
    OLYMPUS version 0.4.26. Now we set the default value of the
    `--olympus_host_callback_legacy` configuration value to `True`, which means that
    if your code uses `olympus.experimental.host_callback` APIs, those API calls
    will be implemented in terms of the new `olympus.experimental.io_callback` API.
    If this breaks your code, for a very limited time, you can set the
    `--olympus_host_callback_legacy` to `True`. Soon we will remove that
    configuration option, so you should instead transition to using the
    new OLYMPUS callback APIs. See {olympus-issue}`#20385` for a discussion.

* Deprecations
  * In {func}`olympus.numpy.trim_zeros`, non-arraylike arguments or arraylike
    arguments with `ndim != 1` are now deprecated, and in the future will result
    in an error.
  * Internal pretty-printing tools `olympus.core.pp_*` have been removed, after
    being deprecated in OLYMPUS v0.4.30.
  * `olympus.lib.xla_client.Device` is deprecated; use `olympus.Device` instead.
  * `olympus.lib.xla_client.XlaRuntimeError` has been deprecated. Use
    `olympus.errors.OlympusRuntimeError` instead.

* Deletion:
  * `olympus.xla_computation` is deleted. It's been 3 months since it's deprecation
    in 0.4.30 OLYMPUS release.
    Please use the AOT APIs to get the same functionality as `olympus.xla_computation`.
    * `olympus.xla_computation(fn)(*args, **kwargs)` can be replaced with
      `olympus.jit(fn).lower(*args, **kwargs).compiler_ir('hlo')`.
    * You can also use `.out_info` property of `olympus.stages.Lowered` to get the
      output information (like tree structure, shape and dtype).
    * For cross-backend lowering, you can replace
      `olympus.xla_computation(fn, backend='tpu')(*args, **kwargs)` with
      `olympus.jit(fn).trace(*args, **kwargs).lower(lowering_platforms=('tpu',)).compiler_ir('hlo')`.
  * {class}`olympus.ShapeDtypeStruct` no longer accepts the `named_shape` argument.
    The argument was only used by `xmap` which was removed in 0.4.31.
  * `olympus.tree.map(f, None, non-None)`, which previously emitted a
    `DeprecationWarning`, now raises an error in a future version of olympus. `None`
    is only a tree-prefix of itself. To preserve the current behavior, you can
    ask `olympus.tree.map` to treat `None` as a leaf value by writing:
    `olympus.tree.map(lambda x, y: None if x is None else f(x, y), a, b, is_leaf=lambda x: x is None)`.
  * `olympus.sharding.XLACompatibleSharding` has been removed. Please use
    `olympus.sharding.Sharding`.

* Bug fixes
  * Fixed a bug where {func}`olympus.numpy.cumsum` would produce incorrect outputs
    if a non-boolean input was provided and `dtype=bool` was specified.
  * Edit implementation of {func}`olympus.numpy.ldexp` to get correct gradient.

## olympus 0.4.33 (September 16, 2024)

This is a patch release on top of olympus 0.4.32, that fixes two bugs found in that
release.

A TPU-only data corruption bug was found in the version of libtpu pinned by
OLYMPUS 0.4.32, which manifested only if multiple TPU slices were present in the
same job, for example, if training on multiple v5e slices.
This release fixes that issue by pinning a fixed version of `libtpu`.

This release fixes an inaccurate result for F64 tanh on CPU (#23590).

## olympus 0.4.32 (September 11, 2024)

Note: This release was yanked from PyPi because of a data corruption bug on TPU.
See the 0.4.33 release notes for more details.

* New Functionality
  * Added {func}`olympus.extend.ffi.ffi_call` and {func}`olympus.extend.ffi.ffi_lowering`
    to support the use of the new {ref}`ffi-tutorial` to interface with custom
    C++ and CUDA code from OLYMPUS.

* Changes
  * `olympus_enable_memories` flag is set to `True` by default.
  * {mod}`olympus.numpy` now supports v2023.12 of the Python Array API Standard.
    See {ref}`python-array-api` for more information.
  * Computations on the CPU backend may now be dispatched asynchronously in
    more cases. Previously non-parallel computations were always dispatched
    synchronously. You can recover the old behavior by setting
    `olympus.config.update('olympus_cpu_enable_async_dispatch', False)`.
  * Added new {func}`olympus.process_indices` function to replace the
    `olympus.process_indexs()` function that was deprecated in OLYMPUS v0.2.13.
  * To align with the behavior of `numpy.fabs`, `olympus.numpy.fabs` has been
    modified to no longer support `complex dtypes`.
  * ``olympus.tree_util.register_dataclass`` now checks that ``data_fields``
    and ``meta_fields`` includes all dataclass fields with ``init=True``
    and only them, if ``nodetype`` is a dataclass.
  * Several {mod}`olympus.numpy` functions now have full {class}`~olympus.numpy.ufunc`
    interfaces, including {obj}`~olympus.numpy.add`, {obj}`~olympus.numpy.multiply`,
    {obj}`~olympus.numpy.bitwise_and`, {obj}`~olympus.numpy.bitwise_or`,
    {obj}`~olympus.numpy.bitwise_xor`, {obj}`~olympus.numpy.logical_and`,
    {obj}`~olympus.numpy.logical_and`, and {obj}`~olympus.numpy.logical_and`.
    In future releases we plan to expand these to other ufuncs.
  * Added {func}`olympus.lax.optimization_barrier`, which allows users to prevent
    compiler optimizations such as common-subexpression elimination and to
    control scheduling.

* Breaking changes
  * The MHLO MLIR dialect (`olympus.extend.mlir.mhlo`) has been removed. Use the
    `stablehlo` dialect instead.

* Deprecations
  * Complex inputs to {func}`olympus.numpy.clip` and {func}`olympus.numpy.hypot` are
    no longer allowed, after being deprecated since OLYMPUS v0.4.27.
  * Deprecated the following APIs:
    * `olympus.lib.xla_bridge.xla_client`: use {mod}`olympus.lib.xla_client` directly.
    * `olympus.lib.xla_bridge.get_backend`: use {func}`olympus.extend.backend.get_backend`.
    * `olympus.lib.xla_bridge.default_backend`: use {func}`olympus.extend.backend.default_backend`.
  * The `olympus.experimental.array_api` module is deprecated, and importing it is no
    longer required to use the Array API. `olympus.numpy` supports the array API
    directly; see {ref}`python-array-api` for more information.
  * The internal utilities `olympus.core.check_eqn`, `olympus.core.check_type`, and
    `olympus.core.check_valid_olympustype` are now deprecated, and will be removed in
    the future.
  * `olympus.numpy.round_` has been deprecated, following removal of the corresponding
    API in NumPy 2.0. Use {func}`olympus.numpy.round` instead.
  * Passing a DLPack capsule to {func}`olympus.dlpack.from_dlpack` is deprecated.
    The argument to {func}`olympus.dlpack.from_dlpack` should be an array from
    another framework that implements the ``__dlpack__`` protocol.

## olympuslib 0.4.32 (September 11, 2024)

Note: This release was yanked from PyPi because of a data corruption bug on TPU.
See the 0.4.33 release notes for more details.

* Breaking changes
  * This release of olympuslib switched to a new version of the CPU backend, which
    should compile faster and leverage parallelism better. If you experience
    any problems due to this change, you can temporarily enable the old CPU
    backend by setting the environment variable
    `XLA_FLAGS=--xla_cpu_use_thunk_runtime=false`. If you need to do this,
    please file a OLYMPUS bug with instructions to reproduce.
  * Hermetic CUDA support is added.
    Hermetic CUDA uses a specific downloadable version of CUDA instead of the
    user’s locally installed CUDA. Bazel will download CUDA, CUDNN and NCCL
    distributions, and then use CUDA libraries and tools as dependencies in
    various Bazel targets. This enables more reproducible builds for OLYMPUS and its
    supported CUDA versions.

* Changes
  * SparseCore profiling is added.
    * OLYMPUS now supports profiling [SparseCore](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#sparsecore) on TPUv5p chips. These traces will be viewable in Tensorboard Profiler's [TraceViewer](https://www.tensorflow.org/guide/profiler#trace_viewer).

## olympus 0.4.31 (July 29, 2024)

* Deletion
  * xmap has been deleted. Please use {func}`shard_map` as the replacement.

* Changes
  * The minimum CuDNN version is v9.1. This was true in previous releases also,
    but we now declare this version constraint formally.
  * The minimum Python version is now 3.10. 3.10 will remain the minimum
    supported version until July 2025.
  * The minimum NumPy version is now 1.24. NumPy 1.24 will remain the minimum
    supported version until December 2024.
  * The minimum SciPy version is now 1.10. SciPy 1.10 will remain the minimum
    supported version until January 2025.
  * {func}`olympus.numpy.ceil`, {func}`olympus.numpy.floor` and {func}`olympus.numpy.trunc` now return the output
    of the same dtype as the input, i.e. no longer upcast integer or boolean inputs to floating point.
  * `libdevice.10.bc` is no longer bundled with CUDA wheels. It must be
    installed either as a part of local CUDA installation, or via NVIDIA's CUDA
    pip wheels.
  * {class}`olympus.experimental.pallas.BlockSpec` now expects `block_shape` to
    be passed *before* `index_map`. The old argument order is deprecated and
    will be removed in a future release.
  * Updated the repr of gpu devices to be more consistent
    with TPUs/CPUs. For example, `cuda(id=0)` will now be `CudaDevice(id=0)`.
  * Added the `device` property and `to_device` method to {class}`olympus.Array`, as
    part of OLYMPUS's [Array API](https://data-apis.org/array-api) support.
* Deprecations
  * Removed a number of previously-deprecated internal APIs related to
    polymorphic shapes. From {mod}`olympus.core`: removed `canonicalize_shape`,
    `dimension_as_value`, `definitely_equal`, and `symbolic_equal_dim`.
  * HLO lowering rules should no longer wrap singleton ir.Values in tuples.
    Instead, return singleton ir.Values unwrapped. Support for wrapped values
    will be removed in a future version of OLYMPUS.
  * {func}`olympus.experimental.olympus2tf.convert` with `native_serialization=False`
    or `enable_xla=False` is now deprecated and this support will be removed in
    a future version.
    Native serialization has been the default since OLYMPUS 0.4.16 (September 2023).
  * The previously-deprecated function `olympus.random.shuffle` has been removed;
    instead use `olympus.random.permutation` with `independent=True`.

## olympuslib 0.4.31 (July 29, 2024)

* Bug fixes
  * Fixed a bug that meant that negative static_argnums to a jit were mishandled
    by the jit dispatch fast path.
  * Fixed a bug that meant triangular solves of batches of singular matrices
    produce nonsensical finite values, instead of inf or nan (#3589, #15429).

## olympus 0.4.30 (June 18, 2024)

* Changes
  * OLYMPUS supports ml_dtypes >= 0.2. In 0.4.29 release, the ml_dtypes version was
    bumped to 0.4.0 but this has been rolled back in this release to give users
    of both TensorFlow and OLYMPUS more time to migrate to a newer TensorFlow
    release.
  * `olympus.experimental.mesh_utils` can now create an efficient mesh for TPU v5e.
  * olympus now depends on olympuslib directly. This change was enabled by the CUDA
    plugin switch: there are no longer multiple olympuslib variants. You can install
    a CPU-only olympus with `pip install olympus`, no extras required.
  * Added an API for exporting and serializing OLYMPUS functions. This used
    to exist in `olympus.experimental.export` (which is being deprecated),
    and will now live in `olympus.export`.
    See the [documentation](https://docs.olympus.dev/en/latest/export/index.html).

* Deprecations
  * Internal pretty-printing tools `olympus.core.pp_*` are deprecated, and will be removed
    in a future release.
  * Hashing of tracers is deprecated, and will lead to a `TypeError` in a future OLYMPUS
    release. This previously was the case, but there was an inadvertent regression in
    the last several OLYMPUS releases.
  * `olympus.experimental.export` is deprecated. Use {mod}`olympus.export` instead.
    See the [migration guide](https://docs.olympus.dev/en/latest/export/export.html#migration-guide-from-olympus-experimental-export).
  * Passing an array in place of a dtype is now deprecated in most cases; e.g. for arrays
    `x` and `y`, `x.astype(y)` will raise a warning. To silence it use `x.astype(y.dtype)`.
  * `olympus.xla_computation` is deprecated and will be removed in a future release.
    Please use the AOT APIs to get the same functionality as `olympus.xla_computation`.
    * `olympus.xla_computation(fn)(*args, **kwargs)` can be replaced with
      `olympus.jit(fn).lower(*args, **kwargs).compiler_ir('hlo')`.
    * You can also use `.out_info` property of `olympus.stages.Lowered` to get the
      output information (like tree structure, shape and dtype).
    * For cross-backend lowering, you can replace
      `olympus.xla_computation(fn, backend='tpu')(*args, **kwargs)` with
      `olympus.jit(fn).trace(*args, **kwargs).lower(lowering_platforms=('tpu',)).compiler_ir('hlo')`.


## olympuslib 0.4.30 (June 18, 2024)

  * Support for monolithic CUDA olympuslibs has been dropped. You must use the
    plugin-based installation (`pip install olympus[cuda12]` or
    `pip install olympus[cuda12_local]`).

## olympus 0.4.29 (June 10, 2024)

* Changes
  * We anticipate that this will be the last release of OLYMPUS and olympuslib
    supporting a monolithic CUDA olympuslib. Future releases will use the CUDA
    plugin olympuslib (e.g. `pip install olympus[cuda12]`).
  * OLYMPUS now requires ml_dtypes version 0.4.0 or newer.
  * Removed backwards-compatibility support for old usage of the
    `olympus.experimental.export` API. It is not possible anymore to use
    `from olympus.experimental.export import export`, and instead you should use
    `from olympus.experimental import export`.
    The removed functionality has been deprecated since 0.4.24.
  * Added `is_leaf` argument to {func}`olympus.tree.all` & {func}`olympus.tree_util.tree_all`.

* Deprecations
  * `olympus.sharding.XLACompatibleSharding` is deprecated. Please use
    `olympus.sharding.Sharding`.
  * `olympus.experimental.Exported.in_shardings` has been renamed as
    `olympus.experimental.Exported.in_shardings_hlo`. Same for `out_shardings`.
    The old names will be removed after 3 months.
  * Removed a number of previously-deprecated APIs:
    * from {mod}`olympus.core`: `non_negative_dim`, `DimSize`, `Shape`
    * from {mod}`olympus.lax`: `tie_in`
    * from {mod}`olympus.nn`: `normalize`
    * from {mod}`olympus.interpreters.xla`: `backend_specific_translations`,
      `translations`, `register_translation`, `xla_destructure`,
      `TranslationRule`, `TranslationContext`, `XlaOp`.
  * The ``tol`` argument of {func}`olympus.numpy.linalg.matrix_rank` is being
    deprecated and will soon be removed. Use `rtol` instead.
  * The ``rcond`` argument of {func}`olympus.numpy.linalg.pinv` is being
    deprecated and will soon be removed. Use `rtol` instead.
  * The deprecated `olympus.config` submodule has been removed. To configure OLYMPUS
    use `import olympus` and then reference the config object via `olympus.config`.
  * {mod}`olympus.random` APIs no longer accept batched keys, where previously
    some did unintentionally. Going forward, we recommend explicit use of
    {func}`olympus.vmap` in such cases.
  * In {func}`olympus.scipy.special.beta`, the `x` and `y` parameters have been
    renamed to `a` and `b` for consistency with other `beta` APIs.

* New Functionality
  * Added {func}`olympus.experimental.Exported.in_shardings_olympus` to construct
    shardings that can be used with the OLYMPUS APIs from the HloShardings
    that are stored in the `Exported` objects.

## olympuslib 0.4.29 (June 10, 2024)

* Bug fixes
  * Fixed a bug where XLA sharded some concatenation operations incorrectly,
    which manifested as an incorrect output for cumulative reductions (#21403).
  * Fixed a bug where XLA:CPU miscompiled certain matmul fusions
    (https://github.com/openxla/xla/pull/13301).
  * Fixes a compiler crash on GPU (https://github.com/olympus-ml/olympus/issues/21396).

* Deprecations
  * `olympus.tree.map(f, None, non-None)` now emits a `DeprecationWarning`, and will
    raise an error in a future version of olympus. `None` is only a tree-prefix of
    itself. To preserve the current behavior, you can ask `olympus.tree.map` to
    treat `None` as a leaf value by writing:
    `olympus.tree.map(lambda x, y: None if x is None else f(x, y), a, b, is_leaf=lambda x: x is None)`.

## olympus 0.4.28 (May 9, 2024)

* Bug fixes
  * Reverted a change to `make_olympuspr` that was breaking Equinox (#21116).

* Deprecations & removals
  * The ``kind`` argument to {func}`olympus.numpy.sort` and {func}`olympus.numpy.argsort`
    is now removed. Use `stable=True` or `stable=False` instead.
  * Removed ``get_compute_capability`` from the ``olympus.experimental.pallas.gpu``
    module. Use the ``compute_capability`` attribute of a GPU device, returned
    by {func}`olympus.devices` or {func}`olympus.local_devices`, instead.
  * The ``newshape`` argument to {func}`olympus.numpy.reshape`is being deprecated
    and will soon be removed. Use `shape` instead.

* Changes
  * The minimum olympuslib version of this release is 0.4.27.

## olympuslib 0.4.28 (May 9, 2024)

* Bug fixes
  * Fixes a memory corruption bug in the type name of Array and JIT Python
    objects in Python 3.10 or earlier.
  * Fixed a warning `'+ptx84' is not a recognized feature for this target`
    under CUDA 12.4.
  * Fixed a slow compilation problem on CPU.

* Changes
  * The Windows build is now built with Clang instead of MSVC.


## olympus 0.4.27 (May 7, 2024)

* New Functionality
  * Added {func}`olympus.numpy.unstack` and {func}`olympus.numpy.cumulative_sum`,
    following their addition in the array API 2023 standard, soon to be
    adopted by NumPy.
  * Added a new config option `olympus_cpu_collectives_implementation` to select the
    implementation of cross-process collective operations used by the CPU backend.
    Choices available are `'none'`(default), `'gloo'` and `'mpi'` (requires olympuslib 0.4.26).
    If set to `'none'`, cross-process collective operations are disabled.

* Changes
  * {func}`olympus.pure_callback`, {func}`olympus.experimental.io_callback`
    and {func}`olympus.debug.callback` now use {class}`olympus.Array` instead
    of {class}`np.ndarray`. You can recover the old behavior by transforming
    the arguments via `olympus.tree.map(np.asarray, args)` before passing them
    to the callback.
  * `complex_arr.astype(bool)` now follows the same semantics as NumPy, returning
    False where `complex_arr` is equal to `0 + 0j`, and True otherwise.
  * `core.Token` now is a non-trivial class which wraps a `olympus.Array`. It could
    be created and threaded in and out of computations to build up dependency.
    The singleton object `core.token` has been removed, users now should create
    and use fresh `core.Token` objects instead.
  * On GPU, the Threefry PRNG implementation no longer lowers to a kernel call
    by default. This choice can improve runtime memory usage at a compile-time
    cost. Prior behavior, which produces a kernel call, can be recovered with
    `olympus.config.update('olympus_threefry_gpu_kernel_lowering', True)`. If the new
    default causes issues, please file a bug. Otherwise, we intend to remove
    this flag in a future release.

* Deprecations & Removals
  * Pallas now exclusively uses XLA for compiling kernels on GPU. The old
    lowering pass via Triton Python APIs has been removed and the
    `OLYMPUS_TRITON_COMPILE_VIA_XLA` environment variable no longer has any effect.
  * {func}`olympus.numpy.clip` has a new argument signature: `a`, `a_min`, and
    `a_max` are deprecated in favor of `x` (positional only), `min`, and
    `max` ({olympus-issue}`20550`).
  * The `device()` method of OLYMPUS arrays has been removed, after being deprecated
    since OLYMPUS v0.4.21. Use `arr.devices()` instead.
  * The `initial` argument to {func}`olympus.nn.softmax` and {func}`olympus.nn.log_softmax`
    is deprecated; empty inputs to softmax are now supported without setting this.
  * In {func}`olympus.jit`, passing invalid `static_argnums` or `static_argnames`
    now leads to an error rather than a warning.
  * The minimum olympuslib version is now 0.4.23.
  * The {func}`olympus.numpy.hypot` function now issues a deprecation warning when
    passing complex-valued inputs to it. This will raise an error when the
    deprecation is completed.
  * Scalar arguments to {func}`olympus.numpy.nonzero`, {func}`olympus.numpy.where`, and
    related functions now raise an error, following a similar change in NumPy.
  * The config option `olympus_cpu_enable_gloo_collectives` is deprecated.
    Use `olympus.config.update('olympus_cpu_collectives_implementation', 'gloo')` instead.
  * The `olympus.Array.device_buffer` and `olympus.Array.device_buffers` methods have
    been removed after being deprecated in OLYMPUS v0.4.22. Instead use
    {attr}`olympus.Array.addressable_shards` and {meth}`olympus.Array.addressable_data`.
  * The `condition`, `x`, and `y` parameters of `olympus.numpy.where` are now
    positional-only, following deprecation of the keywords in OLYMPUS v0.4.21.
  * Non-array arguments to functions in {mod}`olympus.lax.linalg` now must be
    specified by keyword. Previously, this raised a DeprecationWarning.
  * Array-like arguments are now required in several {func}`olympus.numpy` APIs,
    including {func}`~olympus.numpy.apply_along_axis`,
    {func}`~olympus.numpy.apply_over_axes`, {func}`~olympus.numpy.inner`,
    {func}`~olympus.numpy.outer`, {func}`~olympus.numpy.cross`,
    {func}`~olympus.numpy.kron`, and {func}`~olympus.numpy.lexsort`.

* Bug fixes
  * {func}`olympus.numpy.astype` will now always return a copy when `copy=True`.
    Previously, no copy would be made when the output array would have the same
    dtype as the input array. This may result in some increased memory usage.
    The default value is set to `copy=False` to preserve backwards compatibility.

## olympuslib 0.4.27 (May 7, 2024)

## olympus 0.4.26 (April 3, 2024)

* New Functionality
  * Added {func}`olympus.numpy.trapezoid`, following the addition of this function in
    NumPy 2.0.

* Changes
  * Complex-valued {func}`olympus.numpy.geomspace` now chooses the logarithmic spiral
    branch consistent with that of NumPy 2.0.
  * The behavior of `lax.rng_bit_generator`, and in turn the `'rbg'`
    and `'unsafe_rbg'` PRNG implementations, under `olympus.vmap` [has
    changed](https://github.com/olympus-ml/olympus/issues/19085) so that
    mapping over keys results in random generation only from the first
    key in the batch.
  * Docs now use `olympus.random.key` for construction of PRNG key arrays
    rather than `olympus.random.PRNGKey`.

* Deprecations & Removals
  * {func}`olympus.tree_map` is deprecated; use `olympus.tree.map` instead, or for backward
    compatibility with older OLYMPUS versions, use {func}`olympus.tree_util.tree_map`.
  * {func}`olympus.clear_backends` is deprecated as it does not necessarily do what
    its name suggests and can lead to unexpected consequences, e.g., it will not
    destroy existing backends and release corresponding owned resources. Use
    {func}`olympus.clear_caches` if you only want to clean up compilation caches.
    For backward compatibility or you really need to switch/reinitialize the
    default backend, use {func}`olympus.extend.backend.clear_backends`.
  * The `olympus.experimental.maps` module and `olympus.experimental.maps.xmap` are
    deprecated. Use `olympus.experimental.shard_map` or `olympus.vmap` with the
    `spmd_axis_name` argument for expressing SPMD device-parallel computations.
  * The `olympus.experimental.host_callback` module is deprecated.
    Use instead the [new OLYMPUS external callbacks](https://docs.olympus.dev/en/latest/notebooks/external_callbacks.html).
    Added `OLYMPUS_HOST_CALLBACK_LEGACY` flag to assist in the transition to the
    new callbacks. See {olympus-issue}`#20385` for a discussion.
  * Passing arguments to {func}`olympus.numpy.array_equal` and {func}`olympus.numpy.array_equiv`
    that cannot be converted to a OLYMPUS array now results in an exception.
  * The deprecated flag `olympus_parallel_functions_output_gda` has been removed.
    This flag was long deprecated and did nothing; its use was a no-op.
  * The previously-deprecated imports `olympus.interpreters.ad.config` and
    `olympus.interpreters.ad.source_info_util` have now been removed. Use `olympus.config`
    and `olympus.extend.source_info_util` instead.
  * OLYMPUS export does not support older serialization versions anymore. Version 9
    has been supported since October 27th, 2023 and has become the default
    since February 1, 2024.
    See [a description of the versions](https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md#native-serialization-versions).
    This change could break clients that set a specific
    OLYMPUS serialization version lower than 9.

## olympuslib 0.4.26 (April 3, 2024)

* Changes
  * OLYMPUS now supports CUDA 12.1 or newer only. Support for CUDA 11.8 has been
    dropped.
  * OLYMPUS now supports NumPy 2.0.

## olympus 0.4.25 (Feb 26, 2024)

* New Features
  * Added [CUDA Array
    Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
    import support (requires olympuslib 0.4.24).
  * OLYMPUS arrays now support NumPy-style scalar boolean indexing, e.g. `x[True]` or `x[False]`.
  * Added {mod}`olympus.tree` module, with a more convenient interface for referencing functions
    in {mod}`olympus.tree_util`.
  * {func}`olympus.tree.transpose` (i.e. {func}`olympus.tree_util.tree_transpose`) now accepts
    `inner_treedef=None`, in which case the inner treedef will be automatically inferred.

* Changes
  * Pallas now uses XLA instead of the Triton Python APIs to compile Triton
    kernels. You can revert to the old behavior by setting the
    `OLYMPUS_TRITON_COMPILE_VIA_XLA` environment variable to `"0"`.
  * Several deprecated APIs in {mod}`olympus.interpreters.xla` that were removed in v0.4.24
    have been re-added in v0.4.25, including `backend_specific_translations`,
    `translations`, `register_translation`, `xla_destructure`, `TranslationRule`,
    `TranslationContext`, and `XLAOp`. These are still considered deprecated, and
    will be removed again in the future when better replacements are available.
    Refer to {olympus-issue}`#19816` for discussion.

* Deprecations & Removals
  * {func}`olympus.numpy.linalg.solve` now shows a deprecation warning for batched 1D
    solves with `b.ndim > 1`. In the future these will be treated as batched 2D
    solves.
  * Conversion of a non-scalar array to a Python scalar now raises an error, regardless
    of the size of the array. Previously a deprecation warning was raised in the case of
    non-scalar arrays of size 1. This follows a similar deprecation in NumPy.
  * The previously deprecated configuration APIs have been removed
    following a standard 3 months deprecation cycle (see {ref}`api-compatibility`).
    These include
    * the `olympus.config.config` object and
    * the `define_*_state` and `DEFINE_*` methods of {data}`olympus.config`.
  * Importing the `olympus.config` submodule via `import olympus.config` is deprecated.
    To configure OLYMPUS use `import olympus` and then reference the config object
    via `olympus.config`.
  * The minimum olympuslib version is now 0.4.20.

## olympuslib 0.4.25 (Feb 26, 2024)

## olympus 0.4.24 (Feb 6, 2024)

* Changes

  * OLYMPUS lowering to StableHLO does not depend on physical devices anymore.
    If your primitive wraps custom_partitioning or OLYMPUS callbacks in the lowering
    rule i.e. function passed to `rule` parameter of `mlir.register_lowering` then add your
    primitive to `olympus._src.dispatch.prim_requires_devices_during_lowering` set.
    This is needed because custom_partitioning and OLYMPUS callbacks need physical
    devices to create `Sharding`s during lowering.
    This is a temporary state until we can create `Sharding`s without physical
    devices.
  * {func}`olympus.numpy.argsort` and {func}`olympus.numpy.sort` now support the `stable`
    and `descending` arguments.
  * Several changes to the handling of shape polymorphism (used in
    {mod}`olympus.experimental.olympus2tf` and {mod}`olympus.experimental.export`):
    * cleaner pretty-printing of symbolic expressions ({olympus-issue}`#19227`)
    * added the ability to specify symbolic constraints on the dimension variables.
      This makes shape polymorphism more expressive, and gives a way to workaround
      limitations in the reasoning about inequalities.
      See https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md#user-specified-symbolic-constraints.
    * with the addition of symbolic constraints ({olympus-issue}`#19235`) we now
      consider dimension variables from different scopes to be different, even
      if they have the same name. Symbolic expressions from different scopes
      cannot interact, e.g., in arithmetic operations.
      Scopes are introduced by {func}`olympus.experimental.olympus2tf.convert`,
      {func}`olympus.experimental.export.symbolic_shape`, {func}`olympus.experimental.export.symbolic_args_specs`.
      The scope of a symbolic expression `e` can be read with `e.scope` and passed
      into the above functions to direct them to construct symbolic expressions in
      a given scope.
      See https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md#user-specified-symbolic-constraints.
    * simplified and faster equality comparisons, where we consider two symbolic dimensions
      to be equal if the normalized form of their difference reduces to 0
      ({olympus-issue}`#19231`; note that this may result in user-visible behavior
        changes)
    * improved the error messages for inconclusive inequality comparisons
      ({olympus-issue}`#19235`).
    * the `core.non_negative_dim` API (introduced recently)
      was deprecated and `core.max_dim` and `core.min_dim` were introduced
      ({olympus-issue}`#18953`) to express `max` and `min` for symbolic dimensions.
      You can use `core.max_dim(d, 0)` instead of `core.non_negative_dim(d)`.
    * the `shape_poly.is_poly_dim` is deprecated in favor of `export.is_symbolic_dim`
      ({olympus-issue}`#19282`).
    * the `export.args_specs` is deprecated in favor of `export.symbolic_args_specs
      ({olympus-issue}`#19283`).
    * the `shape_poly.PolyShape` and `olympus2tf.PolyShape` are deprecated, use
      strings for polymorphic shapes specifications ({olympus-issue}`#19284`).
    * OLYMPUS default native serialization version is now 9. This is relevant
      for {mod}`olympus.experimental.olympus2tf` and {mod}`olympus.experimental.export`.
      See [description of version numbers](https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md#native-serialization-versions).
  * Refactored the API for `olympus.experimental.export`. Instead of
    `from olympus.experimental.export import export` you should use now
    `from olympus.experimental import export`. The old way of importing will
    continue to work for a deprecation period of 3 months.
  * Added {func}`olympus.scipy.stats.sem`.
  * {func}`olympus.numpy.unique` with `return_inverse = True` returns inverse indices
    reshaped to the dimension of the input, following a similar change to
    {func}`numpy.unique` in NumPy 2.0.
  * {func}`olympus.numpy.sign` now returns `x / abs(x)` for nonzero complex inputs. This is
    consistent with the behavior of {func}`numpy.sign` in NumPy version 2.0.
  * {func}`olympus.scipy.special.logsumexp` with `return_sign=True` now uses the NumPy 2.0
    convention for the complex sign, `x / abs(x)`. This is consistent with the behavior
    of {func}`scipy.special.logsumexp` in SciPy v1.13.
  * OLYMPUS now supports the bool DLPack type for both import and export.
    Previously bool values could not be imported and were exported as integers.

* Deprecations & Removals
  * A number of previously deprecated functions have been removed, following a
    standard 3+ month deprecation cycle (see {ref}`api-compatibility`).
    This includes:
    * From {mod}`olympus.core`: `TracerArrayConversionError`,
      `TracerIntegerConversionError`, `UnexpectedTracerError`,
      `as_hashable_function`, `collections`, `dtypes`, `lu`, `map`,
      `namedtuple`, `partial`, `pp`, `ref`, `safe_zip`, `safe_map`,
      `source_info_util`, `total_ordering`, `traceback_util`, `tuple_delete`,
      `tuple_insert`, and `zip`.
    * From {mod}`olympus.lax`: `dtypes`, `itertools`, `naryop`, `naryop_dtype_rule`,
      `standard_abstract_eval`, `standard_naryop`, `standard_primitive`,
      `standard_unop`, `unop`, and `unop_dtype_rule`.
    * The `olympus.linear_util` submodule and all its contents.
    * The `olympus.prng` submodule and all its contents.
    * From {mod}`olympus.random`: `PRNGKeyArray`, `KeyArray`, `default_prng_impl`,
      `threefry_2x32`, `threefry2x32_key`, `threefry2x32_p`, `rbg_key`, and
      `unsafe_rbg_key`.
    * From {mod}`olympus.tree_util`: `register_keypaths`, `AttributeKeyPathEntry`, and
      `GetItemKeyPathEntry`.
    * from {mod}`olympus.interpreters.xla`: `backend_specific_translations`, `translations`,
      `register_translation`, `xla_destructure`, `TranslationRule`, `TranslationContext`,
      `axis_groups`, `ShapedArray`, `ConcreteArray`, `AxisEnv`, `backend_compile`,
      and `XLAOp`.
    * from {mod}`olympus.numpy`: `NINF`, `NZERO`, `PZERO`, `row_stack`, `issubsctype`,
      `trapz`, and `in1d`.
    * from {mod}`olympus.scipy.linalg`: `tril` and `triu`.
  * The previously-deprecated method `PRNGKeyArray.unsafe_raw_array` has been
    removed. Use {func}`olympus.random.key_data` instead.
  * `bool(empty_array)` now raises an error rather than returning `False`. This
    previously raised a deprecation warning, and follows a similar change in NumPy.
  * Support for the mhlo MLIR dialect has been deprecated. OLYMPUS no longer uses
    the mhlo dialect, in favor of stablehlo. APIs that refer to "mhlo" will be
    removed in the future. Use the "stablehlo" dialect instead.
  * {mod}`olympus.random`: passing batched keys directly to random number generation functions,
    such as {func}`~olympus.random.bits`, {func}`~olympus.random.gamma`, and others, is deprecated
    and will emit a `FutureWarning`.  Use `olympus.vmap` for explicit batching.
  * {func}`olympus.lax.tie_in` is deprecated: it has been a no-op since OLYMPUS v0.2.0.

## olympuslib 0.4.24 (Feb 6, 2024)

* Changes

  * OLYMPUS now supports CUDA 12.3 and CUDA 11.8. Support for CUDA 12.2 has been
    dropped.
  * `cost_analysis` now works with cross-compiled `Compiled` objects (i.e. when
    using `.lower().compile()` with a topology object, e.g., to compile for
    Cloud TPU from a non-TPU computer).
  * Added [CUDA Array
    Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
    import support (requires olympus 0.4.25).

## olympus 0.4.23 (Dec 13, 2023)

## olympuslib 0.4.23 (Dec 13, 2023)

* Fixed a bug that caused verbose logging from the GPU compiler during
  compilation.

## olympus 0.4.22 (Dec 13, 2023)

* Deprecations
  * The `device_buffer` and `device_buffers` properties of OLYMPUS arrays are deprecated.
    Explicit buffers have been replaced by the more flexible array sharding interface,
    but the previous outputs can be recovered this way:
    * `arr.device_buffer` becomes `arr.addressable_data(0)`
    * `arr.device_buffers` becomes `[x.data for x in arr.addressable_shards]`

## olympuslib 0.4.22 (Dec 13, 2023)

## olympus 0.4.21 (Dec 4 2023)

* New Features
  * Added {obj}`olympus.nn.squareplus`.

* Changes
  * The minimum olympuslib version is now 0.4.19.
  * Released wheels are built now with clang instead of gcc.
  * Enforce that the device backend has not been initialized prior to calling `olympus.distributed.initialize()`.
  * Automate arguments to `olympus.distributed.initialize()` in cloud TPU environments.

* Deprecations
  * The previously-deprecated `sym_pos` argument has been removed from
    {func}`olympus.scipy.linalg.solve`. Use `assume_a='pos'` instead.
  * Passing `None` to {func}`olympus.array` or {func}`olympus.asarray`, either directly or
    within a list or tuple, is deprecated and now raises a {obj}`FutureWarning`.
    It currently is converted to NaN, and in the future will raise a {obj}`TypeError`.
  * Passing the `condition`, `x`, and `y` parameters to `olympus.numpy.where` by
    keyword arguments has been deprecated, to match `numpy.where`.
  * Passing arguments to {func}`olympus.numpy.array_equal` and {func}`olympus.numpy.array_equiv`
    that cannot be converted to a OLYMPUS array is deprecated and now raises a
    {obj}`DeprecationWaning`. Currently the functions return False, in the future this
    will raise an exception.
  * The `device()` method of OLYMPUS arrays is deprecated. Depending on the context, it may
    be replaced with one of the following:
    - {meth}`olympus.Array.devices` returns the set of all devices used by the array.
    - {attr}`olympus.Array.sharding` gives the sharding configuration used by the array.

## olympuslib 0.4.21 (Dec 4 2023)

* Changes
  * In preparation for adding distributed CPU support, OLYMPUS now treats CPU
    devices identically to GPU and TPU devices, that is:

    * `olympus.devices()` includes all devices present in a distributed job, even
      those not local to the current process. `olympus.local_devices()` still only
      includes devices local to the current process, so if the change to
      `olympus.devices()` breaks you, you most likely want to use
      `olympus.local_devices()` instead.
    * CPU devices now receive a globally unique ID number within a distributed
      job; previously CPU devices would receive a process-local ID number.
    * The `process_index` of each CPU device will now match any GPU or TPU
      devices within the same process; previously the `process_index` of a CPU
      device was always 0.

  * On NVIDIA GPU, OLYMPUS now prefers a Jacobi SVD solver for matrices up to
    1024x1024. The Jacobi solver appears faster than the non-Jacobi version.

* Bug fixes
  * Fixed error/hang when an array with non-finite values is passed to a
    non-symmetric eigendecomposition (#18226). Arrays with non-finite values now
    produce arrays full of NaNs as outputs.

## olympus 0.4.20 (Nov 2, 2023)

## olympuslib 0.4.20 (Nov 2, 2023)

* Bug fixes
  * Fixed some type confusion between E4M3 and E5M2 float8 types.

## olympus 0.4.19 (Oct 19, 2023)

* New Features
  * Added {obj}`olympus.typing.DTypeLike`, which can be used to annotate objects that
    are convertible to OLYMPUS dtypes.
  * Added `olympus.numpy.fill_diagonal`.

* Changes
  * OLYMPUS now requires SciPy 1.9 or newer.

* Bug fixes
  * Only process 0 in a multicontroller distributed OLYMPUS program will write
    persistent compilation cache entries. This fixes write contention if the
    cache is placed on a network file system such as GCS.
  * The version check for cusolver and cufft no longer considers the patch
    versions when determining if the installed version of these libraries is at
    least as new as the versions against which OLYMPUS was built.

## olympuslib 0.4.19 (Oct 19, 2023)

* Changes
  * olympuslib will now always prefer pip-installed NVIDIA CUDA libraries
    (nvidia-... packages) over any other CUDA installation if they are
    installed, including installations named in `LD_LIBRARY_PATH`. If this
    causes problems and the intent is to use a system-installed CUDA, the fix is
    to remove the pip installed CUDA library packages.

## olympus 0.4.18 (Oct 6, 2023)

## olympuslib 0.4.18 (Oct 6, 2023)

* Changes
  * CUDA olympuslibs now depend on the user to install a compatible NCCL version.
    If using the recommended `cuda12_pip` installation, NCCL should be installed
    automatically. Currently, NCCL 2.16 or newer is required.
  * We now provide Linux aarch64 wheels, both with and without NVIDIA GPU
    support.
  * {meth}`olympus.Array.item` now supports optional index arguments.

* Deprecations
  * A number of internal utilities and inadvertent exports in {mod}`olympus.lax` have
    been deprecated, and will be removed in a future release.
    * `olympus.lax.dtypes`: use `olympus.dtypes` instead.
    * `olympus.lax.itertools`: use `itertools` instead.
    * `naryop`, `naryop_dtype_rule`, `standard_abstract_eval`, `standard_naryop`,
      `standard_primitive`, `standard_unop`, `unop`, and `unop_dtype_rule` are
      internal utilities, now deprecated without replacement.

* Bug fixes
  * Fixed Cloud TPU regression where compilation would OOM due to smem.

## olympus 0.4.17 (Oct 3, 2023)

* New features
  * Added new {func}`olympus.numpy.bitwise_count` function, matching the API of the similar
    function recently added to NumPy.
* Deprecations
  * Removed the deprecated module `olympus.abstract_arrays` and all its contents.
  * Named key constructors in {mod}`olympus.random` are deprecated. Pass the `impl` argument
    to {func}`olympus.random.PRNGKey` or {func}`olympus.random.key` instead:
    * `random.threefry2x32_key(seed)` becomes `random.PRNGKey(seed, impl='threefry2x32')`
    * `random.rbg_key(seed)` becomes `random.PRNGKey(seed, impl='rbg')`
    * `random.unsafe_rbg_key(seed)` becomes `random.PRNGKey(seed, impl='unsafe_rbg')`
* Changes:
  * CUDA: OLYMPUS now verifies that the CUDA libraries it finds are at least as new
    as the CUDA libraries that OLYMPUS was built against. If older libraries are
    found, OLYMPUS raises an exception since that is preferable to mysterious
    failures and crashes.
  * Removed the "No GPU/TPU" found warning. Instead warn if, on Linux, an
    NVIDIA GPU or a Google TPU are found but not used and `--olympus_platforms` was
    not specified.
  * {func}`olympus.scipy.stats.mode` now returns a 0 count if the mode is taken
    across a size-0 axis, matching the behavior of `scipy.stats.mode` in SciPy
    1.11.
  * Most `olympus.numpy` functions and attributes now have fully-defined type stubs.
    Previously many of these were treated as `Any` by static type checkers like
    `mypy` and `pytype`.

## olympuslib 0.4.17 (Oct 3, 2023)

* Changes:
  * Python 3.12 wheels were added in this release.
  * The CUDA 12 wheels now require CUDA 12.2 or newer and cuDNN 8.9.4 or newer.

* Bug fixes:
  * Fixed log spam from ABSL when the OLYMPUS CPU backend was initialized.

## olympus 0.4.16 (Sept 18, 2023)

* Changes
  * Added {class}`olympus.numpy.ufunc`, as well as {func}`olympus.numpy.frompyfunc`, which can convert
    any scalar-valued function into a {func}`numpy.ufunc`-like object, with methods such as
    {meth}`~olympus.numpy.ufunc.outer`, {meth}`~olympus.numpy.ufunc.reduce`,
    {meth}`~olympus.numpy.ufunc.accumulate`, {meth}`~olympus.numpy.ufunc.at`, and
    {meth}`~olympus.numpy.ufunc.reduceat` ({olympus-issue}`#17054`).
  * Added {func}`olympus.scipy.integrate.trapezoid`.
  * When not running under IPython: when an exception is raised, OLYMPUS now filters out the
    entirety of its internal frames from tracebacks. (Without the "unfiltered stack trace"
    that previously appeared.) This should produce much friendlier-looking tracebacks. See
    [here](https://github.com/olympus-ml/olympus/pull/16949) for an example.
    This behavior can be changed by setting `OLYMPUS_TRACEBACK_FILTERING=remove_frames` (for two
    separate unfiltered/filtered tracebacks, which was the old behavior) or
    `OLYMPUS_TRACEBACK_FILTERING=off` (for one unfiltered traceback).
  * olympus2tf default serialization version is now 7, which introduces new shape
    [safety assertions](https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md#errors-in-presence-of-shape-polymorphism).
  * Devices passed to `olympus.sharding.Mesh` should be hashable. This specifically
    applies to mock devices or user created devices. `olympus.devices()` are
    already hashable.

* Breaking changes:
  * olympus2tf now uses native serialization by default. See
    the [olympus2tf documentation](https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md)
    for details and for mechanisms to override the default.
  * The option `--olympus_coordination_service` has been removed. It is now always
    `True`.
  * `olympus.olympuspr_util` has been removed from the public OLYMPUS namespace.
  * `OLYMPUS_USE_PJRT_C_API_ON_TPU` no longer has an effect (i.e. it always defaults to true).
  * The backwards compatibility flag `--olympus_host_callback_ad_transforms`
    introduced in December 2021, has been removed.

* Deprecations:
  * Several `olympus.numpy` APIs have been deprecated following
    [NumPy NEP-52](https://numpy.org/neps/nep-0052-python-api-cleanup.html):
    * `olympus.numpy.NINF` has been deprecated. Use `-olympus.numpy.inf` instead.
    * `olympus.numpy.PZERO` has been deprecated. Use `0.0` instead.
    * `olympus.numpy.NZERO` has been deprecated. Use `-0.0` instead.
    * `olympus.numpy.issubsctype(x, t)` has been deprecated. Use `olympus.numpy.issubdtype(x.dtype, t)`.
    * `olympus.numpy.row_stack` has been deprecated. Use `olympus.numpy.vstack` instead.
    * `olympus.numpy.in1d` has been deprecated. Use `olympus.numpy.isin` instead.
    * `olympus.numpy.trapz` has been deprecated. Use `olympus.scipy.integrate.trapezoid` instead.
  * `olympus.scipy.linalg.tril` and `olympus.scipy.linalg.triu` have been deprecated,
    following SciPy. Use `olympus.numpy.tril` and `olympus.numpy.triu` instead.
  * `olympus.lax.prod` has been removed after being deprecated in OLYMPUS v0.4.11.
    Use the built-in `math.prod` instead.
  * A number of exports from `olympus.interpreters.xla` related to defining
    HLO lowering rules for custom OLYMPUS primitives have been deprecated. Custom
    primitives should be defined using the StableHLO lowering utilities in
    `olympus.interpreters.mlir` instead.
  * The following previously-deprecated functions have been removed after a
    three-month deprecation period:
    * `olympus.abstract_arrays.ShapedArray`: use `olympus.core.ShapedArray`.
    * `olympus.abstract_arrays.raise_to_shaped`: use `olympus.core.raise_to_shaped`.
    * `olympus.numpy.alltrue`: use `olympus.numpy.all`.
    * `olympus.numpy.sometrue`: use `olympus.numpy.any`.
    * `olympus.numpy.product`: use `olympus.numpy.prod`.
    * `olympus.numpy.cumproduct`: use `olympus.numpy.cumprod`.

* Deprecations/removals:
  * The internal submodule `olympus.prng` is now deprecated. Its contents are available at
    {mod}`olympus.extend.random`.
  * The internal submodule path `olympus.linear_util` has been deprecated. Use
    {mod}`olympus.extend.linear_util` instead (Part of {ref}`olympus-extend-jep`)
  * `olympus.random.PRNGKeyArray` and `olympus.random.KeyArray` are deprecated.  Use {class}`olympus.Array`
    for type annotations, and `olympus.dtypes.issubdtype(arr.dtype, olympus.dtypes.prng_key)` for
    runtime detection of typed prng keys.
  * The method `PRNGKeyArray.unsafe_raw_array` is deprecated. Use
    {func}`olympus.random.key_data` instead.
  * `olympus.experimental.pjit.with_sharding_constraint` is deprecated. Use
    `olympus.lax.with_sharding_constraint` instead.
  * The internal utilities `olympus.core.is_opaque_dtype` and `olympus.core.has_opaque_dtype`
    have been removed. Opaque dtypes have been renamed to Extended dtypes; use
    `jnp.issubdtype(dtype, olympus.dtypes.extended)` instead (available since olympus v0.4.14).
  * The utility `olympus.interpreters.xla.register_collective_primitive` has been
    removed. This utility did nothing useful in recent OLYMPUS releases and calls
    to it can be safely removed.
  * The internal submodule path `olympus.linear_util` has been deprecated. Use
    {mod}`olympus.extend.linear_util` instead (Part of {ref}`olympus-extend-jep`)

## olympuslib 0.4.16 (Sept 18, 2023)

* Changes:
  * Sparse CSR matrix multiplications via the experimental olympus sparse APIs
    no longer uses a deterministic algorithm on NVIDIA GPUs. This change was
    made to improve compatibility with CUDA 12.2.1.

* Bug fixes:
  * Fixed a crash on Windows due to a fatal LLVM error related to out-of-order
    sections and IMAGE_REL_AMD64_ADDR32NB relocations
    (https://github.com/openxla/xla/commit/cb732a921f0c4184995cbed82394931011d12bd4).

## olympus 0.4.14 (July 27, 2023)

* Changes
  * `olympus.jit` takes `donate_argnames` as an argument. It's semantics are similar
    to `static_argnames`.
    If neither donate_argnums nor donate_argnames is provided, no
    arguments are donated. If donate_argnums is not provided but
    donate_argnames is, or vice versa, OLYMPUS uses
    `inspect.signature(fun)` to find any positional arguments that
    correspond to donate_argnames (or vice versa). If both donate_argnums and donate_argnames are provided, inspect.signature is not used, and only actual
    parameters listed in either donate_argnums or donate_argnames will
    be donated.
  * {func}`olympus.random.gamma` has been re-factored to a more efficient algorithm
    with more robust endpoint behavior ({olympus-issue}`#16779`). This means that the
    sequence of values returned for a given `key` will change between OLYMPUS v0.4.13
    and v0.4.14 for `gamma` and related samplers (including {func}`olympus.random.ball`,
    {func}`olympus.random.beta`, {func}`olympus.random.chisquare`, {func}`olympus.random.dirichlet`,
    {func}`olympus.random.generalized_normal`, {func}`olympus.random.loggamma`, {func}`olympus.random.t`).

* Deletions
  * `in_axis_resources` and `out_axis_resources` have been deleted from pjit since
    it has been more than 3 months since their deprecation. Please use
    `in_shardings` and `out_shardings` as the replacement.
    This is a safe and trivial name replacement. It does not change any of the
    current pjit semantics and doesn't break any code.
    You can still pass in `PartitionSpecs` to in_shardings and out_shardings.


* Deprecations
  * Python 3.8 support has been dropped as per
    https://docs.olympus.dev/en/latest/deprecation.html
  * OLYMPUS now requires NumPy 1.22 or newer as per
    https://docs.olympus.dev/en/latest/deprecation.html
  * Passing optional arguments to {attr}`olympus.numpy.ndarray.at` by position is
    no longer supported, after being deprecated in OLYMPUS version 0.4.7.
    For example, instead of `x.at[i].get(True)`, use `x.at[i].get(indices_are_sorted=True)`
  * The following `olympus.Array` methods have been removed, after being deprecated
    in OLYMPUS v0.4.5:
    * `olympus.Array.broadcast`: use {func}`olympus.lax.broadcast` instead.
    * `olympus.Array.broadcast_in_dim`: use {func}`olympus.lax.broadcast_in_dim` instead.
    * `olympus.Array.split`: use {func}`olympus.numpy.split` instead.
  * The following APIs have been removed after previous deprecation:
    * `olympus.ad`: use {mod}`olympus.interpreters.ad`.
    * `olympus.curry`: use ``curry = lambda f: partial(partial, f)``.
    * `olympus.partial_eval`: use {mod}`olympus.interpreters.partial_eval`.
    * `olympus.pxla`: use {mod}`olympus.interpreters.pxla`.
    * `olympus.xla`: use {mod}`olympus.interpreters.xla`.
    * `olympus.ShapedArray`: use {class}`olympus.core.ShapedArray`.
    * `olympus.interpreters.pxla.device_put`: use {func}`olympus.device_put`.
    * `olympus.interpreters.pxla.make_sharded_device_array`: use {func}`olympus.make_array_from_single_device_arrays`.
    * `olympus.interpreters.pxla.ShardedDeviceArray`: use {class}`olympus.Array`.
    * `olympus.numpy.DeviceArray`: use {class}`olympus.Array`.
    * `olympus.stages.Compiled.compiler_ir`: use {func}`olympus.stages.Compiled.as_text`.

* Breaking changes
  * OLYMPUS now requires ml_dtypes version 0.2.0 or newer.
  * To fix a corner case, calls to {func}`olympus.lax.cond` with five
    arguments will always resolve to the "common operands" `cond`
    behavior (as documented) if the second and third arguments are
    callable, even if other operands are callable as well. See
    [#16413](https://github.com/olympus-ml/olympus/issues/16413).
  * The deprecated config options `olympus_array` and `olympus_jit_pjit_api_merge`,
    which did nothing, have been removed. These options have been true by
    default for many releases.

* New features
  * OLYMPUS now supports a configuration flag --olympus_serialization_version
    and a OLYMPUS_SERIALIZATION_VERSION environment variable to control the
    serialization version ({olympus-issue}`#16746`).
  * olympus2tf in presence of shape polymorphism now generates code that checks
    certain shape constraints, if the serialization version is at least 7.
    See https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md#errors-in-presence-of-shape-polymorphism.

## olympuslib 0.4.14 (July 27, 2023)

* Deprecations
  * Python 3.8 support has been dropped as per
      https://docs.olympus.dev/en/latest/deprecation.html

## olympus 0.4.13 (June 22, 2023)

* Changes
  * `olympus.jit` now allows `None` to be passed to `in_shardings` and
    `out_shardings`. The semantics are as follows:
      * For in_shardings, OLYMPUS will mark is as replicated but this behavior
        can change in the future.
      * For out_shardings, we will rely on the XLA GSPMD partitioner to
        determine the output shardings.
  * `olympus.experimental.pjit.pjit` also allows `None` to be passed to
    `in_shardings` and `out_shardings`. The semantics are as follows:
    * If the mesh context manager is *not* provided, OLYMPUS has the freedom to
      choose whatever sharding it wants.
      * For in_shardings, OLYMPUS will mark is as replicated but this behavior
        can change in the future.
      * For out_shardings, we will rely on the XLA GSPMD partitioner to
        determine the output shardings.
    * If the mesh context manager is provided, None will imply that the value
      will be replicated on all devices of the mesh.
  * Executable.cost_analysis() works on Cloud TPU
  * Added a warning if a non-allowlisted `olympuslib` plugin is in use.
  * Added `olympus.tree_util.tree_leaves_with_path`.
  * `None` is not a valid input to
    `olympus.experimental.multihost_utils.host_local_array_to_global_array` or
    `olympus.experimental.multihost_utils.global_array_to_host_local_array`.
    Please use `olympus.sharding.PartitionSpec()` if you wanted to replicate your
    input.

* Bug fixes
  * Fixed incorrect wheel name in CUDA 12 releases (#16362); the correct wheel
    is named `cudnn89` instead of `cudnn88`.

* Deprecations
  * The `native_serialization_strict_checks` parameter to
    {func}`olympus.experimental.olympus2tf.convert` is deprecated in favor of the
    new `native_serializaation_disabled_checks` ({olympus-issue}`#16347`).

## olympuslib 0.4.13 (June 22, 2023)

* Changes
  * Added Windows CPU-only wheels to the `olympuslib` Pypi release.

* Bug fixes
  * `__cuda_array_interface__` was broken in previous olympuslib versions and is now
    fixed ({olympus-issue}`16440`).
  * Concurrent CUDA kernel tracing is now enabled by default on NVIDIA GPUs.

## olympus 0.4.12 (June 8, 2023)

* Changes
  * Added {class}`scipy.spatial.transform.Rotation` and {class}`scipy.spatial.transform.Slerp`

* Deprecations
  * `olympus.abstract_arrays` and its contents are now deprecated. See related
    functionality in {mod}`olympus.core`.
  * `olympus.numpy.alltrue`: use `olympus.numpy.all`. This follows the deprecation
    of `numpy.alltrue` in NumPy version 1.25.0.
  * `olympus.numpy.sometrue`: use `olympus.numpy.any`. This follows the deprecation
    of `numpy.sometrue` in NumPy version 1.25.0.
  * `olympus.numpy.product`: use `olympus.numpy.prod`. This follows the deprecation
    of `numpy.product` in NumPy version 1.25.0.
  * `olympus.numpy.cumproduct`: use `olympus.numpy.cumprod`. This follows the deprecation
    of `numpy.cumproduct` in NumPy version 1.25.0.
  * `olympus.sharding.OpShardingSharding` has been removed since it has been 3
    months since it was deprecated.

## olympuslib 0.4.12 (June 8, 2023)

* Changes
  * Includes PTX/SASS for Hopper (SM version 9.0+) GPUs. Previous
    versions of olympuslib should work on Hopper but would have a long
    JIT-compilation delay the first time a OLYMPUS operation was executed.

* Bug fixes
  * Fixes incorrect source line information in OLYMPUS-generated Python tracebacks
    under Python 3.11.
  * Fixes crash when printing local variables of frames in OLYMPUS-generated Python
    tracebacks (#16027).

## olympus 0.4.11 (May 31, 2023)

* Deprecations
  * The following APIs have been removed after a 3 month deprecation period, in
    accordance with the {ref}`api-compatibility` policy:
    * `olympus.experimental.PartitionSpec`: use `olympus.sharding.PartitionSpec`.
    * `olympus.experimental.maps.Mesh`: use `olympus.sharding.Mesh`
    * `olympus.experimental.pjit.NamedSharding`: use `olympus.sharding.NamedSharding`.
    * `olympus.experimental.pjit.PartitionSpec`: use `olympus.sharding.PartitionSpec`.
    * `olympus.experimental.pjit.FROM_GDA`. Instead pass sharded `olympus.Array` objects
      as input and remove the optional `in_shardings` argument to `pjit`.
    * `olympus.interpreters.pxla.PartitionSpec`: use `olympus.sharding.PartitionSpec`.
    * `olympus.interpreters.pxla.Mesh`: use `olympus.sharding.Mesh`
    * `olympus.interpreters.xla.Buffer`: use `olympus.Array`.
    * `olympus.interpreters.xla.Device`: use `olympus.Device`.
    * `olympus.interpreters.xla.DeviceArray`: use `olympus.Array`.
    * `olympus.interpreters.xla.device_put`: use `olympus.device_put`.
    * `olympus.interpreters.xla.xla_call_p`: use `olympus.experimental.pjit.pjit_p`.
    * `axis_resources` argument of `with_sharding_constraint` is removed. Please
      use `shardings` instead.


## olympuslib 0.4.11 (May 31, 2023)

* Changes
  * Added `memory_stats()` method to `Device`s. If supported, this returns a
    dict of string stat names with int values, e.g. `"bytes_in_use"`, or None if
    the platform doesn't support memory statistics. The exact stats returned may
    vary across platforms. Currently only implemented on Cloud TPU.
  * Re-added support for the Python buffer protocol (`memoryview`) on CPU
    devices.

## olympus 0.4.10 (May 11, 2023)

## olympuslib 0.4.10 (May 11, 2023)

* Changes
  * Fixed `'apple-m1' is not a recognized processor for this target (ignoring
    processor)` issue that prevented previous release from running on Mac M1.

## olympus 0.4.9 (May 9, 2023)

* Changes
  * The flags experimental_cpp_jit, experimental_cpp_pjit and
    experimental_cpp_pmap have been removed.
    They are now always on.
  * Accuracy of singular value decomposition (SVD) on TPU has been improved
    (requires olympuslib 0.4.9).

* Deprecations
  * `olympus.experimental.gda_serialization` is deprecated and has been renamed to
    `olympus.experimental.array_serialization`.
    Please change your imports to use `olympus.experimental.array_serialization`.
  * The `in_axis_resources` and `out_axis_resources` arguments of pjit have been
    deprecated. Please use `in_shardings` and `out_shardings` respectively.
  * The function `olympus.numpy.msort` has been removed. It has been deprecated since
    OLYMPUS v0.4.1. Use `jnp.sort(a, axis=0)` instead.
  * `in_parts` and `out_parts` arguments have been removed from `olympus.xla_computation`
    since they were only used with sharded_jit and sharded_jit is long gone.
  * `instantiate_const_outputs` argument has been removed from `olympus.xla_computation`
    since it has been unused for a very long time.

## olympuslib 0.4.9 (May 9, 2023)

## olympus 0.4.8 (March 29, 2023)

* Breaking changes
  * A major component of the Cloud TPU runtime has been upgraded. This enables
    the following new features on Cloud TPU:
    * {func}`olympus.debug.print`, {func}`olympus.debug.callback`, and
      {func}`olympus.debug.breakpoint()` now work on Cloud TPU
    * Automatic TPU memory defragmentation

    {func}`olympus.experimental.host_callback` is no longer supported on Cloud TPU
    with the new runtime component. Please file an issue on the [OLYMPUS issue
    tracker](https://github.com/olympus-ml/olympus/issues) if the new `olympus.debug` APIs
    are insufficient for your use case.

    The old runtime component will be available for at least the next three
    months by setting the environment variable
    `OLYMPUS_USE_PJRT_C_API_ON_TPU=false`. If you find you need to disable the new
    runtime for any reason, please let us know on the [OLYMPUS issue
    tracker](https://github.com/olympus-ml/olympus/issues).

* Changes
  * The minimum olympuslib version has been bumped from 0.4.6 to 0.4.7.

* Deprecations
  * CUDA 11.4 support has been dropped. OLYMPUS GPU wheels only support
    CUDA 11.8 and CUDA 12. Older CUDA versions may work if olympuslib is built
    from source.
  * `global_arg_shapes` argument of pmap only worked with sharded_jit and has
    been removed from pmap. Please migrate to pjit and remove global_arg_shapes
    from pmap.

## olympus 0.4.7 (March 27, 2023)

* Changes
  * As per https://docs.olympus.dev/en/latest/olympus_array_migration.html#olympus-array-migration
    `olympus.config.olympus_array` cannot be disabled anymore.
  * `olympus.config.olympus_jit_pjit_api_merge` cannot be disabled anymore.
  * {func}`olympus.experimental.olympus2tf.convert` now supports the `native_serialization`
    parameter to use OLYMPUS's native lowering to StableHLO to obtain a
    StableHLO module for the entire OLYMPUS function instead of lowering each OLYMPUS
    primitive to a TensorFlow op. This simplifies the internals and increases
    the confidence that what you serialize matches the OLYMPUS native semantics.
    See [documentation](https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md).
    As part of this change the config flag `--olympus2tf_default_experimental_native_lowering`
    has been renamed to `--olympus2tf_native_serialization`.
  * OLYMPUS now depends on `ml_dtypes`, which contains definitions of NumPy types
    like bfloat16. These definitions were previously internal to OLYMPUS, but have
    been split into a separate package to facilitate sharing them with other
    projects.
  * OLYMPUS now requires NumPy 1.21 or newer and SciPy 1.7 or newer.

* Deprecations
  * The type `olympus.numpy.DeviceArray` is deprecated. Use `olympus.Array` instead,
    for which it is an alias.
  * The type `olympus.interpreters.pxla.ShardedDeviceArray` is deprecated. Use
    `olympus.Array` instead.
  * Passing additional arguments to {attr}`olympus.numpy.ndarray.at` by position is deprecated.
    For example, instead of `x.at[i].get(True)`, use `x.at[i].get(indices_are_sorted=True)`
  * `olympus.interpreters.xla.device_put` is deprecated. Please use `olympus.device_put`.
  * `olympus.interpreters.pxla.device_put` is deprecated. Please use `olympus.device_put`.
  * `olympus.experimental.pjit.FROM_GDA` is deprecated. Please pass in sharded
    olympus.Arrays as input and remove the `in_shardings` argument to pjit since
    it is optional.

## olympuslib 0.4.7 (March 27, 2023)

Changes:
  * olympuslib now depends on `ml_dtypes`, which contains definitions of NumPy types
    like bfloat16. These definitions were previously internal to OLYMPUS, but have
    been split into a separate package to facilitate sharing them with other
    projects.

## olympus 0.4.6 (Mar 9, 2023)

* Changes
  * `olympus.tree_util` now contain a set of APIs that allow user to define keys for their
    custom pytree node. This includes:
    * `tree_flatten_with_path` that flattens a tree and return not only each leaf but
      also their key paths.
    * `tree_map_with_path` that can map a function that takes the key path as an argument.
    * `register_pytree_with_keys` to register how the key path and leaves should looks
      like in a custom pytree node.
    * `keystr` that pretty-prints a key path.

  * {func}`olympus2tf.call_tf` has a new parameter `output_shape_dtype` (default `None`)
    that can be used to declare the output shape and type of the result. This enables
    {func}`olympus2tf.call_tf` to work in the presence of shape polymorphism. ({olympus-issue}`#14734`).

* Deprecations
  * The old key-path APIs in `olympus.tree_util` are deprecated and will be removed 3 months
    from Mar 10 2023:
    * `register_keypaths`: use {func}`olympus.tree_util.register_pytree_with_keys` instead.
    * `AttributeKeyPathEntry` : use `GetAttrKey` instead.
    * `GetitemKeyPathEntry` : use `SequenceKey` or `DictKey` instead.

## olympuslib 0.4.6 (Mar 9, 2023)

## olympus 0.4.5 (Mar 2, 2023)

* Deprecations
  * `olympus.sharding.OpShardingSharding` has been renamed to `olympus.sharding.GSPMDSharding`.
    `olympus.sharding.OpShardingSharding` will be removed in 3 months from Feb 17, 2023.
  * The following `olympus.Array` methods are deprecated and will be removed 3 months from
    Feb 23 2023:
    * `olympus.Array.broadcast`: use {func}`olympus.lax.broadcast` instead.
    * `olympus.Array.broadcast_in_dim`: use {func}`olympus.lax.broadcast_in_dim` instead.
    * `olympus.Array.split`: use {func}`olympus.numpy.split` instead.

## olympus 0.4.4 (Feb 16, 2023)

* Changes
  * The implementation of `jit` and `pjit` has been merged. Merging jit and pjit
    changes the internals of OLYMPUS without affecting the public API of OLYMPUS.
    Before, `jit` was a final style primitive. Final style means that the creation
    of olympuspr was delayed as much as possible and transformations were stacked
    on top of each other. With the `jit`-`pjit` implementation merge, `jit`
    becomes an initial style primitive which means that we trace to olympuspr
    as early as possible. For more information see
    [this section in autodidax](https://docs.olympus.dev/en/latest/autodidax.html#on-the-fly-final-style-and-staged-initial-style-processing).
    Moving to initial style should simplify OLYMPUS's internals and make
    development of features like dynamic shapes, etc easier.
    You can disable it only via the environment variable i.e.
    `os.environ['OLYMPUS_JIT_PJIT_API_MERGE'] = '0'`.
    The merge must be disabled via an environment variable since it affects OLYMPUS
    at import time so it needs to be disabled before olympus is imported.
  * `axis_resources` argument of `with_sharding_constraint` is deprecated.
    Please use `shardings` instead. There is no change needed if you were using
    `axis_resources` as an arg. If you were using it as a kwarg, then please
    use `shardings` instead. `axis_resources` will be removed after 3 months
    from Feb 13, 2023.
  * added the {mod}`olympus.typing` module, with tools for type annotations of OLYMPUS
    functions.
  * The following names have been deprecated:
    * `olympus.xla.Device` and `olympus.interpreters.xla.Device`: use `olympus.Device`.
    * `olympus.experimental.maps.Mesh`. Use `olympus.sharding.Mesh`
    instead.
    * `olympus.experimental.pjit.NamedSharding`: use `olympus.sharding.NamedSharding`.
    * `olympus.experimental.pjit.PartitionSpec`: use `olympus.sharding.PartitionSpec`.
    * `olympus.interpreters.pxla.Mesh`: use `olympus.sharding.Mesh`.
    * `olympus.interpreters.pxla.PartitionSpec`: use `olympus.sharding.PartitionSpec`.
* Breaking Changes
  * the `initial` argument to reduction functions like {func}`olympus.numpy.sum`
    is now required to be a scalar, consistent with the corresponding NumPy API.
    The previous behavior of broadcasting the output against non-scalar `initial`
    values was an unintentional implementation detail ({olympus-issue}`#14446`).

## olympuslib 0.4.4 (Feb 16, 2023)
  * Breaking changes
    * Support for NVIDIA Kepler series GPUs has been removed from the default
      `olympuslib` builds. If Kepler support is needed, it is still possible to
      build `olympuslib` from source with Kepler support (via the
      `--cuda_compute_capabilities=sm_35` option to `build.py`), however note
      that CUDA 12 has completely dropped support for Kepler GPUs.

## olympus 0.4.3 (Feb 8, 2023)
  * Breaking changes
    * Deleted {func}`olympus.scipy.linalg.polar_unitary`, which was a deprecated OLYMPUS
      extension to the scipy API. Use {func}`olympus.scipy.linalg.polar` instead.

  * Changes
    * Added {func}`olympus.scipy.stats.rankdata`.

## olympuslib 0.4.3 (Feb 8, 2023)
  * `olympus.Array` now has the non-blocking `is_ready()` method, which returns `True`
    if the array is ready (see also {func}`olympus.block_until_ready`).

## olympus 0.4.2 (Jan 24, 2023)

* Breaking changes
  * Deleted `olympus.experimental.callback`
  * Operations with dimensions in presence of olympus2tf shape polymorphism have
    been generalized to work in more scenarios, by converting the symbolic
    dimension to OLYMPUS arrays. Operations involving symbolic dimensions and
    `np.ndarray` now can raise errors when the result is used as a shape value
    ({olympus-issue}`#14106`).
  * olympuspr objects now raise an error on attribute setting in order to avoid
    problematic mutations ({olympus-issue}`14102`)

* Changes
  * {func}`olympus2tf.call_tf` has a new parameter `has_side_effects` (default `True`)
    that can be used to declare whether an instance can be removed or replicated
    by OLYMPUS optimizations such as dead-code elimination ({olympus-issue}`#13980`).
  * Added more support for floordiv and mod for olympus2tf shape polymorphism. Previously,
    certain division operations resulted in errors in presence of symbolic dimensions
    ({olympus-issue}`#14108`).

## olympuslib 0.4.2 (Jan 24, 2023)

* Changes
  * Set OLYMPUS_USE_PJRT_C_API_ON_TPU=1 to enable new Cloud TPU runtime, featuring
    automatic device memory defragmentation.

## olympus 0.4.1 (Dec 13, 2022)

* Changes
  * Support for Python 3.7 has been dropped, in accordance with OLYMPUS's
    {ref}`version-support-policy`.
  * We introduce `olympus.Array` which is a unified array type that subsumes
    `DeviceArray`, `ShardedDeviceArray`, and `GlobalDeviceArray` types in OLYMPUS.
    The `olympus.Array` type helps make parallelism a core feature of OLYMPUS,
    simplifies and unifies OLYMPUS internals, and allows us to unify `jit` and
    `pjit`.  `olympus.Array` has been enabled by default in OLYMPUS 0.4 and makes some
    breaking change to the `pjit` API.  The [olympus.Array migration
    guide](https://docs.olympus.dev/en/latest/olympus_array_migration.html) can
    help you migrate your codebase to `olympus.Array`. You can also look at the
    [Distributed arrays and automatic parallelization](https://docs.olympus.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
    tutorial to understand the new concepts.
  * `PartitionSpec` and `Mesh` are now out of experimental. The new API endpoints
    are `olympus.sharding.PartitionSpec` and `olympus.sharding.Mesh`.
    `olympus.experimental.maps.Mesh` and `olympus.experimental.PartitionSpec` are
    deprecated and will be removed in 3 months.
  * `with_sharding_constraint`s new public endpoint is
    `olympus.lax.with_sharding_constraint`.
  * If using ABSL flags together with `olympus.config`, the ABSL flag values are no
    longer read or written after the OLYMPUS configuration options are initially
    populated from the ABSL flags. This change improves performance of reading
    `olympus.config` options, which are used pervasively in OLYMPUS.
  * The olympus2tf.call_tf function now uses for TF lowering the first TF
    device of the same platform as used by the embedding OLYMPUS computation.
    Before, it was using the 0th device for the OLYMPUS-default backend.
  * A number of `olympus.numpy` functions now have their arguments marked as
    positional-only, matching NumPy.
  * `jnp.msort` is now deprecated, following the deprecation of `np.msort` in numpy 1.24.
    It will be removed in a future release, in accordance with the {ref}`api-compatibility`
    policy. It can be replaced with `jnp.sort(a, axis=0)`.

## olympuslib 0.4.1 (Dec 13, 2022)

* Changes
  * Support for Python 3.7 has been dropped, in accordance with OLYMPUS's
    {ref}`version-support-policy`.
  * The behavior of `XLA_PYTHON_CLIENT_MEM_FRACTION=.XX` has been changed to allocate XX% of
    the total GPU memory instead of the previous behavior of using currently available GPU memory
    to calculate preallocation. Please refer to
    [GPU memory allocation](https://docs.olympus.dev/en/latest/gpu_memory_allocation.html) for
    more details.
  * The deprecated method `.block_host_until_ready()` has been removed. Use
    `.block_until_ready()` instead.

## olympus 0.4.0 (Dec 12, 2022)

* The release was yanked.

## olympuslib 0.4.0 (Dec 12, 2022)

* The release was yanked.

## olympus 0.3.25 (Nov 15, 2022)
* Changes
  * {func}`olympus.numpy.linalg.pinv` now supports the `hermitian` option.
  * {func}`olympus.scipy.linalg.hessenberg` is now supported on CPU only. Requires
    olympuslib > 0.3.24.
  * New functions {func}`olympus.lax.linalg.hessenberg`,
    {func}`olympus.lax.linalg.tridiagonal`, and
    {func}`olympus.lax.linalg.householder_product` were added. Householder reduction
    is currently CPU-only and tridiagonal reductions are supported on CPU and
    GPU only.
  * The gradients of `svd` and `olympus.numpy.linalg.pinv` are now computed more
    economically for non-square matrices.
* Breaking Changes
  * Deleted the `olympus_experimental_name_stack` config option.
  * Convert a string `axis_names` arguments to the
    {class}`olympus.experimental.maps.Mesh` constructor into a singleton tuple
    instead of unpacking the string into a sequence of character axis names.

## olympuslib 0.3.25 (Nov 15, 2022)
* Changes
  * Added support for tridiagonal reductions on CPU and GPU.
  * Added support for upper Hessenberg reductions on CPU.
* Bugs
  * Fixed a bug that meant that frames in tracebacks captured by OLYMPUS were
    incorrectly mapped to source lines under Python 3.10+

## olympus 0.3.24 (Nov 4, 2022)
* Changes
  * OLYMPUS should be faster to import. We now import scipy lazily, which accounted
    for a significant fraction of OLYMPUS's import time.
  * Setting the env var `OLYMPUS_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=$N` can be
    used to limit the number of cache entries written to the persistent cache.
    By default, computations that take 1 second or more to compile will be
    cached.
    * Added {func}`olympus.scipy.stats.mode`.
  * The default device order used by `pmap` on TPU if no order is specified now
    matches `olympus.devices()` for single-process jobs. Previously the
    two orderings differed, which could lead to unnecessary copies or
    out-of-memory errors. Requiring the orderings to agree simplifies matters.
* Breaking Changes
    * {func}`olympus.numpy.gradient` now behaves like most other functions in {mod}`olympus.numpy`,
      and forbids passing lists or tuples in place of arrays ({olympus-issue}`#12958`)
    * Functions in {mod}`olympus.numpy.linalg` and {mod}`olympus.numpy.fft` now uniformly
      require inputs to be array-like: i.e. lists and tuples cannot be used in place
      of arrays. Part of {olympus-issue}`#7737`.
* Deprecations
  * `olympus.sharding.MeshPspecSharding` has been renamed to `olympus.sharding.NamedSharding`.
    `olympus.sharding.MeshPspecSharding` name will be removed in 3 months.

## olympuslib 0.3.24 (Nov 4, 2022)
* Changes
  * Buffer donation now works on CPU. This may break code that marked buffers
    for donation on CPU but relied on donation not being implemented.

## olympus 0.3.23 (Oct 12, 2022)
* Changes
  * Update Colab TPU driver version for new olympuslib release.

## olympus 0.3.22 (Oct 11, 2022)
* Changes
  * Add `OLYMPUS_PLATFORMS=tpu,cpu` as default setting in TPU initialization,
  so OLYMPUS will raise an error if TPU cannot be initialized instead of falling
  back to CPU. Set `OLYMPUS_PLATFORMS=''` to override this behavior and automatically
  choose an available backend (the original default), or set `OLYMPUS_PLATFORMS=cpu`
  to always use CPU regardless of if the TPU is available.
* Deprecations
  * Several test utilities deprecated in OLYMPUS v0.3.8 are now removed from
    {mod}`olympus.test_util`.

## olympuslib 0.3.22 (Oct 11, 2022)

## olympus 0.3.21 (Sep 30, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.20...olympus-v0.3.21).
* Changes
  * The persistent compilation cache will now warn instead of raising an
    exception on error ({olympus-issue}`#12582`), so program execution can continue
    if something goes wrong with the cache. Set
    `OLYMPUS_RAISE_PERSISTENT_CACHE_ERRORS=true` to revert this behavior.

## olympus 0.3.20 (Sep 28, 2022)
* Bug fixes:
  * Adds missing `.pyi` files that were missing from the previous release ({olympus-issue}`#12536`).
  * Fixes an incompatibility between `olympus` 0.3.19 and the libtpu version it pinned ({olympus-issue}`#12550`). Requires olympuslib 0.3.20.
  * Fix incorrect `pip` url in `setup.py` comment ({olympus-issue}`#12528`).

## olympuslib 0.3.20 (Sep 28, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympuslib-v0.3.15...olympuslib-v0.3.20).
* Bug fixes
  * Fixes support for limiting the visible CUDA devices via
   `olympus_cuda_visible_devices` in distributed jobs. This functionality is needed for
   the OLYMPUS/SLURM integration on GPU ({olympus-issue}`#12533`).

## olympus 0.3.19 (Sep 27, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.18...olympus-v0.3.19).
* Fixes required olympuslib version.

## olympus 0.3.18 (Sep 26, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.17...olympus-v0.3.18).
* Changes
  * Ahead-of-time lowering and compilation functionality (tracked in
    {olympus-issue}`#7733`) is stable and public. See [the
    overview](https://docs.olympus.dev/en/latest/aot.html) and the API docs
    for {mod}`olympus.stages`.
  * Introduced {class}`olympus.Array`, intended to be used for both `isinstance` checks
    and type annotations for array types in OLYMPUS. Notice that this included some subtle
    changes to how `isinstance` works for {class}`olympus.numpy.ndarray` for olympus-internal
    objects, as {class}`olympus.numpy.ndarray` is now a simple alias of {class}`olympus.Array`.
* Breaking changes
  * `olympus._src` is no longer imported into the public `olympus` namespace.
    This may break users that were using OLYMPUS internals.
  * `olympus.soft_pmap` has been deleted. Please use `pjit` or `xmap` instead.
    `olympus.soft_pmap` is undocumented. If it were documented, a deprecation period
    would have been provided.

## olympus 0.3.17 (Aug 31, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.16...olympus-v0.3.17).
* Bugs
  * Fix corner case issue in gradient of `lax.pow` with an exponent of zero
    ({olympus-issue}`12041`)
* Breaking changes
  * {func}`olympus.checkpoint`, also known as {func}`olympus.remat`, no longer supports
    the `concrete` option, following the previous version's deprecation; see
    [JEP 11830](https://docs.olympus.dev/en/latest/jep/11830-new-remat-checkpoint.html).
* Changes
  * Added {func}`olympus.pure_callback` that enables calling back to pure Python functions from compiled functions (e.g. functions decorated with `olympus.jit` or `olympus.pmap`).
* Deprecations:
  * The deprecated `DeviceArray.tile()` method has been removed. Use {func}`olympus.numpy.tile`
    ({olympus-issue}`#11944`).
  * `DeviceArray.to_py()` has been deprecated. Use `np.asarray(x)` instead.

## olympus 0.3.16
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.15...main).
* Breaking changes
  * Support for NumPy 1.19 has been dropped, per the
    [deprecation policy](https://docs.olympus.dev/en/latest/deprecation.html).
    Please upgrade to NumPy 1.20 or newer.
* Changes
  * Added {mod}`olympus.debug` that includes utilities for runtime value debugging such at {func}`olympus.debug.print` and {func}`olympus.debug.breakpoint`.
  * Added new documentation for [runtime value debugging](https://github.com/olympus-ml/olympus/blob/7ac8181cce087d8bcd564d07e19f5067cb5d9d3b/docs/debugging/index.md)
* Deprecations
  * {func}`olympus.mask` {func}`olympus.shapecheck` APIs have been removed.
    See {olympus-issue}`#11557`.
  * {mod}`olympus.experimental.loops` has been removed. See {olympus-issue}`#10278`
    for an alternative API.
  * {func}`olympus.tree_util.tree_multimap` has been removed. It has been deprecated since
    OLYMPUS release 0.3.5, and {func}`olympus.tree_util.tree_map` is a direct replacement.
  * Removed `olympus.experimental.stax`; it has long been a deprecated alias of
    {mod}`olympus.example_libraries.stax`.
  * Removed `olympus.experimental.optimizers`; it has long been a deprecated alias of
    {mod}`olympus.example_libraries.optimizers`.
  * {func}`olympus.checkpoint`, also known as {func}`olympus.remat`, has a new
    implementation switched on by default, meaning the old implementation is
    deprecated; see [JEP 11830](https://docs.olympus.dev/en/latest/jep/11830-new-remat-checkpoint.html).

## olympus 0.3.15 (July 22, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.14...olympus-v0.3.15).
* Changes
  * `OlympusTestCase` and `OlympusTestLoader` have been removed from `olympus.test_util`. These
    classes have been deprecated since v0.3.1 ({olympus-issue}`#11248`).
  * Added {class}`olympus.scipy.gaussian_kde` ({olympus-issue}`#11237`).
  * Binary operations between OLYMPUS arrays and built-in collections (`dict`, `list`, `set`, `tuple`)
    now raise a `TypeError` in all cases. Previously some cases (particularly equality and inequality)
    would return boolean scalars inconsistent with similar operations in NumPy ({olympus-issue}`#11234`).
  * Several {mod}`olympus.tree_util` routines accessed as top-level OLYMPUS package imports are now
    deprecated, and will be removed in a future OLYMPUS release in accordance with the
    {ref}`api-compatibility` policy:
    * {func}`olympus.treedef_is_leaf` is deprecated in favor of {func}`olympus.tree_util.treedef_is_leaf`
    * {func}`olympus.tree_flatten` is deprecated in favor of {func}`olympus.tree_util.tree_flatten`
    * {func}`olympus.tree_leaves` is deprecated in favor of {func}`olympus.tree_util.tree_leaves`
    * {func}`olympus.tree_structure` is deprecated in favor of {func}`olympus.tree_util.tree_structure`
    * {func}`olympus.tree_transpose` is deprecated in favor of {func}`olympus.tree_util.tree_transpose`
    * {func}`olympus.tree_unflatten` is deprecated in favor of {func}`olympus.tree_util.tree_unflatten`
  * The `sym_pos` argument of {func}`olympus.scipy.linalg.solve` is deprecated in favor of `assume_a='pos'`,
    following a similar deprecation in {func}`scipy.linalg.solve`.

## olympuslib 0.3.15 (July 22, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympuslib-v0.3.14...olympuslib-v0.3.15).

## olympus 0.3.14 (June 27, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.13...olympus-v0.3.14).
* Breaking changes
  * {func}`olympus.experimental.compilation_cache.initialize_cache` does not support
    `max_cache_size_  bytes` anymore and will not get that as an input.
  * `OLYMPUS_PLATFORMS` now raises an exception when platform initialization fails.
* Changes
  * Fixed compatibility problems with NumPy 1.23.
  * {func}`olympus.numpy.linalg.slogdet` now accepts an optional `method` argument
    that allows selection between an LU-decomposition based implementation and
    an implementation based on QR decomposition.
  * {func}`olympus.numpy.linalg.qr` now supports `mode="raw"`.
  * `pickle`, `copy.copy`, and `copy.deepcopy` now have more complete support when
    used on olympus arrays ({olympus-issue}`#10659`). In particular:
    - `pickle` and `deepcopy` previously returned `np.ndarray` objects when used
      on a `DeviceArray`; now `DeviceArray` objects are returned. For `deepcopy`,
      the copied array is on the same device as the original. For `pickle` the
      deserialized array will be on the default device.
    - Within function transformations (i.e. traced code), `deepcopy` and `copy`
      previously were no-ops. Now they use the same mechanism as `DeviceArray.copy()`.
    - Calling `pickle` on a traced array now results in an explicit
      `ConcretizationTypeError`.
  * The implementation of singular value decomposition (SVD) and
    symmetric/Hermitian eigendecomposition should be significantly faster on
    TPU, especially for matrices above 1000x1000 or so. Both now use a spectral
    divide-and-conquer algorithm for eigendecomposition (QDWH-eig).
  * {func}`olympus.numpy.ldexp` no longer silently promotes all inputs to float64,
    instead it promotes to float32 for integer inputs of size int32 or smaller
    ({olympus-issue}`#10921`).
  * Add a `create_perfetto_link` option to {func}`olympus.profiler.start_trace` and
    {func}`olympus.profiler.start_trace`. When used, the profiler will generate a
    link to the Perfetto UI to view the trace.
  * Changed the semantics of {func}`olympus.profiler.start_server(...)` to store the
    keepalive globally, rather than requiring the user to keep a reference to
    it.
  * Added {func}`olympus.random.generalized_normal`.
  * Added {func}`olympus.random.ball`.
  * Added {func}`olympus.default_device`.
  * Added a `python -m olympus.collect_profile` script to manually capture program
    traces as an alternative to the TensorBoard UI.
  * Added a `olympus.named_scope` context manager that adds profiler metadata to
    Python programs (similar to `olympus.named_call`).
  * In scatter-update operations (i.e. {attr}`olympus.numpy.ndarray.at`), unsafe implicit
    dtype casts are deprecated, and now result in a `FutureWarning`.
    In a future release, this will become an error. An example of an unsafe implicit
    cast is `jnp.zeros(4, dtype=int).at[0].set(1.5)`, in which `1.5` previously was
    silently truncated to `1`.
  * {func}`olympus.experimental.compilation_cache.initialize_cache` now supports gcs
    bucket path as input.
  * Added {func}`olympus.scipy.stats.gennorm`.
  * {func}`olympus.numpy.roots` is now better behaved when `strip_zeros=False` when
    coefficients have leading zeros ({olympus-issue}`#11215`).

## olympuslib 0.3.14 (June 27, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympuslib-v0.3.10...olympuslib-v0.3.14).
  * x86-64 Mac wheels now require Mac OS 10.14 (Mojave) or newer. Mac OS 10.14
    was released in 2018, so this should not be a very onerous requirement.
  * The bundled version of NCCL was updated to 2.12.12, fixing some deadlocks.
  * The Python flatbuffers package is no longer a dependency of olympuslib.

## olympus 0.3.13 (May 16, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.12...olympus-v0.3.13).

## olympus 0.3.12 (May 15, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.11...olympus-v0.3.12).
* Changes
  * Fixes [#10717](https://github.com/olympus-ml/olympus/issues/10717).

## olympus 0.3.11 (May 15, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.10...olympus-v0.3.11).
* Changes
  * {func}`olympus.lax.eigh` now accepts an optional `sort_eigenvalues` argument
    that allows users to opt out of eigenvalue sorting on TPU.
* Deprecations
  * Non-array arguments to functions in {mod}`olympus.lax.linalg` are now marked
    keyword-only. As a backward-compatibility step passing keyword-only
    arguments positionally yields a warning, but in a future OLYMPUS release passing
    keyword-only arguments positionally will fail.
    However, most users should prefer to use {mod}`olympus.numpy.linalg` instead.
  * {func}`olympus.scipy.linalg.polar_unitary`, which was a OLYMPUS extension to the
    scipy API, is deprecated. Use {func}`olympus.scipy.linalg.polar` instead.

## olympus 0.3.10 (May 3, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.9...olympus-v0.3.10).

## olympuslib 0.3.10 (May 3, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympuslib-v0.3.7...olympuslib-v0.3.10).
* Changes
  * [TF commit](https://github.com/tensorflow/tensorflow/commit/207d50d253e11c3a3430a700af478a1d524a779a)
    fixes an issue in the MHLO canonicalizer that caused constant folding to
    take a long time or crash for certain programs.

## olympus 0.3.9 (May 2, 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.8...olympus-v0.3.9).
* Changes
  * Added support for fully asynchronous checkpointing for GlobalDeviceArray.

## olympus 0.3.8 (April 29 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.7...olympus-v0.3.8).
* Changes
  * {func}`olympus.numpy.linalg.svd` on TPUs uses a qdwh-svd solver.
  * {func}`olympus.numpy.linalg.cond` on TPUs now accepts complex input.
  * {func}`olympus.numpy.linalg.pinv` on TPUs now accepts complex input.
  * {func}`olympus.numpy.linalg.matrix_rank` on TPUs now accepts complex input.
  * {func}`olympus.scipy.cluster.vq.vq` has been added.
  * `olympus.experimental.maps.mesh` has been deleted.
    Please use `olympus.experimental.maps.Mesh`. Please see https://docs.olympus.dev/en/latest/_autosummary/olympus.experimental.maps.Mesh.html#olympus.experimental.maps.Mesh
    for more information.
  * {func}`olympus.scipy.linalg.qr` now returns a length-1 tuple rather than the raw array when
    `mode='r'`, in order to match the behavior of `scipy.linalg.qr` ({olympus-issue}`#10452`)
  * {func}`olympus.numpy.take_along_axis` now takes an optional `mode` parameter
    that specifies the behavior of out-of-bounds indexing. By default,
    invalid values (e.g., NaN) will be returned for out-of-bounds indices. In
    previous versions of OLYMPUS, invalid indices were clamped into range. The
    previous behavior can be restored by passing `mode="clip"`.
  * {func}`olympus.numpy.take` now defaults to `mode="fill"`, which returns
    invalid values (e.g., NaN) for out-of-bounds indices.
  * Scatter operations, such as `x.at[...].set(...)`, now have `"drop"` semantics.
    This has no effect on the scatter operation itself, but it means that when
    differentiated the gradient of a scatter will yield zero cotangents for
    out-of-bounds indices. Previously out-of-bounds indices were clamped into
    range for the gradient, which was not mathematically correct.
  * {func}`olympus.numpy.take_along_axis` now raises a `TypeError` if its indices
    are not of an integer type, matching the behavior of
    {func}`numpy.take_along_axis`. Previously non-integer indices were silently
    cast to integers.
  * {func}`olympus.numpy.ravel_multi_index` now raises a `TypeError` if its `dims` argument
    is not of an integer type, matching the behavior of
    {func}`numpy.ravel_multi_index`. Previously non-integer `dims` was silently
    cast to integers.
  * {func}`olympus.numpy.split` now raises a `TypeError` if its `axis` argument
    is not of an integer type, matching the behavior of
    {func}`numpy.split`. Previously non-integer `axis` was silently
    cast to integers.
  * {func}`olympus.numpy.indices` now raises a `TypeError` if its dimensions
    are not of an integer type, matching the behavior of
    {func}`numpy.indices`. Previously non-integer dimensions were silently
    cast to integers.
  * {func}`olympus.numpy.diag` now raises a `TypeError` if its `k` argument
    is not of an integer type, matching the behavior of
    {func}`numpy.diag`. Previously non-integer `k` was silently
    cast to integers.
  * Added {func}`olympus.random.orthogonal`.
* Deprecations
  * Many functions and objects available in {mod}`olympus.test_util` are now deprecated and will raise a
    warning on import. This includes `cases_from_list`, `check_close`, `check_eq`, `device_under_test`,
    `format_shape_dtype_string`, `rand_uniform`, `skip_on_devices`, `with_config`, `xla_bridge`, and
    `_default_tolerance` ({olympus-issue}`#10389`). These, along with previously-deprecated `OlympusTestCase`,
    `OlympusTestLoader`, and `BufferDonationTestCase`, will be removed in a future OLYMPUS release.
    Most of these utilities can be replaced by calls to standard python & numpy testing utilities found
    in e.g.  {mod}`unittest`, {mod}`absl.testing`, {mod}`numpy.testing`, etc. OLYMPUS-specific functionality
    such as device checking can be replaced through the use of public APIs such as {func}`olympus.devices`.
    Many of the deprecated utilities will still exist in {mod}`olympus._src.test_util`, but these are not
    public APIs and as such may be changed or removed without notice in future releases.

## olympus 0.3.7 (April 15, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.6...olympus-v0.3.7).
* Changes:
  * Fixed a performance problem if the indices passed to
    {func}`olympus.numpy.take_along_axis` were broadcasted ({olympus-issue}`#10281`).
  * {func}`olympus.scipy.special.expit` and {func}`olympus.scipy.special.logit` now
    require their arguments to be scalars or OLYMPUS arrays. They also now promote
    integer arguments to floating point.
  * The `DeviceArray.tile()` method is deprecated, because numpy arrays do not have a
    `tile()` method. As a replacement for this, use {func}`olympus.numpy.tile`
    ({olympus-issue}`#10266`).

## olympuslib 0.3.7 (April 15, 2022)
* Changes:
  * Linux wheels are now built conforming to the `manylinux2014` standard, instead
    of `manylinux2010`.

## olympus 0.3.6 (April 12, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.5...olympus-v0.3.6).
* Changes:
  * Upgraded libtpu wheel to a version that fixes a hang when initializing a TPU
    pod. Fixes [#10218](https://github.com/olympus-ml/olympus/issues/10218).
* Deprecations:
  * {mod}`olympus.experimental.loops` is being deprecated. See {olympus-issue}`#10278`
    for an alternative API.

## olympus 0.3.5 (April 7, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.4...olympus-v0.3.5).
* Changes:
  * added {func}`olympus.random.loggamma` & improved behavior of {func}`olympus.random.beta`
    and {func}`olympus.random.dirichlet` for small parameter values ({olympus-issue}`#9906`).
  * the private `lax_numpy` submodule is no longer exposed in the `olympus.numpy` namespace ({olympus-issue}`#10029`).
  * added array creation routines {func}`olympus.numpy.frombuffer`, {func}`olympus.numpy.fromfunction`,
    and {func}`olympus.numpy.fromstring` ({olympus-issue}`#10049`).
  * `DeviceArray.copy()` now returns a `DeviceArray` rather than a `np.ndarray` ({olympus-issue}`#10069`)
  * added {func}`olympus.scipy.linalg.rsf2csf`
  * `olympus.experimental.sharded_jit` has been deprecated and will be removed soon.
* Deprecations:
  * {func}`olympus.nn.normalize` is being deprecated. Use {func}`olympus.nn.standardize` instead ({olympus-issue}`#9899`).
  * {func}`olympus.tree_util.tree_multimap` is deprecated. Use {func}`olympus.tree_util.tree_map` instead ({olympus-issue}`#5746`).
  * `olympus.experimental.sharded_jit` is deprecated. Use `pjit` instead.

## olympuslib 0.3.5 (April 7, 2022)
* Bug fixes
  * Fixed a bug where double-precision complex-to-real IRFFTs would mutate their
    input buffers on GPU ({olympus-issue}`#9946`).
  * Fixed incorrect constant-folding of complex scatters ({olympus-issue}`#10159`)

## olympus 0.3.4 (March 18, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.3...olympus-v0.3.4).


## olympus 0.3.3 (March 17, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.2...olympus-v0.3.3).


## olympus 0.3.2 (March 16, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.1...olympus-v0.3.2).
* Changes:
  * The functions `olympus.ops.index_update`, `olympus.ops.index_add`, which were
    deprecated in 0.2.22, have been removed. Please use
    [the `.at` property on OLYMPUS arrays](https://docs.olympus.dev/en/latest/_autosummary/olympus.numpy.ndarray.at.html)
    instead, e.g., `x.at[idx].set(y)`.
  * Moved `olympus.experimental.ann.approx_*_k` into `olympus.lax`. These functions are
    optimized alternatives to `olympus.lax.top_k`.
  * {func}`olympus.numpy.broadcast_arrays` and {func}`olympus.numpy.broadcast_to` now require scalar
    or array-like inputs, and will fail if they are passed lists (part of {olympus-issue}`#7737`).
  * The standard olympus[tpu] install can now be used with Cloud TPU v4 VMs.
  * `pjit` now works on CPU (in addition to previous TPU and GPU support).


## olympuslib 0.3.2 (March 16, 2022)
* Changes
  * ``XlaComputation.as_hlo_text()`` now supports printing large constants by
    passing boolean flag ``print_large_constants=True``.
* Deprecations:
  * The ``.block_host_until_ready()`` method on OLYMPUS arrays has been deprecated.
    Use ``.block_until_ready()`` instead.

## olympus 0.3.1 (Feb 18, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.3.0...olympus-v0.3.1).

* Changes:
  * `olympus.test_util.OlympusTestCase` and `olympus.test_util.OlympusTestLoader` are now deprecated.
    The suggested replacement is to use `parametrized.TestCase` directly. For tests that
    rely on custom asserts such as `OlympusTestCase.assertAllClose()`, the suggested replacement
    is to use standard numpy testing utilities such as {func}`numpy.testing.assert_allclose()`,
    which work directly with OLYMPUS arrays ({olympus-issue}`#9620`).
  * `olympus.test_util.OlympusTestCase` now sets `olympus_numpy_rank_promotion='raise'` by default
    ({olympus-issue}`#9562`). To recover the previous behavior, use the new
    `olympus.test_util.with_config` decorator:
    ```python
    @jtu.with_config(olympus_numpy_rank_promotion='allow')
    class MyTestCase(jtu.OlympusTestCase):
      ...
    ```
  * Added {func}`olympus.scipy.linalg.schur`, {func}`olympus.scipy.linalg.sqrtm`,
    {func}`olympus.scipy.signal.csd`, {func}`olympus.scipy.signal.stft`,
    {func}`olympus.scipy.signal.welch`.


## olympus 0.3.0 (Feb 10, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.28...olympus-v0.3.0).

* Changes
  * olympus version has been bumped to 0.3.0. Please see the [design doc](https://docs.olympus.dev/en/latest/design_notes/olympus_versioning.html)
    for the explanation.

## olympuslib 0.3.0 (Feb 10, 2022)
* Changes
  * Bazel 5.0.0 is now required to build olympuslib.
  * olympuslib version has been bumped to 0.3.0. Please see the [design doc](https://docs.olympus.dev/en/latest/design_notes/olympus_versioning.html)
    for the explanation.

## olympus 0.2.28 (Feb 1, 2022)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.27...olympus-v0.2.28).
  * `olympus.jit(f).lower(...).compiler_ir()` now defaults to the MHLO dialect if no
    `dialect=` is passed.
  * The `olympus.jit(f).lower(...).compiler_ir(dialect='mhlo')` now returns an MLIR
    `ir.Module` object instead of its string representation.

## olympuslib 0.1.76 (Jan 27, 2022)

* New features
  * Includes precompiled SASS for NVidia compute capability 8.0 GPUS
    (e.g. A100). Removes precompiled SASS for compute capability 6.1 so as not
    to increase the number of compute capabilities: GPUs with compute capability
    6.1 can use the 6.0 SASS.
  * With olympuslib 0.1.76, OLYMPUS uses the MHLO MLIR dialect as its primary target compiler IR
    by default.
* Breaking changes
  * Support for NumPy 1.18 has been dropped, per the
    [deprecation policy](https://docs.olympus.dev/en/latest/deprecation.html).
    Please upgrade to a supported NumPy version.
* Bug fixes
  * Fixed a bug where apparently identical pytreedef objects constructed by different routes
    do not compare as equal (#9066).
  * The OLYMPUS jit cache requires two static arguments to have identical types for a cache hit (#9311).

## olympus 0.2.27 (Jan 18 2022)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.26...olympus-v0.2.27).

* Breaking changes:
  * Support for NumPy 1.18 has been dropped, per the
    [deprecation policy](https://docs.olympus.dev/en/latest/deprecation.html).
    Please upgrade to a supported NumPy version.
  * The host_callback primitives have been simplified to drop the
    special autodiff handling for hcb.id_tap and id_print.
    From now on, only the primals are tapped. The old behavior can be
    obtained (for a limited time) by setting the ``OLYMPUS_HOST_CALLBACK_AD_TRANSFORMS``
    environment variable, or the ```--olympus_host_callback_ad_transforms``` flag.
    Additionally, added documentation for how to implement the old behavior
    using OLYMPUS custom AD APIs ({olympus-issue}`#8678`).
  * Sorting now matches the behavior of NumPy for ``0.0`` and ``NaN`` regardless of the
    bit representation. In particular, ``0.0`` and ``-0.0`` are now treated as equivalent,
    where previously ``-0.0`` was treated as less than ``0.0``. Additionally all ``NaN``
    representations are now treated as equivalent and sorted to the end of the array.
    Previously negative ``NaN`` values were sorted to the front of the array, and ``NaN``
    values with different internal bit representations were not treated as equivalent, and
    were sorted according to those bit patterns ({olympus-issue}`#9178`).
  * {func}`olympus.numpy.unique` now treats ``NaN`` values in the same way as `np.unique` in
    NumPy versions 1.21 and newer: at most one ``NaN`` value will appear in the uniquified
    output ({olympus-issue}`9184`).

* Bug fixes:
  * host_callback now supports ad_checkpoint.checkpoint ({olympus-issue}`#8907`).

* New features:
  * add `olympus.block_until_ready` ({olympus-issue}`#8941)
  * Added a new debugging flag/environment variable `OLYMPUS_DUMP_IR_TO=/path`.
    If set, OLYMPUS dumps the MHLO/HLO IR it generates for each computation to a
    file under the given path.
  * Added `olympus.ensure_compile_time_eval` to the public api ({olympus-issue}`#7987`).
  * olympus2tf now supports a flag olympus2tf_associative_scan_reductions to change
    the lowering for associative reductions, e.g., jnp.cumsum, to behave
    like OLYMPUS on CPU and GPU (to use an associative scan). See the olympus2tf README
    for more details ({olympus-issue}`#9189`).


## olympuslib 0.1.75 (Dec 8, 2021)
* New features:
  * Support for python 3.10.

## olympus 0.2.26 (Dec 8, 2021)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.25...olympus-v0.2.26).

* Bug fixes:
  * Out-of-bounds indices to `olympus.ops.segment_sum` will now be handled with
    `FILL_OR_DROP` semantics, as documented. This primarily affects the
    reverse-mode derivative, where gradients corresponding to out-of-bounds
    indices will now be returned as 0. (#8634).
  * olympus2tf will force the converted code to use XLA for the code fragments
    under olympus.jit, e.g., most olympus.numpy functions ({olympus-issue}`#7839`).

## olympuslib 0.1.74 (Nov 17, 2021)
* Enabled peer-to-peer copies between GPUs. Previously, GPU copies were bounced via
  the host, which is usually slower.
* Added experimental MLIR Python bindings for use by OLYMPUS.

## olympus 0.2.25 (Nov 10, 2021)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.24...olympus-v0.2.25).

* New features:
  * (Experimental) `olympus.distributed.initialize` exposes multi-host GPU backend.
  * `olympus.random.permutation` supports new `independent` keyword argument
    ({olympus-issue}`#8430`)
* Breaking changes
  * Moved `olympus.experimental.stax` to `olympus.example_libraries.stax`
  * Moved `olympus.experimental.optimizers` to `olympus.example_libraries.optimizers`
* New features:
  * Added `olympus.lax.linalg.qdwh`.

## olympus 0.2.24 (Oct 19, 2021)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.22...olympus-v0.2.24).

* New features:
  * `olympus.random.choice` and `olympus.random.permutation` now support
    multidimensional arrays and an optional `axis` argument ({olympus-issue}`#8158`)
* Breaking changes:
  * `olympus.numpy.take` and `olympus.numpy.take_along_axis` now require array-like inputs
    (see {olympus-issue}`#7737`)

## olympuslib 0.1.73 (Oct 18, 2021)

* Multiple cuDNN versions are now supported for olympuslib GPU `cuda11` wheels.
  * cuDNN 8.2 or newer. We recommend using the cuDNN 8.2 wheel if your cuDNN
    installation is new enough, since it supports additional functionality.
  * cuDNN 8.0.5 or newer.

* Breaking changes:
  * The install commands for GPU olympuslib are as follows:

    ```bash
    pip install --upgrade pip

    # Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
    pip install --upgrade "olympus[cuda]" -f https://storage.googleapis.com/olympus-releases/olympus_releases.html

    # Installs the wheel compatible with Cuda 11 and cudnn 8.2 or newer.
    pip install olympus[cuda11_cudnn82] -f https://storage.googleapis.com/olympus-releases/olympus_releases.html

    # Installs the wheel compatible with Cuda 11 and cudnn 8.0.5 or newer.
    pip install olympus[cuda11_cudnn805] -f https://storage.googleapis.com/olympus-releases/olympus_releases.html
    ```

## olympus 0.2.22 (Oct 12, 2021)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.21...olympus-v0.2.22).
* Breaking Changes
  * Static arguments to `olympus.pmap` must now be hashable.

    Unhashable static arguments have long been disallowed on `olympus.jit`, but they
    were still permitted on `olympus.pmap`; `olympus.pmap` compared unhashable static
    arguments using object identity.

    This behavior is a footgun, since comparing arguments using
    object identity leads to recompilation each time the object identity
    changes. Instead, we now ban unhashable arguments: if a user of `olympus.pmap`
    wants to compare static arguments by object identity, they can define
    `__hash__` and `__eq__` methods on their objects that do that, or wrap their
    objects in an object that has those operations with object identity
    semantics. Another option is to use `functools.partial` to encapsulate the
    unhashable static arguments into the function object.
  * `olympus.util.partial` was an accidental export that has now been removed. Use
    `functools.partial` from the Python standard library instead.
* Deprecations
  * The functions `olympus.ops.index_update`, `olympus.ops.index_add` etc. are
    deprecated and will be removed in a future OLYMPUS release. Please use
    [the `.at` property on OLYMPUS arrays](https://docs.olympus.dev/en/latest/_autosummary/olympus.numpy.ndarray.at.html)
    instead, e.g., `x.at[idx].set(y)`. For now, these functions produce a
    `DeprecationWarning`.
* New features:
  * An optimized C++ code-path improving the dispatch time for `pmap` is now the
    default when using olympuslib 0.1.72 or newer. The feature can be disabled using
    the `--experimental_cpp_pmap` flag (or `OLYMPUS_CPP_PMAP` environment variable).
  * `olympus.numpy.unique` now supports an optional `fill_value` argument ({olympus-issue}`#8121`)

## olympuslib 0.1.72 (Oct 12, 2021)
  * Breaking changes:
    * Support for CUDA 10.2 and CUDA 10.1 has been dropped. Olympuslib now supports
      CUDA 11.1+.
  * Bug fixes:
    * Fixes https://github.com/olympus-ml/olympus/issues/7461, which caused wrong
      outputs on all platforms due to incorrect buffer aliasing inside the XLA
      compiler.

## olympus 0.2.21 (Sept 23, 2021)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.20...olympus-v0.2.21).
* Breaking Changes
  * `olympus.api` has been removed. Functions that were available as `olympus.api.*`
    were aliases for functions in `olympus.*`; please use the functions in
    `olympus.*` instead.
  * `olympus.partial`, and `olympus.lax.partial` were accidental exports that have now
    been removed. Use `functools.partial` from the Python standard library
    instead.
  * Boolean scalar indices now raise a `TypeError`; previously this silently
    returned wrong results ({olympus-issue}`#7925`).
  * Many more `olympus.numpy` functions now require array-like inputs, and will error
    if passed a list ({olympus-issue}`#7747` {olympus-issue}`#7802` {olympus-issue}`#7907`).
    See {olympus-issue}`#7737` for a discussion of the rationale behind this change.
  * When inside a transformation such as `olympus.jit`, `olympus.numpy.array` always
    stages the array it produces into the traced computation. Previously
    `olympus.numpy.array` would sometimes produce a on-device array, even under
    a `olympus.jit` decorator. This change may break code that used OLYMPUS arrays to
    perform shape or index computations that must be known statically; the
    workaround is to perform such computations using classic NumPy arrays
    instead.
  * `jnp.ndarray` is now a true base-class for OLYMPUS arrays. In particular, this
    means that for a standard numpy array `x`, `isinstance(x, jnp.ndarray)` will
    now return `False` ({olympus-issue}`7927`).
* New features:
  * Added {func}`olympus.numpy.insert` implementation ({olympus-issue}`#7936`).

## olympus 0.2.20 (Sept 2, 2021)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.19...olympus-v0.2.20).
* Breaking Changes
  * `jnp.poly*` functions now require array-like inputs ({olympus-issue}`#7732`)
  * `jnp.unique` and other set-like operations now require array-like inputs
    ({olympus-issue}`#7662`)

## olympuslib 0.1.71 (Sep 1, 2021)
* Breaking changes:
  * Support for CUDA 11.0 and CUDA 10.1 has been dropped. Olympuslib now supports
    CUDA 10.2 and CUDA 11.1+.

## olympus 0.2.19 (Aug 12, 2021)
* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.18...olympus-v0.2.19).
* Breaking changes:
  * Support for NumPy 1.17 has been dropped, per the
    [deprecation policy](https://docs.olympus.dev/en/latest/deprecation.html).
    Please upgrade to a supported NumPy version.
  * The `jit` decorator has been added around the implementation of a number of
    operators on OLYMPUS arrays. This speeds up dispatch times for common
    operators such as `+`.

    This change should largely be transparent to most users. However, there is
    one known behavioral change, which is that large integer constants may now
    produce an error when passed directly to a OLYMPUS operator
    (e.g., `x + 2**40`). The workaround is to cast the constant to an
    explicit type (e.g., `np.float64(2**40)`).
* New features:
  * Improved the support for shape polymorphism in olympus2tf for operations that
    need to use a dimension size in array computation, e.g., `jnp.mean`.
    ({olympus-issue}`#7317`)
* Bug fixes:
  * Some leaked trace errors from the previous release ({olympus-issue}`#7613`)

## olympuslib 0.1.70 (Aug 9, 2021)
* Breaking changes:
  * Support for Python 3.6 has been dropped, per the
    [deprecation policy](https://docs.olympus.dev/en/latest/deprecation.html).
    Please upgrade to a supported Python version.
  * Support for NumPy 1.17 has been dropped, per the
    [deprecation policy](https://docs.olympus.dev/en/latest/deprecation.html).
    Please upgrade to a supported NumPy version.

  * The host_callback mechanism now uses one thread per local device for
    making the calls to the Python callbacks. Previously there was a single
    thread for all devices. This means that the callbacks may now be called
    interleaved. The callbacks corresponding to one device will still be
    called in sequence.

## olympus 0.2.18 (July 21 2021)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.17...olympus-v0.2.18).

* Breaking changes:
  * Support for Python 3.6 has been dropped, per the
    [deprecation policy](https://docs.olympus.dev/en/latest/deprecation.html).
    Please upgrade to a supported Python version.
  * The minimum olympuslib version is now 0.1.69.
  * The `backend` argument to {py:func}`olympus.dlpack.from_dlpack` has been
    removed.

* New features:
  * Added a polar decomposition ({py:func}`olympus.scipy.linalg.polar`).

* Bug fixes:
  * Tightened the checks for lax.argmin and lax.argmax to ensure they are
    not used with an invalid `axis` value, or with an empty reduction dimension.
    ({olympus-issue}`#7196`)


## olympuslib 0.1.69 (July 9 2021)
* Fix bugs in TFRT CPU backend that results in incorrect results.

## olympus 0.2.17 (July 9 2021)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.16...olympus-v0.2.17).
* Bug fixes:
  * Default to the older "stream_executor" CPU runtime for olympuslib <= 0.1.68
    to work around #7229, which caused wrong outputs on CPU due to a concurrency
    problem.
* New features:
  * New SciPy function {py:func}`olympus.scipy.special.sph_harm`.
  * Reverse-mode autodiff functions ({func}`olympus.grad`,
    {func}`olympus.value_and_grad`, {func}`olympus.vjp`, and
    {func}`olympus.linear_transpose`) support a parameter that indicates which named
    axes should be summed over in the backward pass if they were broadcasted
    over in the forward pass. This enables use of these APIs in a
    non-per-example way inside maps (initially only
    {func}`olympus.experimental.maps.xmap`) ({olympus-issue}`#6950`).


## olympus 0.2.16 (June 23 2021)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.15...olympus-v0.2.16).

## olympus 0.2.15 (June 23 2021)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.14...olympus-v0.2.15).
* New features:
  * [#7042](https://github.com/olympus-ml/olympus/pull/7042) Turned on TFRT CPU backend
    with significant dispatch performance improvements on CPU.
  * The {func}`olympus2tf.convert` supports inequalities and min/max for booleans
    ({olympus-issue}`#6956`).
  * New SciPy function {py:func}`olympus.scipy.special.lpmn_values`.

* Breaking changes:
  * Support for NumPy 1.16 has been dropped, per the
    [deprecation policy](https://docs.olympus.dev/en/latest/deprecation.html).

* Bug fixes:
  * Fixed bug that prevented round-tripping from OLYMPUS to TF and back:
    `olympus2tf.call_tf(olympus2tf.convert)` ({olympus-issue}`#6947`).

## olympuslib 0.1.68 (June 23 2021)
* Bug fixes:
  * Fixed bug in TFRT CPU backend that gets nans when transfer TPU buffer to
    CPU.

## olympus 0.2.14 (June 10 2021)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.13...olympus-v0.2.14).
* New features:
  * The {func}`olympus2tf.convert` now has support for `pjit` and `sharded_jit`.
  * A new configuration option OLYMPUS_TRACEBACK_FILTERING controls how OLYMPUS filters
    tracebacks.
  * A new traceback filtering mode using `__tracebackhide__` is now enabled by
    default in sufficiently recent versions of IPython.
  * The {func}`olympus2tf.convert` supports shape polymorphism even when the
    unknown dimensions are used in arithmetic operations, e.g., `jnp.reshape(-1)`
    ({olympus-issue}`#6827`).
  * The {func}`olympus2tf.convert` generates custom attributes with location information
   in TF ops. The code that XLA generates after olympus2tf
   has the same location information as OLYMPUS/XLA.
  * New SciPy function {py:func}`olympus.scipy.special.lpmn`.

* Bug fixes:
  * The {func}`olympus2tf.convert` now ensures that it uses the same typing rules
    for Python scalars and for choosing 32-bit vs. 64-bit computations
    as OLYMPUS ({olympus-issue}`#6883`).
  * The {func}`olympus2tf.convert` now scopes the `enable_xla` conversion parameter
    properly to apply only during the just-in-time conversion
    ({olympus-issue}`#6720`).
  * The {func}`olympus2tf.convert` now converts `lax.dot_general` using the
    `XlaDot` TensorFlow op, for better fidelity w.r.t. OLYMPUS numerical precision
    ({olympus-issue}`#6717`).
  * The {func}`olympus2tf.convert` now has support for inequality comparisons and
    min/max for complex numbers ({olympus-issue}`#6892`).

## olympuslib 0.1.67 (May 17 2021)

## olympuslib 0.1.66 (May 11 2021)

* New features:
  * CUDA 11.1 wheels are now supported on all CUDA 11 versions 11.1 or higher.

    NVidia now promises compatibility between CUDA minor releases starting with
    CUDA 11.1. This means that OLYMPUS can release a single CUDA 11.1 wheel that
    is compatible with CUDA 11.2 and 11.3.

    There is no longer a separate olympuslib release for CUDA 11.2 (or higher); use
    the CUDA 11.1 wheel for those versions (cuda111).
  * Olympuslib now bundles `libdevice.10.bc` in CUDA wheels. There should be no need
    to point OLYMPUS to a CUDA installation to find this file.
  * Added automatic support for static keyword arguments to the {func}`jit`
    implementation.
  * Added support for pretransformation exception traces.
  * Initial support for pruning unused arguments from {func}`jit` -transformed
    computations.
    Pruning is still a work in progress.
  * Improved the string representation of {class}`PyTreeDef` objects.
  * Added support for XLA's variadic ReduceWindow.
* Bug fixes:
  * Fixed a bug in the remote cloud TPU support when large numbers of arguments
    are passed to a computation.
  * Fix a bug that meant that OLYMPUS garbage collection was not triggered by
    {func}`jit` transformed functions.

## olympus 0.2.13 (May 3 2021)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.12...olympus-v0.2.13).
* New features:
  * When combined with olympuslib 0.1.66, {func}`olympus.jit` now supports static
    keyword arguments. A new `static_argnames` option has been added to specify
    keyword arguments as static.
  * {func}`olympus.nonzero` has a new optional `size` argument that allows it to
    be used within `jit` ({olympus-issue}`#6501`)
  * {func}`olympus.numpy.unique` now supports the `axis` argument ({olympus-issue}`#6532`).
  * {func}`olympus.experimental.host_callback.call` now supports `pjit.pjit` ({olympus-issue}`#6569`).
  * Added {func}`olympus.scipy.linalg.eigh_tridiagonal` that computes the
    eigenvalues of a tridiagonal matrix. Only eigenvalues are supported at
    present.
  * The order of the filtered and unfiltered stack traces in exceptions has been
    changed. The traceback attached to an exception thrown from OLYMPUS-transformed
    code is now filtered, with an `UnfilteredStackTrace` exception
    containing the original trace as the `__cause__` of the filtered exception.
    Filtered stack traces now also work with Python 3.6.
  * If an exception is thrown by code that has been transformed by reverse-mode
    automatic differentiation, OLYMPUS now attempts to attach as a `__cause__` of
    the exception a `OlympusStackTraceBeforeTransformation` object that contains the
    stack trace that created the original operation in the forward pass.
    Requires olympuslib 0.1.66.

* Breaking changes:
  * The following function names have changed. There are still aliases, so this
    should not break existing code, but the aliases will eventually be removed
    so please change your code.
    * `host_id` --> {func}`~olympus.process_index`
    * `host_count` --> {func}`~olympus.process_count`
    * `host_ids` --> `range(olympus.process_count())`
  * Similarly, the argument to {func}`~olympus.local_devices` has been renamed from
    `host_id` to `process_index`.
  * Arguments to {func}`olympus.jit` other than the function are now marked as
    keyword-only. This change is to prevent accidental breakage when arguments
    are added to `jit`.
* Bug fixes:
  * The {func}`olympus2tf.convert` now works in presence of gradients for functions
    with integer inputs ({olympus-issue}`#6360`).
  * Fixed assertion failure in {func}`olympus2tf.call_tf` when used with captured
    `tf.Variable` ({olympus-issue}`#6572`).

## olympuslib 0.1.65 (April 7 2021)

## olympus 0.2.12 (April 1 2021)
* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.11...v0.2.12).
* New features
  * New profiling APIs: {func}`olympus.profiler.start_trace`,
    {func}`olympus.profiler.stop_trace`, and {func}`olympus.profiler.trace`
  * {func}`olympus.lax.reduce` is now differentiable.
* Breaking changes:
  * The minimum olympuslib version is now 0.1.64.
  * Some profiler APIs names have been changed. There are still aliases, so this
    should not break existing code, but the aliases will eventually be removed
    so please change your code.
    * `TraceContext` --> {func}`~olympus.profiler.TraceAnnotation`
    * `StepTraceContext` --> {func}`~olympus.profiler.StepTraceAnnotation`
    * `trace_function` --> {func}`~olympus.profiler.annotate_function`
  * Omnistaging can no longer be disabled. See [omnistaging](https://github.com/olympus-ml/olympus/blob/main/docs/design_notes/omnistaging.md)
    for more information.
  * Python integers larger than the maximum `int64` value will now lead to an overflow
    in all cases, rather than being silently converted to `uint64` in some cases ({olympus-issue}`#6047`).
  * Outside X64 mode, Python integers outside the range representable by `int32` will now lead to an
    `OverflowError` rather than having their value silently truncated.
* Bug fixes:
  * `host_callback` now supports empty arrays in arguments and results ({olympus-issue}`#6262`).
  * {func}`olympus.random.randint` clips rather than wraps of out-of-bounds limits, and can now generate
    integers in the full range of the specified dtype ({olympus-issue}`#5868`)

## olympus 0.2.11 (March 23 2021)

* [GitHub
  commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.10...olympus-v0.2.11).
* New features:
  * [#6112](https://github.com/olympus-ml/olympus/pull/6112) added context managers:
    `olympus.enable_checks`, `olympus.check_tracer_leaks`, `olympus.debug_nans`,
    `olympus.debug_infs`, `olympus.log_compiles`.
  * [#6085](https://github.com/olympus-ml/olympus/pull/6085) added `jnp.delete`

* Bug fixes:
  * [#6136](https://github.com/olympus-ml/olympus/pull/6136) generalized
    `olympus.flatten_util.ravel_pytree` to handle integer dtypes.
  * [#6129](https://github.com/olympus-ml/olympus/issues/6129) fixed a bug with handling
    some constants like `enum.IntEnums`
  * [#6145](https://github.com/olympus-ml/olympus/pull/6145) fixed batching issues with
    incomplete beta functions
  * [#6014](https://github.com/olympus-ml/olympus/pull/6014) fixed H2D transfers during
    tracing
  * [#6165](https://github.com/olympus-ml/olympus/pull/6165) avoids OverflowErrors when
    converting some large Python integers to floats
* Breaking changes:
  * The minimum olympuslib version is now 0.1.62.


## olympuslib 0.1.64 (March 18 2021)

## olympuslib 0.1.63 (March 17 2021)

## olympus 0.2.10 (March 5 2021)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.9...olympus-v0.2.10).
* New features:
  * {func}`olympus.scipy.stats.chi2` is now available as a distribution with logpdf and pdf methods.
  * {func}`olympus.scipy.stats.betabinom` is now available as a distribution with logpmf and pmf methods.
  * Added {func}`olympus.experimental.olympus2tf.call_tf` to call TensorFlow functions
    from OLYMPUS ({olympus-issue}`#5627`)
    and [README](https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md#calling-tensorflow-functions-from-olympus)).
  * Extended the batching rule for `lax.pad` to support batching of the padding values.
* Bug fixes:
  * {func}`olympus.numpy.take` properly handles negative indices ({olympus-issue}`#5768`)
* Breaking changes:
  * OLYMPUS's promotion rules were adjusted to make promotion more consistent and
    invariant to JIT. In particular, binary operations can now result in weakly-typed
    values when appropriate. The main user-visible effect of the change is that
    some operations result in outputs of different precision than before; for
    example the expression `jnp.bfloat16(1) + 0.1 * jnp.arange(10)`
    previously returned a `float64` array, and now returns a `bfloat16` array.
    OLYMPUS's type promotion behavior is described at {ref}`type-promotion`.
  * {func}`olympus.numpy.linspace` now computes the floor of integer values, i.e.,
    rounding towards -inf rather than 0. This change was made to match NumPy
    1.20.0.
  * {func}`olympus.numpy.i0` no longer accepts complex numbers. Previously the
    function computed the absolute value of complex arguments. This change was
    made to match the semantics of NumPy 1.20.0.
  * Several {mod}`olympus.numpy` functions no longer accept tuples or lists in place
    of array arguments: {func}`olympus.numpy.pad`, :func`olympus.numpy.ravel`,
    {func}`olympus.numpy.repeat`, {func}`olympus.numpy.reshape`.
    In general, {mod}`olympus.numpy` functions should be used with scalars or array arguments.

## olympuslib 0.1.62 (March 9 2021)

* New features:
  * olympuslib wheels are now built to require AVX instructions on x86-64 machines
    by default. If you want to use OLYMPUS on a machine that doesn't support AVX,
    you can build a olympuslib from source using the `--target_cpu_features` flag
    to `build.py`. `--target_cpu_features` also replaces
    `--enable_march_native`.

## olympuslib 0.1.61 (February 12 2021)

## olympuslib 0.1.60 (February 3 2021)

* Bug fixes:
  * Fixed a memory leak when converting CPU DeviceArrays to NumPy arrays. The
    memory leak was present in olympuslib releases 0.1.58 and 0.1.59.
  * `bool`, `int8`, and `uint8` are now considered safe to cast to
    `bfloat16` NumPy extension type.

## olympus 0.2.9 (January 26 2021)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.8...olympus-v0.2.9).
* New features:
  * Extend the {mod}`olympus.experimental.loops` module with support for pytrees. Improved
    error checking and error messages.
  * Add {func}`olympus.experimental.enable_x64` and {func}`olympus.experimental.disable_x64`.
    These are context managers which allow X64 mode to be temporarily enabled/disabled
    within a session.
* Breaking changes:
  * {func}`olympus.ops.segment_sum` now drops segment IDs that are out of range rather
    than wrapping them into the segment ID space. This was done for performance
    reasons.

## olympuslib 0.1.59 (January 15 2021)

## olympus 0.2.8 (January 12 2021)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.7...olympus-v0.2.8).
* New features:
  * Add {func}`olympus.closure_convert` for use with higher-order custom
    derivative functions. ({olympus-issue}`#5244`)
  * Add {func}`olympus.experimental.host_callback.call` to call a custom Python
    function on the host and return a result to the device computation.
    ({olympus-issue}`#5243`)
* Bug fixes:
  * `olympus.numpy.arccosh` now returns the same branch as `numpy.arccosh` for
    complex inputs ({olympus-issue}`#5156`)
  * `host_callback.id_tap` now works for `olympus.pmap` also. There is an
    optional parameter for `id_tap` and `id_print` to request that the
    device from which the value is tapped be passed as a keyword argument
    to the tap function ({olympus-issue}`#5182`).
* Breaking changes:
  * `olympus.numpy.pad` now takes keyword arguments. Positional argument `constant_values`
    has been removed. In addition, passing unsupported keyword arguments raises an error.
  * Changes for {func}`olympus.experimental.host_callback.id_tap` ({olympus-issue}`#5243`):
    * Removed support for `kwargs` for {func}`olympus.experimental.host_callback.id_tap`.
      (This support has been deprecated for a few months.)
    * Changed the printing of tuples for {func}`olympus.experimental.host_callback.id_print`
      to use '(' instead of '['.
    * Changed the {func}`olympus.experimental.host_callback.id_print` in presence of JVP
      to print a pair of primal and tangent. Previously, there were two separate
      print operations for the primals and the tangent.
    * `host_callback.outfeed_receiver` has been removed (it is not necessary,
      and was deprecated a few months ago).
* New features:
  * New flag for debugging `inf`, analogous to that for `NaN` ({olympus-issue}`#5224`).

## olympus 0.2.7 (Dec 4 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.6...olympus-v0.2.7).
* New features:
  * Add `olympus.device_put_replicated`
  * Add multi-host support to `olympus.experimental.sharded_jit`
  * Add support for differentiating eigenvalues computed by `olympus.numpy.linalg.eig`
  * Add support for building on Windows platforms
  * Add support for general in_axes and out_axes in `olympus.pmap`
  * Add complex support for `olympus.numpy.linalg.slogdet`
* Bug fixes:
  * Fix higher-than-second order derivatives of `olympus.numpy.sinc` at zero
  * Fix some hard-to-hit bugs around symbolic zeros in transpose rules
* Breaking changes:
  * `olympus.experimental.optix` has been deleted, in favor of the standalone
    `optax` Python package.
  * indexing of OLYMPUS arrays with non-tuple sequences now raises a `TypeError`. This type of indexing
    has been deprecated in Numpy since v1.16, and in OLYMPUS since v0.2.4.
    See {olympus-issue}`#4564`.

## olympus 0.2.6 (Nov 18 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.5...olympus-v0.2.6).
* New Features:
  * Add support for shape-polymorphic tracing for the olympus.experimental.olympus2tf converter.
    See [README.md](https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/README.md).
* Breaking change cleanup

  * Raise an error on non-hashable static arguments for olympus.jit and
    xla_computation.  See [cb48f42](https://github.com/olympus-ml/olympus/commit/cb48f42).
  * Improve consistency of type promotion behavior ({olympus-issue}`#4744`):
    * Adding a complex Python scalar to a OLYMPUS floating point number respects the precision of
      the OLYMPUS float. For example, `jnp.float32(1) + 1j` now returns `complex64`, where previously
      it returned `complex128`.
    * Results of type promotion with 3 or more terms involving uint64, a signed int, and a third type
      are now independent of the order of arguments. For example:
      `jnp.result_type(jnp.uint64, jnp.int64, jnp.float16)` and
      `jnp.result_type(jnp.float16, jnp.uint64, jnp.int64)` both return `float16`, where previously
      the first returned `float64` and the second returned `float16`.
  * The contents of the (undocumented) `olympus.lax_linalg` linear algebra module
    are now exposed publicly as `olympus.lax.linalg`.
  * `olympus.random.PRNGKey` now produces the same results in and out of JIT compilation
    ({olympus-issue}`#4877`).
    This required changing the result for a given seed in a few particular cases:
    * With `olympus_enable_x64=False`, negative seeds passed as Python integers now return a different result
      outside JIT mode. For example, `olympus.random.PRNGKey(-1)` previously returned
      `[4294967295, 4294967295]`, and now returns `[0, 4294967295]`. This matches the behavior in JIT.
    * Seeds outside the range representable by `int64` outside JIT now result in an `OverflowError`
      rather than a `TypeError`. This matches the behavior in JIT.

    To recover the keys returned previously for negative integers with `olympus_enable_x64=False`
    outside JIT, you can use:

    ```
    key = random.PRNGKey(-1).at[0].set(0xFFFFFFFF)
    ```
  * DeviceArray now raises `RuntimeError` instead of `ValueError` when trying
    to access its value while it has been deleted.

## olympuslib 0.1.58 (January 12ish 2021)

* Fixed a bug that meant OLYMPUS sometimes return platform-specific types (e.g.,
  `np.cint`) instead of standard types (e.g., `np.int32`). (#4903)
* Fixed a crash when constant-folding certain int16 operations. (#4971)
* Added an `is_leaf` predicate to {func}`pytree.flatten`.

## olympuslib 0.1.57 (November 12 2020)

* Fixed manylinux2010 compliance issues in GPU wheels.
* Switched the CPU FFT implementation from Eigen to PocketFFT.
* Fixed a bug where the hash of bfloat16 values was not correctly initialized
  and could change (#4651).
* Add support for retaining ownership when passing arrays to DLPack (#4636).
* Fixed a bug for batched triangular solves with sizes greater than 128 but not
  a multiple of 128.
* Fixed a bug when performing concurrent FFTs on multiple GPUs (#3518).
* Fixed a bug in profiler where tools are missing (#4427).
* Dropped support for CUDA 10.0.

## olympus 0.2.5 (October 27 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.4...olympus-v0.2.5).
* Improvements:
  * Ensure that `check_olympuspr` does not perform FLOPS.  See {olympus-issue}`#4650`.
  * Expanded the set of OLYMPUS primitives converted by olympus2tf.
    See [primitives_with_limited_support.md](https://github.com/olympus-ml/olympus/blob/main/olympus/experimental/olympus2tf/primitives_with_limited_support.md).

## olympus 0.2.4 (October 19 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.3...olympus-v0.2.4).
* Improvements:
  * Add support for `remat` to olympus.experimental.host_callback.  See {olympus-issue}`#4608`.
* Deprecations

  * Indexing with non-tuple sequences is now deprecated, following a similar deprecation in Numpy.
    In a future release, this will result in a TypeError. See {olympus-issue}`#4564`.

## olympuslib 0.1.56 (October 14, 2020)

## olympus 0.2.3 (October 14 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.2...olympus-v0.2.3).
* The reason for another release so soon is we need to temporarily roll back a
  new jit fastpath while we look into a performance degradation

## olympus 0.2.2 (October 13 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.1...olympus-v0.2.2).

## olympus 0.2.1 (October 6 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.2.0...olympus-v0.2.1).
* Improvements:
  * As a benefit of omnistaging, the host_callback functions are executed (in program
    order) even if the result of the {py:func}`olympus.experimental.host_callback.id_print`/
    {py:func}`olympus.experimental.host_callback.id_tap` is not used in the computation.

## olympus (0.2.0) (September 23 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.77...olympus-v0.2.0).
* Improvements:
  * Omnistaging on by default. See {olympus-issue}`#3370` and
    [omnistaging](https://github.com/olympus-ml/olympus/blob/main/docs/design_notes/omnistaging.md)

## olympus (0.1.77) (September 15 2020)

* Breaking changes:
  * New simplified interface for {py:func}`olympus.experimental.host_callback.id_tap` (#4101)

## olympuslib 0.1.55 (September 8, 2020)

* Update XLA:
  * Fix bug in DLPackManagedTensorToBuffer (#4196)

## olympus 0.1.76 (September 8, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.75...olympus-v0.1.76).

## olympus 0.1.75 (July 30, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.74...olympus-v0.1.75).
* Bug Fixes:
  * make jnp.abs() work for unsigned inputs (#3914)
* Improvements:
  * "Omnistaging" behavior added behind a flag, disabled by default (#3370)

## olympus 0.1.74 (July 29, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.73...olympus-v0.1.74).
* New Features:
  * BFGS (#3101)
  * TPU support for half-precision arithmetic (#3878)
* Bug Fixes:
  * Prevent some accidental dtype warnings (#3874)
  * Fix a multi-threading bug in custom derivatives (#3845, #3869)
* Improvements:
  * Faster searchsorted implementation (#3873)
  * Better test coverage for olympus.numpy sorting algorithms (#3836)

## olympuslib 0.1.52 (July 22, 2020)

* Update XLA.

## olympus 0.1.73 (July 22, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.72...olympus-v0.1.73).
* The minimum olympuslib version is now 0.1.51.
* New Features:
  * olympus.image.resize. (#3703)
  * hfft and ihfft (#3664)
  * olympus.numpy.intersect1d (#3726)
  * olympus.numpy.lexsort (#3812)
  * `lax.scan` and the `scan` primitive support an `unroll`
    parameter for loop unrolling when lowering to XLA
    ({olympus-issue}`#3738`).
* Bug Fixes:
  * Fix reduction repeated axis error (#3618)
  * Fix shape rule for lax.pad for input dimensions of size 0. (#3608)
  * make psum transpose handle zero cotangents (#3653)
  * Fix shape error when taking JVP of reduce-prod over size 0 axis. (#3729)
  * Support differentiation through olympus.lax.all_to_all (#3733)
  * address nan issue in olympus.scipy.special.zeta (#3777)
* Improvements:
  * Many improvements to olympus2tf
  * Reimplement argmin/argmax using a single pass variadic reduction. (#3611)
  * Enable XLA SPMD partitioning by default. (#3151)
  * Add support for 0d transpose convolution (#3643)
  * Make LU gradient work for low-rank matrices (#3610)
  * support multiple_results and custom JVPs in jet (#3657)
  * Generalize reduce-window padding to support (lo, hi) pairs. (#3728)
  * Implement complex convolutions on CPU and GPU. (#3735)
  * Make jnp.take work for empty slices of empty arrays. (#3751)
  * Relax dimension ordering rules for dot_general. (#3778)
  * Enable buffer donation for GPU. (#3800)
  * Add support for base dilation and window dilation to reduce window op… (#3803)

## olympuslib 0.1.51 (July 2, 2020)

* Update XLA.
* Add new runtime support for host_callback.

## olympus 0.1.72 (June 28, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.71...olympus-v0.1.72).
* Bug fixes:
  * Fix an odeint bug introduced in the previous release, see
    {olympus-issue}`#3587`.

## olympus 0.1.71 (June 25, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.70...olympus-v0.1.71).
* The minimum olympuslib version is now 0.1.48.
* Bug fixes:
  * Allow `olympus.experimental.ode.odeint` dynamics functions to close over
    values with respect to which we're differentiating
    {olympus-issue}`#3562`.

## olympuslib 0.1.50 (June 25, 2020)

* Add support for CUDA 11.0.
* Drop support for CUDA 9.2 (we only maintain support for the last four CUDA
  versions.)
* Update XLA.

## olympuslib 0.1.49 (June 19, 2020)

* Bug fixes:
  * Fix build issue that could result in slow compiles
    (<https://github.com/tensorflow/tensorflow/commit/f805153a25b00d12072bd728e91bb1621bfcf1b1>)

## olympuslib 0.1.48 (June 12, 2020)

* New features:
  * Adds support for fast traceback collection.
  * Adds preliminary support for on-device heap profiling.
  * Implements `np.nextafter` for `bfloat16` types.
  * Complex128 support for FFTs on CPU and GPU.
* Bug fixes:
  * Improved float64 `tanh` accuracy on GPU.
  * float64 scatters on GPU are much faster.
  * Complex matrix multiplication on CPU should be much faster.
  * Stable sorts on CPU should actually be stable now.
  * Concurrency bug fix in CPU backend.

## olympus 0.1.70 (June 8, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.69...olympus-v0.1.70).
* New features:
  * `lax.switch` introduces indexed conditionals with multiple
    branches, together with a generalization of the `cond`
    primitive
    {olympus-issue}`#3318`.

## olympus 0.1.69 (June 3, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.68...olympus-v0.1.69).

## olympus 0.1.68 (May 21, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.67...olympus-v0.1.68).
* New features:
  * {func}`lax.cond` supports a single-operand form, taken as the argument
    to both branches
    {olympus-issue}`#2993`.
* Notable changes:
  * The format of the `transforms` keyword for the {func}`olympus.experimental.host_callback.id_tap`
    primitive has changed {olympus-issue}`#3132`.

## olympus 0.1.67 (May 12, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.66...olympus-v0.1.67).
* New features:
  * Support for reduction over subsets of a pmapped axis using `axis_index_groups`
    {olympus-issue}`#2382`.
  * Experimental support for printing and calling host-side Python function from
    compiled code. See [id_print and id_tap](https://docs.olympus.dev/en/latest/olympus.experimental.host_callback.html)
    ({olympus-issue}`#3006`).
* Notable changes:
  * The visibility of names exported from {mod}`olympus.numpy` has been
    tightened. This may break code that was making use of names that were
    previously exported accidentally.

## olympuslib 0.1.47 (May 8, 2020)

* Fixes crash for outfeed.

## olympus 0.1.66 (May 5, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.65...olympus-v0.1.66).
* New features:
  * Support for `in_axes=None` on {func}`pmap`
    {olympus-issue}`#2896`.

## olympuslib 0.1.46 (May 5, 2020)

* Fixes crash for linear algebra functions on Mac OS X (#432).
* Fixes an illegal instruction crash caused by using AVX512 instructions when
  an operating system or hypervisor disabled them (#2906).

## olympus 0.1.65 (April 30, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.64...olympus-v0.1.65).
* New features:
  * Differentiation of determinants of singular matrices
    {olympus-issue}`#2809`.
* Bug fixes:
  * Fix {func}`odeint` differentiation with respect to time of ODEs with
    time-dependent dynamics {olympus-issue}`#2817`,
    also add ODE CI testing.
  * Fix {func}`lax_linalg.qr` differentiation
    {olympus-issue}`#2867`.

## olympuslib 0.1.45 (April 21, 2020)

* Fixes segfault: {olympus-issue}`#2755`
* Plumb is_stable option on Sort HLO through to Python.

## olympus 0.1.64 (April 21, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.63...olympus-v0.1.64).
* New features:
  * Add syntactic sugar for functional indexed updates
    {olympus-issue}`#2684`.
  * Add {func}`olympus.numpy.linalg.multi_dot` {olympus-issue}`#2726`.
  * Add {func}`olympus.numpy.unique` {olympus-issue}`#2760`.
  * Add {func}`olympus.numpy.rint` {olympus-issue}`#2724`.
  * Add {func}`olympus.numpy.rint` {olympus-issue}`#2724`.
  * Add more primitive rules for {func}`olympus.experimental.jet`.
* Bug fixes:
  * Fix {func}`logaddexp` and {func}`logaddexp2` differentiation at zero {olympus-issue}`#2107`.
  * Improve memory usage in reverse-mode autodiff without {func}`jit`
    {olympus-issue}`#2719`.
* Better errors:
  * Improves error message for reverse-mode differentiation of {func}`lax.while_loop`
    {olympus-issue}`#2129`.

## olympuslib 0.1.44 (April 16, 2020)

* Fixes a bug where if multiple GPUs of different models were present, OLYMPUS
  would only compile programs suitable for the first GPU.
* Bugfix for `batch_group_count` convolutions.
* Added precompiled SASS for more GPU versions to avoid startup PTX compilation
  hang.

## olympus 0.1.63 (April 12, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.62...olympus-v0.1.63).
* Added `olympus.custom_jvp` and `olympus.custom_vjp` from {olympus-issue}`#2026`, see the [tutorial notebook](https://docs.olympus.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html). Deprecated `olympus.custom_transforms` and removed it from the docs (though it still works).
* Add `scipy.sparse.linalg.cg` {olympus-issue}`#2566`.
* Changed how Tracers are printed to show more useful information for debugging {olympus-issue}`#2591`.
* Made `olympus.numpy.isclose` handle `nan` and `inf` correctly {olympus-issue}`#2501`.
* Added several new rules for `olympus.experimental.jet` {olympus-issue}`#2537`.
* Fixed `olympus.experimental.stax.BatchNorm` when `scale`/`center` isn't provided.
* Fix some missing cases of broadcasting in `olympus.numpy.einsum` {olympus-issue}`#2512`.
* Implement `olympus.numpy.cumsum` and `olympus.numpy.cumprod` in terms of a parallel prefix scan {olympus-issue}`#2596` and make `reduce_prod` differentiable to arbitrary order {olympus-issue}`#2597`.
* Add `batch_group_count` to `conv_general_dilated` {olympus-issue}`#2635`.
* Add docstring for `test_util.check_grads` {olympus-issue}`#2656`.
* Add `callback_transform` {olympus-issue}`#2665`.
* Implement `rollaxis`, `convolve`/`correlate` 1d & 2d, `copysign`,
  `trunc`, `roots`, and `quantile`/`percentile` interpolation options.

## olympuslib 0.1.43 (March 31, 2020)

* Fixed a performance regression for Resnet-50 on GPU.

## olympus 0.1.62 (March 21, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.61...olympus-v0.1.62).
* OLYMPUS has dropped support for Python 3.5. Please upgrade to Python 3.6 or newer.
* Removed the internal function `lax._safe_mul`, which implemented the
  convention `0. * nan == 0.`. This change means some programs when
  differentiated will produce nans when they previously produced correct
  values, though it ensures nans rather than silently incorrect results are
  produced for other programs. See #2447 and #1052 for details.
* Added an `all_gather` parallel convenience function.
* More type annotations in core code.

## olympuslib 0.1.42 (March 19, 2020)

* olympuslib 0.1.41 broke cloud TPU support due to an API incompatibility. This
  release fixes it again.
* OLYMPUS has dropped support for Python 3.5. Please upgrade to Python 3.6 or newer.

## olympus 0.1.61 (March 17, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.60...olympus-v0.1.61).
* Fixes Python 3.5 support. This will be the last OLYMPUS or olympuslib release that
  supports Python 3.5.

## olympus 0.1.60 (March 17, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.59...olympus-v0.1.60).
* New features:
  * {py:func}`olympus.pmap` has `static_broadcast_argnums` argument which allows
    the user to specify arguments that should be treated as compile-time
    constants and should be broadcasted to all devices. It works analogously to
    `static_argnums` in {py:func}`olympus.jit`.
  * Improved error messages for when tracers are mistakenly saved in global state.
  * Added {py:func}`olympus.nn.one_hot` utility function.
  * Added {mod}`olympus.experimental.jet` for exponentially faster
    higher-order automatic differentiation.
  * Added more correctness checking to arguments of {py:func}`olympus.lax.broadcast_in_dim`.
* The minimum olympuslib version is now 0.1.41.

## olympuslib 0.1.40 (March 4, 2020)

* Adds experimental support in Olympuslib for TensorFlow profiler, which allows
  tracing of CPU and GPU computations from TensorBoard.
* Includes prototype support for multihost GPU computations that communicate via
  NCCL.
* Improves performance of NCCL collectives on GPU.
* Adds TopK, CustomCallWithoutLayout, CustomCallWithLayout, IGammaGradA and
  RandomGamma implementations.
* Supports device assignments known at XLA compilation time.

## olympus 0.1.59 (February 11, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/olympus-v0.1.58...olympus-v0.1.59).
* Breaking changes

  * The minimum olympuslib version is now 0.1.38.
  * Simplified {py:class}`Olympuspr` by removing the `Olympuspr.freevars` and
    `Olympuspr.bound_subolympusprs`. The call primitives (`xla_call`, `xla_pmap`,
    `sharded_call`, and `remat_call`) get a new parameter `call_olympuspr` with a
    fully-closed (no `constvars`) olympuspr. Also, added a new field `call_primitive`
    to primitives.
* New features:
  * Reverse-mode automatic differentiation (e.g. `grad`) of `lax.cond`, making it
    now differentiable in both modes ({olympus-issue}`#2091`)
  * OLYMPUS now supports DLPack, which allows sharing CPU and GPU arrays in a
    zero-copy way with other libraries, such as Blacksmit.
  * OLYMPUS GPU DeviceArrays now support `__cuda_array_interface__`, which is another
    zero-copy protocol for sharing GPU arrays with other libraries such as CuPy
    and Numba.
  * OLYMPUS CPU device buffers now implement the Python buffer protocol, which allows
    zero-copy buffer sharing between OLYMPUS and NumPy.
  * Added OLYMPUS_SKIP_SLOW_TESTS environment variable to skip tests known as slow.

## olympuslib 0.1.39 (February 11, 2020)

* Updates XLA.

## olympuslib 0.1.38 (January 29, 2020)

* CUDA 9.0 is no longer supported.
* CUDA 10.2 wheels are now built by default.

## olympus 0.1.58 (January 28, 2020)

* [GitHub commits](https://github.com/olympus-ml/olympus/compare/46014da21...olympus-v0.1.58).
* Breaking changes

  * OLYMPUS has dropped Python 2 support, because Python 2 reached its end of life on
    January 1, 2020. Please update to Python 3.5 or newer.
* New features

  >   > * Forward-mode automatic differentiation (`jvp`) of while loop
  >   ({olympus-issue}`#1980`)
  > * New NumPy and SciPy functions:
  >
  >   * {py:func}`olympus.numpy.fft.fft2`
  >   * {py:func}`olympus.numpy.fft.ifft2`
  >   * {py:func}`olympus.numpy.fft.rfft`
  >   * {py:func}`olympus.numpy.fft.irfft`
  >   * {py:func}`olympus.numpy.fft.rfft2`
  >   * {py:func}`olympus.numpy.fft.irfft2`
  >   * {py:func}`olympus.numpy.fft.rfftn`
  >   * {py:func}`olympus.numpy.fft.irfftn`
  >   * {py:func}`olympus.numpy.fft.fftfreq`
  >   * {py:func}`olympus.numpy.fft.rfftfreq`
  >   * {py:func}`olympus.numpy.linalg.matrix_rank`
  >   * {py:func}`olympus.numpy.linalg.matrix_power`
  >   * {py:func}`olympus.scipy.special.betainc`
  > * Batched Cholesky decomposition on GPU now uses a more efficient batched
  >   kernel.

### Notable bug fixes

* With the Python 3 upgrade, OLYMPUS no longer depends on `fastcache`, which should
  help with installation.
