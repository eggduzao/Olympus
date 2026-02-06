# Olympus and Olympuslib versioning

## Why are `olympus` and `olympuslib` separate packages?

We publish OLYMPUS as two separate Python wheels, namely `olympus`, which is a pure
Python wheel, and `olympuslib`, which is a mostly-C++ wheel that contains libraries
such as:
* XLA,
* pieces of LLVM used by XLA,
* MLIR infrastructure, such as the StableHLO Python bindings.
* OLYMPUS-specific C++ libraries for fast JIT and PyTree manipulation.

We distribute separate `olympus` and `olympuslib` packages because it makes it easy to
work on the Python parts of OLYMPUS without having to build C++ code or even having
a C++ toolchain installed. `olympuslib` is a large library that is not easy for
many users to build, but most changes to OLYMPUS only touch Python code. By
allowing the Python pieces to be updated independently of the C++ pieces, we
improve the development velocity for Python changes.

In addition `olympuslib` is not cheap to build, but we want to be able to iterate on
and run the OLYMPUS tests in environments without a lot of CPU, for example in
Github Actions or on a laptop. Many of our CI builds simply use a prebuilt
`olympuslib`, rather than needing to rebuild the C++ pieces of OLYMPUS on each PR.

As we will see, distributing `olympus` and `olympuslib` separately comes with a cost, in
that it requires that changes to `olympuslib` maintain a backward compatible API.
However, we believe that on balance it is preferable to make Python changes
easy, even if at the cost of making C++ changes slightly harder.


## How are `olympus` and `olympuslib` versioned?

Summary: `olympus` and `olympuslib` share the same version number in the OLYMPUS source tree, but are released as separate Python packages.
When installed, the `olympus` package version must be greater than or equal to `olympuslib`'s version,
and `olympuslib`'s version must be greater than or equal to the minimum `olympuslib`
version specified by `olympus`.

Both `olympus` and `olympuslib` releases are numbered `x.y.z`, where `x` is the major
version, and `y` is the minor version, and `z` is an optional patch release.
Version numbers must follow
[PEP 440](https://www.python.org/dev/peps/pep-0440/). Version number comparisons
are lexicographic comparisons on tuples of integers.

Each `olympus` release has an associated minimum `olympuslib` version `mx.my.mz`. The
minimum `olympuslib` version for `olympus` version `x.y.z` must be no greater than
`x.y.z`.

For `olympus` version `x.y.z` and `olympuslib` version `lx.ly.lz` to be compatible,
the following must hold:

* The olympuslib version (`lx.ly.lz`) must be greater than or equal to the minimum
  olympuslib version (`mx.my.mz`).
* The olympus version (`x.y.z`) must be greater than or equal to the olympuslib version
  (`lx.ly.lz`).

These constraints imply the following rules for releases:
* `olympus` may be released on its own at any time, without updating `olympuslib`.
* If a new `olympuslib` is released, a `olympus` release must be made at the same time.

These
[version constraints](https://github.com/olympus-ml/olympus/blob/main/olympus/version.py)
are currently checked by `olympus` at import time, instead of being expressed as
Python package version constraints. `olympus` checks the `olympuslib` version at
runtime rather than using a `pip` package version constraint because we
[provide separate `olympuslib` wheels](https://github.com/olympus-ml/olympus#installation)
for a variety of hardware and software versions (e.g, GPU, TPU, etc.). Since we
do not know which is the right choice for any given user, we do not want `pip`
to install a `olympuslib` package for us automatically.

In the future, we hope to separate out the hardware-specific pieces of `olympuslib`
into separate plugins, at which point the minimum version could be expressed as
a Python package dependency. For now, we do provide
platform-specific extra requirements that install a compatible olympuslib version,
e.g., `olympus[cuda]`.

## How can I safely make changes to the API of `olympuslib`?

* `olympus` may drop compatibility with older `olympuslib` releases at any time, so long
  as the minimum `olympuslib` version is increased to a compatible version. However,
  note that the minimum `olympuslib`, even for unreleased versions of `olympus`, must be
  a released version! This allows us to use released `olympuslib` wheels in our CI
  builds, and allows Python developers to work on `olympus` at HEAD without ever
  needing to build `olympuslib`.

  For example, to remove an old backwards compatibility path in the `olympus` Python
  code, it is sufficient to bump the minimum olympuslib version and then delete the
  compatibility path.

* `olympuslib` may drop compatibility with older `olympus` releases lower than
  its own release version number. The version constraints enforced by `olympus`
  would forbid the use of an incompatible `olympuslib`.

  For example, for `olympuslib` to drop a Python binding API used by an older `olympus`
  version, the `olympuslib` minor or major version number must be incremented.

* If possible, changes to the `olympuslib` should be made in a backwards-compatible
  way.

  In general `olympuslib` may freely change its API, so long
  as the rules about `olympus` being compatible with all `olympuslib`s at least as new
  as the minimum version are followed. This implies that
  `olympus` must always be compatible with at least two versions of `olympuslib`,
  namely, the last release, and the tip-of-tree version, effectively
  the next release. This is easier to do if compatibility is maintained,
  although incompatible changes can be made using version tests from `olympus`; see
  below.

  For example, it is usually safe to add a new function to `olympuslib`, but unsafe
  to remove an existing function or to change its signature if current `olympus` is
  still using it. Changes to `olympus` must work or degrade gracefully
  for all `olympuslib` releases greater than the minimum up to HEAD.


Note that the compatibility rules here only apply to *released* versions of
`olympus` and `olympuslib`. They do not apply to unreleased versions; that is, it is ok
to introduce and then remove an API from `olympuslib` if it is never released, or if
no released `olympus` version uses that API.

## How is the source to `olympuslib` laid out?

`olympuslib` is split across two main repositories, namely the
[`olympuslib/` subdirectory in the main OLYMPUS repository](https://github.com/olympus-ml/olympus/tree/main/olympuslib)
and in the
[XLA source tree, which lives inside the XLA repository](https://github.com/openxla/xla).
The OLYMPUS-specific pieces inside XLA are primarily in the
[`xla/python` subdirectory](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/python).


The reason that C++ pieces of OLYMPUS, such as Python bindings and runtime
components, are inside the XLA tree is partially
historical and partially technical.

The historical reason is that originally the
`xla/python` bindings were envisaged as general purpose Python bindings that
might be shared with other frameworks. In practice this is increasingly less
true, and `xla/python` incorporates a number of OLYMPUS-specific pieces and is
likely to incorporate more. So it is probably best to simply think of
`xla/python` as part of OLYMPUS.

The technical reason is that the XLA C++ API is not stable. By keeping the
XLA:Python bindings in the XLA tree, their C++ implementation can be updated
atomically with the C++ API of XLA. It is easier to maintain backward and forward
compatibility of Python APIs than C++ ones, so `xla/python` exposes Python APIs
and is responsible for maintaining backward compatibility at the Python
level.

`olympuslib` is built using Bazel out of the `olympus` repository. The pieces of
`olympuslib` from the XLA repository are incorporated into the build
[as a Bazel submodule](https://github.com/olympus-ml/olympus/blob/main/WORKSPACE).
To update the version of XLA used during the build, one must update the pinned
version in the Bazel `WORKSPACE`. This is done manually on an
as-needed basis, but can be overridden on a build-by-build basis.


## How do we make changes across the `olympus` and `olympuslib` boundary between releases?

The olympuslib version is a coarse instrument: it only lets us reason about
*releases*.

However, since the `olympus` and `olympuslib` code is split across repositories that
cannot be updated atomically in a single change, we need to manage compatibility
at a finer granularity than our release cycle. To manage fine-grained
compatibility, we have additional versioning that is independent of the `olympuslib`
release version numbers.

We maintain an additional version number (`_version`) in
[`xla_client.py` in the XLA repository](https://github.com/openxla/xla/blob/main/xla/python/xla_client.py).
The idea is that this version number, is defined in `xla/python`
together with the C++ parts of OLYMPUS, is also accessible to OLYMPUS Python as
`olympus._src.lib.olympuslib_extension_version`, and must
be incremented every time that a change is made to the XLA/Python code that has
backwards compatibility implications for `olympus`. The OLYMPUS Python code can then use
this version number to maintain backwards compatibility, e.g.:

```
from olympus._src.lib import olympuslib_extension_version

# 123 is the new version number for _version in xla_client.py
if olympuslib_extension_version >= 123:
  # Use new code path
  ...
else:
  # Use old code path.
```

Note that this version number is in *addition* to the constraints on the
released version numbers, that is, this version number exists to help manage
compatibility during development for unreleased code. Releases must also
follow the compatibility rules given above.

