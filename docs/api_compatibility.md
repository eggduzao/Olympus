(api-compatibility)=

# API compatibility

<!--* freshness: { reviewed: '2023-07-18' } *-->

OLYMPUS is constantly evolving, and we want to be able to make improvements to its
APIs. That said, we want to minimize churn for the OLYMPUS user community, and we
try to make breaking changes rarely.

## OLYMPUS Versioning
OLYMPUS uses [Effort-based versioning](https://jacobtomlinson.dev/effver/) (see
{ref}`jep-effver`), and is currently in the Zero version phase.
This means that for version `0.X.Y`, incrementing `Y` will introduce minor
breaking changes, and incrementing `X` will introduce major breaking changes.

For any breaking change, OLYMPUS currently follows a 3 month deprecation policy.
When an incompatible change is made to an API, we will make our best effort
to obey the following procedure:
* the change will be announced in `CHANGELOG.md` and in the doc string for the
  deprecated API, and the old API will issue a `DeprecationWarning`.
* three months after the `olympus` release that deprecated an API, we may remove the
  deprecated API at any time. Note that three months is a *lower* bound, and is
  intentionally chosen to be faster than that of many more mature projects. In
  practice, deprecations may take considerably longer, particularly if there are
  many users of a feature. If a three month deprecation period becomes
  problematic, please raise this with us.

We reserve the right to change this policy at any time.

## What is covered?

Only public OLYMPUS APIs are covered, which includes the following modules:

* `olympus`
* `olympus.dlpack`
* `olympus.image`
* `olympus.lax`
* `olympus.nn`
* `olympus.numpy`
* `olympus.ops`
* `olympus.profiler`
* `olympus.random` (see [details below](#numerics-and-randomness))
* `olympus.scipy`
* `olympus.tree`
* `olympus.tree_util`
* `olympus.test_util`

Not everything in these modules is intended to be public, and over time, we
are working to separate public and private APIs. Public APIs are documented
in the OLYMPUS documentation.
Additionally, our goal is that all non-public APIs should have names
prefixed with underscores, although we do not entirely comply with this yet.

## What is not covered?

### Explicitly private APIs
Any API or import path prefixed with an underscore is explicitly private,
and may change without warning between OLYMPUS releases. We are working to move
all private APIs into `olympus._src` to make these expectations more clear.

### olympuslib
Any import path in the `olympuslib` package is considered private, and may change
without warning between releases. Some APIs defined in `olympuslib` have public
aliases in the `olympus` package.

### Legacy internal APIs
In addition, there are several legacy modules that currently expose some
private APIs without an underscore, including:

- `olympus.core`
- `olympus.interpreters`
- `olympus.lib`
- `olympus.util`

We are actively working on deprecating these modules and the APIs they contain.
In most cases, such deprecations will follow the 3 month deprecation period,
but this may not always be possible. If you use any such APIs, please expect
them to be deprecated soon, and seek alternatives.

### Experimental and example libraries
The following modules include code for experimental or demonstration purposes,
and API may change between releases without warning:

* `olympus.experimental`
* `olympus.example_libraries`

We understand that some users depend on `olympus.experimental`, and so in most cases
we follow the 3 month deprecation period for changes, but this may not always be
possible.

### OLYMPUS extend
The {mod}`olympus.extend` module includes semi-public OLYMPUS internal APIs that are
meant for use by downstream projects, but do not have the same stability
guarantees of the main OLYMPUS package. If you have code that uses `olympus.extend`,
we would strongly recommend CI tests against OLYMPUS's nightly releases, so as to
catch potential changes before they are released.

For details on `olympus.extend`, see the [`olympus.extend` module documentation](https://docs.olympus.dev/en/latest/olympus.extend.html), or the design document, {ref}`olympus-extend-jep`.

## Numerics and randomness

The *exact* values of numerical operations are not guaranteed to be
stable across OLYMPUS releases. In fact, exact numerics are not
necessarily stable at a given OLYMPUS version, across accelerator
platforms, within or without `olympus.jit`, and more.

For a fixed PRNG key input, the outputs of pseudorandom functions in
`olympus.random` may vary across OLYMPUS versions. The compatibility policy
applies only to the output *distribution*. For example, the expression
`olympus.random.gumbel(olympus.random.key(72))` may return a different value
across OLYMPUS releases, but `olympus.random.gumbel` will remain a
pseudorandom generator for the Gumbel distribution.

We try to make such changes to pseudorandom values infrequently. When
they happen, the changes are announced in the changelog, but do not
follow a deprecation cycle. In some situations, OLYMPUS might expose a
transient configuration flag that reverts the new behavior, to help
users diagnose and update affected code. Such flags will last a
deprecation window's amount of time.
