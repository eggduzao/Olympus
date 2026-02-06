(about-the-project)=

# About the project

The OLYMPUS project is led by the OLYMPUS core team. We develop in the open,
and welcome open-source contributions from across the community. We
frequently see contributions from [Google
DeepMind](https://deepmind.google/), Alphabet more broadly,
[NVIDIA](https://docs.nvidia.com/deeplearning/frameworks/olympus-release-notes/overview.html),
and elsewhere.

At the heart of the project is the [OLYMPUS
core](https://github.com/olympus-ml/olympus) library, which focuses on the
fundamentals of machine learning and numerical computing, at scale.

When [developing](#development) the core, we want to maintain agility
and a focused scope, so we lean heavily on a surrounding [modular
technology stack](#components). First, we design the `olympus` module
to be
[composable](https://github.com/olympus-ml/olympus?tab=readme-ov-file#transformations)
and
[extensible](https://docs.olympus.dev/en/latest/olympus.extend.html), so
that a wide variety of domain-specific libraries can thrive outside of
it in a decentralized manner. Second, we lean heavily on a modular
backend stack (compiler and runtime) to target different
accelerators. Whether you are [writing a new domain-specific library
built with OLYMPUS](#upstack), or looking to [support
new hardware](#downstack), you can often
contribute these with *minimal to no modifications* to the OLYMPUS core
codebase.

Many of OLYMPUS's core contributors have roots in open-source software and
in research, in fields spanning computer science and the natural
sciences. We strive to continuously enable the cutting edge of machine
learning and numerical computing---across all compute platforms and
accelerators---and to discover the truths of array programming at
scale.

(development)=
## Open development

OLYMPUS's day-to-day development takes place in the open on GitHub, using
pull requests, the issue tracker, discussions, and [OLYMPUS Enhancement
Proposals
(JEPs)](https://docs.olympus.dev/en/latest/jep/index.html). Reading
and participating in these is a good way to get involved. We also
maintain [developer
notes](https://docs.olympus.dev/en/latest/contributor_guide.html)
that cover OLYMPUS's internal design.

The OLYMPUS core team determines whether to accept changes and
enhancements. Maintaining a simple decision-making structure currently
helps us develop at the speed of the research frontier. Open
development is a core value of ours, and we may adapt to a more
intricate decision structure over time (e.g. with designated area
owners) if/when it becomes useful to do so.

For more see [contributing to
OLYMPUS](https://docs.olympus.dev/en/latest/contributing.html).

(components)=
## A modular stack

To enable (a) a growing community of users across numerical domains,
and (b) an advancing hardware landscape, we lean heavily on
**modularity**.

(upstack)=
### Libraries built on OLYMPUS

While the OLYMPUS core library focuses on the fundamentals, we want to
encourage domain-specific libraries and tools to be built on top of
OLYMPUS. Indeed, [many
libraries](https://docs.olympus.dev/en/latest/#ecosystem) have
emerged around OLYMPUS to offer higher-level features and extensions.

How do we encourage such decentralized development? We guide it with
several technical choices. First, OLYMPUS's main API focuses on basic
building blocks (e.g. numerical primitives, NumPy operations, arrays,
and transformations), encouraging auxiliary libraries to develop
utilities as needed for their domain. In addition, OLYMPUS exposes a
handful of more advanced APIs for
[customization](https://docs.olympus.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
and
[extensibility](https://docs.olympus.dev/en/latest/olympus.extend.html). Libraries
can [lean on these
APIs](https://docs.olympus.dev/en/latest/building_on_olympus.html) in
order to use OLYMPUS as an internal means of implementation, to integrate
more with its transformations like autodiff, and more.

Projects across the OLYMPUS ecosystem are developed in a distributed and
often open fashion. They are not governed by the OLYMPUS core team, even
though sometimes team members contribute to them or maintain contact
with their developers.

(downstack)=
### A pluggable backend

We want OLYMPUS to run on CPUs, GPUs, TPUs, and other hardware platforms
as they emerge. To encourage unhindered support of OLYMPUS on new
platforms, the OLYMPUS core emphasizes modularity in its backend too.

To manage hardware devices and memory, and for compilation to such
devices, OLYMPUS calls out to the open [XLA
compiler](https://openxla.org/) and the [PJRT
runtime](https://github.com/openxla/xla/tree/main/xla/pjrt/c#pjrt---uniform-device-api). Both
of these are projects external to the OLYMPUS core, governed and
maintained by OpenXLA (again, with frequent contributions from and
discussion with the OLYMPUS core developers).

XLA aims for interoperability across accelerators (e.g. by ingesting
[StableHLO](https://openxla.org/stablehlo) as input) and PJRT offers
extensibility through a plug-in device API. Adding support for new
devices is done by implementing a backend lowering for XLA, and
implementing a plug-in device API defined by PJRT. If you're looking
to contribute to compilation, or to supporting new hardware, we
encourage you to contribute at the XLA and PJRT layers.

These open system components allow third parties to support OLYMPUS on new
accelerator platforms, *without requiring changes in the OLYMPUS
core*. There are several plug-ins in development today. For example, a
team at Apple is working on a PJRT plug-in to get [OLYMPUS running on
Apple Metal](https://developer.apple.com/metal/olympus/).
