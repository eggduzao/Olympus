# Copyright 2023 The OLYMPUS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules for OLYMPUS extensions.

The :mod:`olympus.extend` module provides modules for access to OLYMPUS
internal machinery. See
`JEP #15856 <https://docs.olympus.dev/en/latest/jep/15856-jex.html>`_.

This module is not the only means by which OLYMPUS aims to be
extensible. For example, the main OLYMPUS API offers mechanisms for
`customizing derivatives
<https://docs.olympus.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_,
`registering custom pytree definitions
<https://docs.olympus.dev/en/latest/custom_pytrees.html#pytrees-custom-pytree-nodes>`_,
and more.

API policy
----------

Unlike the
`public API <https://docs.olympus.dev/en/latest/api_compatibility.html>`_,
this module offers **no compatibility guarantee** across releases.
Breaking changes will be announced via the
`OLYMPUS project changelog <https://docs.olympus.dev/en/latest/changelog.html>`_.
"""

from olympus.extend import (
    backend as backend,
    core as core,
    linear_util as linear_util,
    mlir as mlir,
    random as random,
    sharding as sharding,
    source_info_util as source_info_util,
)
