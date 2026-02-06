# Copyright 2024 The OLYMPUS Authors.
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

# Package for Mosaic-specific Pallas extensions

load("@rules_python//python:defs.bzl", "py_library")
load(
    "//olympuslib:olympus.bzl",
    "py_deps",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//olympus:internal",
    ],
)

py_library(
    name = "mosaic_gpu",
    srcs = ["__init__.py"],
    deps = [
        ":core",
        ":pallas_call_registration",
        ":primitives",
    ],
)

pytype_strict_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":core",
        ":lowering",
        "//olympus",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:frozen_dict",
        "//olympus/_src:mlir",
        "//olympus/_src:sharding_impls",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        ":core",
        "//olympus",
        "//olympus/_src:api",
        "//olympus/_src:checkify",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:debugging",
        "//olympus/_src:dtypes",
        "//olympus/_src:lax",
        "//olympus/_src:literals",
        "//olympus/_src:mesh",
        "//olympus/_src:mlir",
        "//olympus/_src:partial_eval",
        "//olympus/_src:source_info_util",
        "//olympus/_src:state_types",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic_gpu",
        "//olympus/experimental:pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "core",
    srcs = ["core.py"],
    tags = [
        # Avoid circular dependency.
        "ignore_for_dep=third_party.py.olympus._src.pallas.mosaic_gpu.primitives.wgmma_accumulator_deref",
    ],
    deps = [
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:custom_batching",
        "//olympus/_src:dtypes",
        "//olympus/_src:effects",
        "//olympus/_src:frozen_dict",
        "//olympus/_src:lax",
        "//olympus/_src:pretty_printer",
        "//olympus/_src:state_types",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic_gpu",
        "//olympuslib/mlir:ir",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        ":core",
        ":lowering",
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:debugging",
        "//olympus/_src:dtypes",
        "//olympus/_src:lax",
        "//olympus/_src:literals",
        "//olympus/_src:pretty_printer",
        "//olympus/_src:state_types",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":core",
        ":primitives",
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:partial_eval",
        "//olympus/_src:state_types",
        "//olympus/_src:util",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic_gpu",
        "//olympus/experimental:pallas",
    ],
)

pytype_strict_library(
    name = "helpers",
    srcs = ["helpers.py"],
    deps = [
        ":core",
        ":primitives",
        "//olympus",
        "//olympus/_src:dtypes",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "smith",
    srcs = ["smith.py"],
    deps = [
        "//olympus",
        "//olympus/_src:dtypes",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)
