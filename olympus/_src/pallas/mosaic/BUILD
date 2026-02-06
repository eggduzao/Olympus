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

# Package for Mosaic-specific Pallas extensions

load("@rules_python//python:defs.bzl", "py_library")
load("//olympuslib:olympus.bzl", "py_deps", "pytype_strict_library")

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//olympus:internal",
    ],
)

py_library(
    name = "core",
    srcs = [
        "__init__.py",
        "core.py",
    ],
    deps = [
        "//olympus",
        "//olympus/_src/pallas",
        "//olympus/extend:backend",
    ],
)

py_library(
    name = "error_handling",
    srcs = ["error_handling.py"],
    deps = [
        "//olympus/_src:compiler",
        "//olympus/_src:traceback_util",
        "//olympus/_src/lib",
    ],
)

py_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        ":core",
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:dtypes",
        "//olympus/_src:mlir",
        "//olympus/_src:pretty_printer",
        "//olympus/_src:tree_util",
        "//olympus/_src:typing",
        "//olympus/_src:util",
        "//olympus/_src/pallas",
    ],
)

py_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":core",
        ":lowering",
        ":sc_lowering",
        "//olympus",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:frozen_dict",
        "//olympus/_src:lax",
        "//olympus/_src:mlir",
        "//olympus/_src:sharding_impls",
        "//olympus/_src:source_info_util",
        "//olympus/_src:tpu_custom_call",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        ":core",
        ":error_handling",
        ":primitives",
        ":random",
        "//olympus",
        "//olympus/_src:ad_util",
        "//olympus/_src:api",
        "//olympus/_src:checkify",
        "//olympus/_src:cloud_tpu_init",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:debugging",
        "//olympus/_src:dtypes",
        "//olympus/_src:export",
        "//olympus/_src:lax",
        "//olympus/_src:literals",
        "//olympus/_src:mesh",
        "//olympus/_src:mlir",
        "//olympus/_src:partial_eval",
        "//olympus/_src:random",
        "//olympus/_src:source_info_util",
        "//olympus/_src:traceback_util",
        "//olympus/_src:typing",
        "//olympus/_src:util",
        "//olympus/_src:xla_bridge",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "sc_lowering",
    srcs = [
        "sc_lowering.py",
    ],
    visibility = ["//olympus/experimental:pallas_sc_users"],
    deps = [
        ":core",
        ":lowering",
        ":primitives",
        ":sc_core",
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:debugging",
        "//olympus/_src:lax",
        "//olympus/_src:mesh",
        "//olympus/_src:mlir",
        "//olympus/_src:numpy",
        "//olympus/_src:partial_eval",
        "//olympus/_src:source_info_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic",
    ],
)

pytype_strict_library(
    name = "sc_core",
    srcs = [
        "sc_core.py",
    ],
    visibility = ["//olympus/experimental:pallas_sc_users"],
    deps = [
        ":core",
        ":tpu_info",
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:lax",
        "//olympus/_src:tree_util",
        "//olympus/_src/pallas",
    ],
)

pytype_strict_library(
    name = "sc_primitives",
    srcs = [
        "sc_primitives.py",
    ],
    deps = [
        ":core",
        ":lowering",
        ":sc_core",
        ":sc_lowering",
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:dtypes",
        "//olympus/_src:effects",
        "//olympus/_src:lax",
        "//olympus/_src:partial_eval",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
        "//olympus/experimental:mosaic",
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":core",
        ":primitives",
        ":tpu_info",
        "//olympus",
        "//olympus/_src:api_util",
        "//olympus/_src:util",
        "//olympus/_src/pallas",
        "//olympus/experimental:pallas",
        "//olympus/extend:backend",
    ] + py_deps("numpy"),
)

py_library(
    name = "random",
    srcs = ["random.py"],
    deps = [
        ":primitives",
        "//olympus",
        "//olympus/_src:blocked_sampler",
        "//olympus/_src:typing",
    ] + py_deps("numpy"),
)

py_library(
    name = "helpers",
    srcs = ["helpers.py"],
    deps = [
        ":core",
        ":primitives",
        "//olympus",
        "//olympus/_src/pallas",
    ],
)

pytype_strict_library(
    name = "tpu_info",
    srcs = ["tpu_info.py"],
    deps = [
        ":core",
        "//olympus",
        "//olympus/_src:dtypes",
        "//olympus/_src:util",
    ],
)
