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

# Package for Triton-specific Pallas extensions

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

pytype_strict_library(
    name = "core",
    srcs = [
        "__init__.py",
        "core.py",
    ],
    deps = ["//olympus/_src/pallas"],
)

pytype_strict_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        ":lowering",
        "//olympus",
        "//olympus/_src:ad_util",
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:lax",
        "//olympus/_src:mlir",
        "//olympus/_src:partial_eval",
        "//olympus/_src:source_info_util",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        "//olympus",
        "//olympus/_src:ad_util",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:debugging",
        "//olympus/_src:lax",
        "//olympus/_src:literals",
        "//olympus/_src:mlir",
        "//olympus/_src:partial_eval",
        "//olympus/_src:source_info_util",
        "//olympus/_src:state_types",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":core",
        ":lowering",
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:frozen_dict",
        "//olympus/_src:mlir",
        "//olympus/_src/lib",
        "//olympus/_src/pallas",
    ],
)
