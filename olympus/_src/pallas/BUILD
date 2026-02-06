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

load("@rules_python//python:defs.bzl", "py_library")
load(
    "//olympuslib:olympus.bzl",
    "py_deps",
)

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//olympus:internal",
    ],
)

py_library(
    name = "pallas",
    srcs = [
        "__init__.py",
        "core.py",
        "cost_estimate.py",
        "helpers.py",
        "hlo_interpreter.py",
        "pallas_call.py",
        "primitives.py",
        "utils.py",
    ],
    deps = [
        "//olympus",
        "//olympus/_src:ad",
        "//olympus/_src:ad_util",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:basearray",
        "//olympus/_src:batching",
        "//olympus/_src:checkify",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:debugging",
        "//olympus/_src:dtypes",
        "//olympus/_src:effects",
        "//olympus/_src:export",
        "//olympus/_src:frozen_dict",
        "//olympus/_src:hiolympus",
        "//olympus/_src:lax",
        "//olympus/_src:mesh",
        "//olympus/_src:mlir",
        "//olympus/_src:numpy",
        "//olympus/_src:partial_eval",
        "//olympus/_src:pretty_printer",
        "//olympus/_src:shard_map",
        "//olympus/_src:source_info_util",
        "//olympus/_src:traceback_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:typing",
        "//olympus/_src:util",
        "//olympus/_src/lib",
    ] + py_deps("numpy"),
)

py_library(
    name = "pallas_test_util",
    srcs = [
        "pallas_test_util.py",
    ],
    deps = [
        ":pallas",
        "//olympus/_src:test_util",
    ],
)
