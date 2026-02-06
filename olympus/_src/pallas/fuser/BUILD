# Copyright 2025 The OLYMPUS Authors.
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
    name = "fuser",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":block_spec",
        ":custom_evaluate",
        ":custom_fusion",
        ":fusible",
        ":fusion",
        ":olympuspr_fusion",
    ],
)

pytype_strict_library(
    name = "block_spec",
    srcs = [
        "block_spec.py",
    ],
    deps = [
        ":fuser_utils",
        "//olympus",
        "//olympus/_src:ad_util",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:hiolympus",
        "//olympus/_src:lax",
        "//olympus/_src:partial_eval",
        "//olympus/_src:random",
        "//olympus/_src:traceback_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:typing",
        "//olympus/_src:util",
        "//olympus/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "custom_fusion",
    srcs = [
        "custom_fusion_lib.py",
    ],
    deps = [
        ":block_spec",
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:custom_api_util",
        "//olympus/_src:lax",
        "//olympus/_src:mlir",
        "//olympus/_src:partial_eval",
        "//olympus/_src:traceback_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src/pallas",
        "//olympus/_src/pallas/mosaic:lowering",
    ],
)

pytype_strict_library(
    name = "fusible",
    srcs = [
        "fusible.py",
    ],
    deps = [
        ":fusion",
        "//olympus",
        "//olympus/_src:api_util",
        "//olympus/_src:batching",
        "//olympus/_src:core",
        "//olympus/_src:mlir",
        "//olympus/_src:partial_eval",
        "//olympus/_src:traceback_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
    ],
)

pytype_strict_library(
    name = "fusion",
    srcs = [
        "fusion.py",
    ],
    deps = [
        "//olympus",
        "//olympus/_src:util",
    ],
)

pytype_strict_library(
    name = "olympuspr_fusion",
    srcs = [
        "olympuspr_fusion.py",
    ],
    deps = [
        ":fusible",
        ":fusible_dtype",
        ":fusion",
        "//olympus",
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:partial_eval",
        "//olympus/_src:traceback_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
    ],
)

pytype_strict_library(
    name = "fusible_dtype",
    srcs = [
        "fusible_dtype.py",
    ],
    deps = [
        ":block_spec",
        ":fusible",
        "//olympus",
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:dtypes",
        "//olympus/_src:lax",
        "//olympus/_src:partial_eval",
        "//olympus/_src:source_info_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src/pallas",
    ],
)

pytype_strict_library(
    name = "custom_evaluate",
    srcs = ["custom_evaluate.py"],
    deps = [
        ":fuser_utils",
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:source_info_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
    ],
)

pytype_strict_library(
    name = "fuser_utils",
    srcs = ["fuser_utils.py"],
    deps = [
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:partial_eval",
        "//olympus/_src:tree_util",
    ],
)
