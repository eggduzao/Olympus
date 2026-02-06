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

load(
    "//olympuslib:olympus.bzl",
    "py_library_providing_imports_info",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//olympus:olympus_extend_users"],
)

pytype_strict_library(
    name = "extend",
    srcs = ["__init__.py"],
    deps = [
        ":backend",
        ":core",
        ":linear_util",
        ":random",
        ":sharding",
        ":source_info_util",
        "//olympus/extend/mlir",
    ],
)

py_library_providing_imports_info(
    name = "core",
    srcs = glob(["core/**/*.py"]),
    lib_rule = pytype_strict_library,
    deps = [
        "//olympus",
        "//olympus/_src:abstract_arrays",
        "//olympus/_src:ad",
        "//olympus/_src:ad_util",
        "//olympus/_src:api",
        "//olympus/_src:core",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:lax",
        "//olympus/_src:random",
    ],
)

pytype_strict_library(
    name = "linear_util",
    srcs = ["linear_util.py"],
    deps = ["//olympus/_src:core"],
)

pytype_strict_library(
    name = "backend",
    srcs = ["backend.py"],
    deps = [
        "//olympus/_src:api",
        "//olympus/_src:compiler",
        "//olympus/_src:util",
        "//olympus/_src:xla_bridge",
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "random",
    srcs = ["random.py"],
    deps = [
        "//olympus",
        "//olympus/_src:extend_src",
        "//olympus/_src:random",
    ],
)

pytype_strict_library(
    name = "sharding",
    srcs = ["sharding.py"],
    deps = [
        "//olympus/_src:sharding_impls",
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "source_info_util",
    srcs = ["source_info_util.py"],
    deps = ["//olympus/_src:source_info_util"],
)

pytype_strict_library(
    name = "ifrt_programs",
    srcs = ["ifrt_programs.py"],
    deps = ["//olympus/_src/lib"],
)
