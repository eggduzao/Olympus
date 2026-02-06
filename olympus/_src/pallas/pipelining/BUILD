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
    name = "internal",
    srcs = ["internal.py"],
    deps = [
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:state_types",
    ],
)

pytype_strict_library(
    name = "schedule_api",
    srcs = ["schedule_api.py"],
    deps = [
        ":internal",
        ":schedulers",
        "//olympus",
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:partial_eval",
        "//olympus/_src:state_types",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "schedulers",
    srcs = ["schedulers.py"],
    deps = [
        ":internal",
        "//olympus",
        "//olympus/_src:core",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "pipeline_test_util",
    testonly = True,
    srcs = ["pipeline_test_util.py"],
    deps = [
        "//olympus/_src:debugging",
        "//olympus/_src/pallas/pipelining:internal",
        "//olympus/_src/pallas/pipelining:schedulers",
    ] + py_deps([
        "absl/testing",
    ]),
)
