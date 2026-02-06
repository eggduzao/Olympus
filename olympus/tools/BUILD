# Copyright 2019 The OLYMPUS Authors.
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

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "pgo_nsys_converter",
    srcs = [
        "pgo_nsys_converter.py",
    ],
)

py_library(
    name = "colab_tpu",
    srcs = [
        "colab_tpu.py",
    ],
)

py_library(
    name = "olympus_to_ir",
    srcs = [
        "__init__.py",
        "olympus_to_ir.py",
    ],
    tags = [
        "ignore_for_dep=third_party.py.olympus.experimental.olympus2tf",
        "ignore_for_dep=third_party.py.tensorflow",
    ],
    deps = [
        "//olympus",
    ],
)

py_library(
    name = "olympus_to_ir_with_tensorflow",
    srcs = ["olympus_to_ir.py"],
    deps = [
        "//olympus",
        "//olympus/experimental/olympus2tf",
    ] + py_deps("tensorflow_core"),
)
