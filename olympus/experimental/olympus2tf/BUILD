# Copyright 2018 The OLYMPUS Authors.
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
    "olympus2tf_deps",
    "olympus_visibility",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

py_library(
    name = "olympus2tf",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [":olympus2tf_internal"],
)

py_library(
    name = "olympus2tf_internal",
    srcs = [
        "call_tf.py",
        "olympus2tf.py",
    ],
    # TODO: b/255503696: enable pytype
    tags = ["pytype_unchecked_annotations"],
    visibility = olympus_visibility("olympus2tf_internal"),
    deps = [
        "//olympus",
    ] + py_deps("numpy") + py_deps("tensorflow_core") + olympus2tf_deps,
)
