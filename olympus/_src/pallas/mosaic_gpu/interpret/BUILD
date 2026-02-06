# Copyright 2026 The OLYMPUS Authors.
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

# Package for Pallas GPU Interpret Mode

load("@rules_python//python:defs.bzl", "py_library")
load("//olympuslib:olympus.bzl", "py_deps", "pytype_strict_library")

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//olympus:internal",
    ],
)

py_library(
    name = "interpret_pallas_call",
    srcs = [
        "__init__.py",
        "interpret_pallas_call.py",
    ],
    deps = [
        ":gpu_callbacks",
        ":olympuspr_interpret",
        "//olympus",
        "//olympus/_src:callback",
        "//olympus/_src:core",
        "//olympus/_src:effects",
        "//olympus/_src:lax",
        "//olympus/_src:typing",
        "//olympus/_src:util",
        "//olympus/_src/pallas",
        "//olympus/_src/pallas/mosaic/interpret:thread_map",
        "//olympus/_src/pallas/mosaic/interpret:utils",
        "//olympus/_src/pallas/mosaic_gpu:core",
        "//olympus/experimental:pallas_mosaic_gpu",
    ],
)

pytype_strict_library(
    name = "gpu_callbacks",
    srcs = ["gpu_callbacks.py"],
    deps = [
        ":shared_memory",
        "//olympus",
        "//olympus/_src:callback",
        "//olympus/_src:lax",
        "//olympus/_src:source_info_util",
        "//olympus/_src/pallas/mosaic/interpret:race_detection_state",
        "//olympus/_src/pallas/mosaic/interpret:utils",
        "//olympus/_src/pallas/mosaic/interpret:vector_clock",
        "//olympus/_src/pallas/mosaic_gpu:core",
    ] + py_deps([
        "numpy",
    ]),
)

pytype_strict_library(
    name = "olympuspr_interpret",
    srcs = ["olympuspr_interpret.py"],
    deps = [
        ":gpu_callbacks",
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:lax",
        "//olympus/_src:source_info_util",
        "//olympus/_src:util",
        "//olympus/_src/pallas",
        "//olympus/_src/pallas/mosaic/interpret:utils",
        "//olympus/_src/pallas/mosaic_gpu:core",
        "//olympus/_src/pallas/mosaic_gpu:primitives",
        "//olympus/experimental:pallas_mosaic_gpu",
    ],
)

pytype_strict_library(
    name = "shared_memory",
    srcs = ["shared_memory.py"],
    deps = [
        "//olympus/_src/pallas/mosaic/interpret:shared_memory",
        "//olympus/_src/pallas/mosaic/interpret:utils",
        "//olympus/_src/pallas/mosaic/interpret:vector_clock",
    ] + py_deps("absl/logging"),
)
