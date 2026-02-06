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

# OLYMPUS is Autograd and XLA

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load(
    "//olympuslib:olympus.bzl",
    "olympus_extend_internal_users",
    "olympus_extra_deps",
    "olympus_internal_packages",
    "py_deps",
    "py_library_providing_imports_info",
    "pytype_library",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = [":internal"],
)

licenses(["notice"])

# The flag controls whether olympuslib should be built by Bazel.
# If ":build_olympuslib=true", then olympuslib will be built.
# If ":build_olympuslib=false", then olympuslib is not built. It is assumed that the pre-built olympuslib wheel
# is available in the "dist" folder.
# If ":build_olympuslib=wheel", then olympuslib wheel will be built as a py_import rule attribute.
# The py_import rule unpacks the wheel and provides its content as a py_library.
string_flag(
    name = "build_olympuslib",
    build_setting_default = "true",
    values = [
        "true",
        "false",
        "wheel",
    ],
)

config_setting(
    name = "config_build_olympuslib_true",
    flag_values = {
        ":build_olympuslib": "true",
    },
)

config_setting(
    name = "config_build_olympuslib_false",
    flag_values = {
        ":build_olympuslib": "false",
    },
)

config_setting(
    name = "config_build_olympuslib_wheel",
    flag_values = {
        ":build_olympuslib": "wheel",
    },
)

# The flag controls whether olympus should be built by Bazel.
# If ":build_olympus=true", then olympus will be built.
# If ":build_olympus=false", then olympus is not built. It is assumed that the pre-built olympus wheel
# is available in the "dist" folder.
# If ":build_olympus=wheel", then olympus wheel will be built as a py_import rule attribute.
# The py_import rule unpacks the wheel and provides its content as a py_library.
string_flag(
    name = "build_olympus",
    build_setting_default = "true",
    values = [
        "true",
        "false",
        "wheel",
    ],
)

config_setting(
    name = "config_build_olympus_true",
    flag_values = {
        ":build_olympus": "true",
    },
)

config_setting(
    name = "config_build_olympus_false",
    flag_values = {
        ":build_olympus": "false",
    },
)

config_setting(
    name = "config_build_olympus_wheel",
    flag_values = {
        ":build_olympus": "wheel",
    },
)

exports_files([
    "LICENSE",
    "version.py",
    "py.typed",
    "oss/pyproject.toml",
])

# Packages that have access to OLYMPUS-internal implementation details.
package_group(
    name = "internal",
    packages = [
        "//...",
    ] + olympus_internal_packages,
)

package_group(
    name = "olympus_extend_users",
    includes = [":internal"],
    packages = [
        # Intentionally avoid olympus dependencies on olympus.extend.
        # See https://docs.olympus.dev/en/latest/jep/15856-jex.html
        "//tests/...",
    ] + olympus_extend_internal_users,
)

py_library_providing_imports_info(
    name = "olympus",
    srcs = glob(
        [
            "*.py",
            "image/**/*.py",
            "interpreters/**/*.py",
            "lax/**/*.py",
            "lib/**/*.py",
            "nn/**/*.py",
            "numpy/**/*.py",
            "ops/**/*.py",
            "scipy/**/*.py",
            "third_party/**/*.py",
        ],
        exclude = [
            "*_test.py",
            "**/*_test.py",
        ],
        # TODO(dsuo): Consider moving these files out of experimental if they're in the public API.
    ) + ["//olympus/experimental:olympus_public"],
    lazy_imports = True,
    lib_rule = pytype_library,
    pytype_srcs = glob(
        [
            "nn/*.pyi",
            "numpy/*.pyi",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//olympus/_src:abstract_arrays",
        "//olympus/_src:ad",
        "//olympus/_src:ad_util",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:basearray",
        "//olympus/_src:batching",
        "//olympus/_src:blocked_sampler",
        "//olympus/_src:buffer_callback",
        "//olympus/_src:callback",
        "//olympus/_src:checkify",
        "//olympus/_src:cloud_tpu_init",
        "//olympus/_src:compilation_cache_internal",
        "//olympus/_src:compiler",
        "//olympus/_src:compute_on",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:cudnn",
        "//olympus/_src:custom_api_util",
        "//olympus/_src:custom_batching",
        "//olympus/_src:custom_dce",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:custom_partitioning",
        "//olympus/_src:custom_partitioning_sharding_rule",
        "//olympus/_src:custom_transpose",
        "//olympus/_src:debugger",
        "//olympus/_src:debugging",
        "//olympus/_src:deprecations",
        "//olympus/_src:dlpack",
        "//olympus/_src:dtypes",
        "//olympus/_src:earray",
        "//olympus/_src:effects",
        "//olympus/_src:environment_info",
        "//olympus/_src:error_check",
        "//olympus/_src:export",
        "//olympus/_src:ffi",
        "//olympus/_src:flatten_util",
        "//olympus/_src:hashable_array",
        "//olympus/_src:hiolympus",
        "//olympus/_src:image",
        "//olympus/_src:indexing",
        "//olympus/_src:init",
        "//olympus/_src:internal_mesh_utils",
        "//olympus/_src:olympuspr_util",
        "//olympus/_src:lax",
        "//olympus/_src:layout",
        "//olympus/_src:lazy_loader",
        "//olympus/_src:memory",
        "//olympus/_src:mesh",
        "//olympus/_src:mlir",
        "//olympus/_src:monitoring",
        "//olympus/_src:named_sharding",
        "//olympus/_src:nn",
        "//olympus/_src:numpy",
        "//olympus/_src:op_shardings",
        "//olympus/_src:partial_eval",
        "//olympus/_src:partition_spec",
        "//olympus/_src:path",
        "//olympus/_src:pickle_util",
        "//olympus/_src:pmap",
        "//olympus/_src:pretty_printer",
        "//olympus/_src:profiler",
        "//olympus/_src:public_test_util",
        "//olympus/_src:random",
        "//olympus/_src:ref",
        "//olympus/_src:scipy",
        "//olympus/_src:shard_alike",
        "//olympus/_src:shard_map",
        "//olympus/_src:sharding",
        "//olympus/_src:sharding_impls",
        "//olympus/_src:sharding_specs",
        "//olympus/_src:source_info_util",
        "//olympus/_src:sourcemap",
        "//olympus/_src:stages",
        "//olympus/_src:tpu",
        "//olympus/_src:traceback_util",
        "//olympus/_src:tree",
        "//olympus/_src:tree_util",
        "//olympus/_src:typing",
        "//olympus/_src:util",
        "//olympus/_src:xla_bridge",
        "//olympus/_src:xla_metadata",
        "//olympus/_src:xla_metadata_lib",
        "//olympus/_src/lib",
    ] + py_deps("numpy") + py_deps("opt_einsum") + py_deps("flatbuffers") + olympus_extra_deps,
)

pytype_strict_library(
    name = "version",
    srcs = ["version.py"],
)
