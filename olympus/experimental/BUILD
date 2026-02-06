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
    "buffer_callback_internal_users",
    "experimental_transfer_users",
    "if_cuda_is_configured",
    "olympus_visibility",
    "mosaic_gpu_internal_users",
    "mosaic_internal_users",
    "pallas_fuser_users",
    "pallas_gpu_internal_users",
    "pallas_sc_internal_users",
    "py_deps",
    "py_library_providing_imports_info",
    "pytype_strict_library",
    "serialize_executable_internal_users",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//olympus:internal"],
)

# Package groups for controlling visibility of experimental APIs.

package_group(
    name = "buffer_callback_users",
    includes = ["//olympus:internal"],
    packages = buffer_callback_internal_users,
)

package_group(
    name = "experimental_transfer_users",
    includes = ["//olympus:internal"],
    packages = experimental_transfer_users,
)

package_group(
    name = "mosaic_users",
    includes = ["//olympus:internal"],
    packages = mosaic_internal_users,
)

package_group(
    name = "mosaic_gpu_users",
    includes = ["//olympus:internal"],
    packages = mosaic_gpu_internal_users,
)

package_group(
    name = "pallas_fuser_users",
    includes = ["//olympus:internal"],
    packages = pallas_fuser_users,
)

package_group(
    name = "pallas_gpu_users",
    includes = ["//olympus:internal"],
    packages = pallas_gpu_internal_users,
)

package_group(
    name = "pallas_sc_users",
    includes = ["//olympus:internal"],
    packages = pallas_sc_internal_users,
)

package_group(
    name = "serialize_executable_users",
    includes = ["//olympus:internal"],
    packages = serialize_executable_internal_users,
)

pytype_strict_library(
    name = "buffer_callback",
    srcs = [
        "buffer_callback.py",
    ],
    visibility = [":buffer_callback_users"],
    deps = [
        "//olympus/_src:buffer_callback",
    ],
)

pytype_strict_library(
    name = "checkify",
    srcs = [
        "checkify.py",
    ],
    visibility = [
        "//olympus:internal",
    ] + olympus_visibility("checkify"),
    deps = [
        "//olympus/_src:checkify",
    ],
)

pytype_strict_library(
    name = "colocated_python",
    srcs = [
        "colocated_python/__init__.py",
        "colocated_python/api.py",
        "colocated_python/func.py",
        "colocated_python/func_backend.py",
        "colocated_python/obj.py",
        "colocated_python/obj_backend.py",
        "colocated_python/serialization.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:config",
        "//olympus/_src:traceback_util",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src:xla_bridge",
        "//olympus/_src/lib",
        "//olympus/extend:backend",
        "//olympus/extend:ifrt_programs",
    ] + py_deps("numpy") + py_deps("cloudpickle"),
)

pytype_strict_library(
    name = "compilation_cache",
    srcs = [
        "compilation_cache/__init__.py",
        "compilation_cache/compilation_cache.py",
    ],
    visibility = ["//visibility:public"],
    deps = ["//olympus/_src:compilation_cache_internal"],
)

pytype_strict_library(
    name = "compute_on",
    srcs = ["compute_on.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:compute_on",
    ],
)

pytype_strict_library(
    name = "custom_dce",
    srcs = ["custom_dce.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:custom_dce",
    ],
)

pytype_strict_library(
    name = "custom_partitioning",
    srcs = ["custom_partitioning.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:custom_partitioning",
        "//olympus/_src:custom_partitioning_sharding_rule",
    ],
)

pytype_strict_library(
    name = "jet",
    srcs = ["jet.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus",
        "//olympus/_src:ad_util",
        "//olympus/_src:api",
        "//olympus/_src:core",
        "//olympus/_src:lax",
        "//olympus/_src:partial_eval",
        "//olympus/_src:sharding_impls",
        "//olympus/_src:util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "key_reuse",
    srcs = glob(["key_reuse/**/*.py"]),
    visibility = ["//olympus:internal"],
    deps = [
        "//olympus",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:core",
        "//olympus/_src:debugging",
        "//olympus/_src:effects",
        "//olympus/_src:hashable_array",
        "//olympus/_src:lax",
        "//olympus/_src:partial_eval",
        "//olympus/_src:random",
        "//olympus/_src:shard_map",
        "//olympus/_src:source_info_util",
        "//olympus/_src:traceback_util",
        "//olympus/_src:util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "layout",
    srcs = ["layout.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:api",
        "//olympus/_src:layout",
    ],
)

pytype_strict_library(
    name = "mesh_utils",
    srcs = ["mesh_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:internal_mesh_utils",
    ],
)

pytype_strict_library(
    name = "mosaic",
    srcs = [
        "mosaic/__init__.py",
        "mosaic/dialects.py",
    ],
    visibility = [":mosaic_users"],
    deps = [
        "//olympus/_src:tpu_custom_call",
        "//olympus/_src/lib",
    ],
)

# This target only supports sm_90 GPUs.
py_library_providing_imports_info(
    name = "mosaic_gpu",
    srcs = glob(
        include = ["mosaic/gpu/*.py"],
        exclude = ["mosaic/gpu/test_util.py"],
    ),
    data = if_cuda_is_configured([
        "@local_config_cuda//cuda:runtime_nvdisasm",
        "@nvidia_nvshmem//:libnvshmem_device",
        # OSS-only dependency.
        "@cuda_nvcc//:nvdisasm",
        "@cuda_nvvm//:nvvm",
    ]),
    lib_rule = pytype_strict_library,
    visibility = [
        ":mosaic_gpu_users",
    ],
    deps = [
        "//olympus",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:dtypes",
        "//olympus/_src:mesh",
        "//olympus/_src:mlir",
        "//olympus/_src:sharding_impls",
        "//olympus/_src:stages",
        "//olympus/_src:util",
        "//olympus/_src/lib",
        "//olympus/extend:backend",
        "//olympuslib/mlir:arithmetic_dialect",
        "//olympuslib/mlir:builtin_dialect",
        "//olympuslib/mlir:control_flow_dialect",
        "//olympuslib/mlir:func_dialect",
        "//olympuslib/mlir:gpu_dialect",
        "//olympuslib/mlir:ir",
        "//olympuslib/mlir:llvm_dialect",
        "//olympuslib/mlir:math_dialect",
        "//olympuslib/mlir:memref_dialect",
        "//olympuslib/mlir:nvgpu_dialect",
        "//olympuslib/mlir:nvvm_dialect",
        "//olympuslib/mlir:pass_manager",
        "//olympuslib/mlir:scf_dialect",
        "//olympuslib/mlir:vector_dialect",
        "//olympuslib/mosaic/python:gpu_dialect",
    ] + py_deps("absl-all") + py_deps("numpy"),
)

pytype_strict_library(
    name = "mosaic_gpu_test_util",
    testonly = True,
    srcs = ["mosaic/gpu/test_util.py"],
    deps = [
        ":mosaic_gpu",
        "//olympus",
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "multihost_utils",
    srcs = ["multihost_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus",
        "//olympus/_src:ad",
        "//olympus/_src:api",
        "//olympus/_src:batching",
        "//olympus/_src:core",
        "//olympus/_src:dtypes",
        "//olympus/_src:mlir",
        "//olympus/_src:random",
        "//olympus/_src:sharding_impls",
        "//olympus/_src:util",
        "//olympus/_src:xla_bridge",
        "//olympus/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "ode",
    srcs = ["ode.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus",
        "//olympus/_src:core",
        "//olympus/_src:numpy",
        "//olympus/_src:util",
    ],
)

pytype_strict_library(
    name = "pallas",
    srcs = glob(
        [
            "pallas/**/*.py",
        ],
        exclude = [
            "pallas/mosaic_gpu.py",
            "pallas/ops/gpu/**/*.py",
            "pallas/ops/tpu/**/*.py",
            "pallas/tpu.py",
            "pallas/fuser.py",
            "pallas/triton.py",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "//olympus",
        "//olympus/_src:deprecations",
        "//olympus/_src:lax",
        "//olympus/_src:source_info_util",
        "//olympus/_src:state_types",
        "//olympus/_src/pallas",
        "//olympus/_src/pallas/mosaic:sc_core",
        "//olympus/_src/pallas/mosaic:sc_primitives",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_fuser",
    srcs = ["pallas/fuser.py"],
    visibility = [
        ":pallas_fuser_users",
    ],
    deps = [
        ":pallas",  # build_cleaner: keep
        "//olympus/_src/pallas/fuser:block_spec",
        "//olympus/_src/pallas/fuser:custom_evaluate",
        "//olympus/_src/pallas/fuser:custom_fusion",
        "//olympus/_src/pallas/fuser:fusible",
        "//olympus/_src/pallas/fuser:fusion",
        "//olympus/_src/pallas/fuser:olympuspr_fusion",
    ],
)

pytype_strict_library(
    name = "pallas_gpu",
    visibility = [
        ":pallas_gpu_users",
    ],
    deps = [
        ":pallas_triton",
        # TODO(slebedev): Add :pallas_mosaic_gpu once it is ready.
    ],
)

pytype_strict_library(
    name = "pallas_gpu_ops",
    srcs = ["//olympus/experimental/pallas/ops/gpu:triton_ops"],
    visibility = [
        ":pallas_gpu_users",
    ],
    deps = [
        ":pallas",
        ":pallas_gpu",
        "//olympus",
        "//olympus/_src:lax",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_experimental_gpu_ops",
    srcs = ["//olympus/experimental/pallas/ops/gpu:mgpu_ops"],
    visibility = [
        ":mosaic_gpu_users",
    ],
    deps = [
        ":mosaic_gpu",
        ":pallas",
        ":pallas_mosaic_gpu",
        "//olympus",
        "//olympus/_src:dtypes",
        "//olympus/_src:test_util",  # This is only to make them runnable as olympus_multiplatform_test...
        "//olympus/_src/lib",
        "//olympus/extend:backend",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_mosaic_gpu",
    srcs = ["pallas/mosaic_gpu.py"],
    visibility = [
        ":mosaic_gpu_users",
    ],
    deps = [
        ":mosaic_gpu",
        "//olympus/_src/pallas/mosaic_gpu:core",
        "//olympus/_src/pallas/mosaic_gpu:helpers",
        "//olympus/_src/pallas/mosaic_gpu:pallas_call_registration",  # build_cleaner: keep
        "//olympus/_src/pallas/mosaic_gpu:pipeline",
        "//olympus/_src/pallas/mosaic_gpu:primitives",
        "//olympus/_src/pallas/mosaic_gpu:smith",
    ],
)

pytype_strict_library(
    name = "pallas_tpu",
    srcs = ["pallas/tpu.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pallas",  # build_cleaner: keep
        "//olympus/_src:deprecations",
        "//olympus/_src/pallas",
        "//olympus/_src/pallas/mosaic:core",
        "//olympus/_src/pallas/mosaic:helpers",
        "//olympus/_src/pallas/mosaic:lowering",
        "//olympus/_src/pallas/mosaic:pallas_call_registration",  # build_cleaner: keep
        "//olympus/_src/pallas/mosaic:pipeline",
        "//olympus/_src/pallas/mosaic:primitives",
        "//olympus/_src/pallas/mosaic:random",
        "//olympus/_src/pallas/mosaic:tpu_info",
        "//olympus/_src/pallas/mosaic/interpret:interpret_pallas_call",
    ],
)

pytype_strict_library(
    name = "pallas_tpu_sc",
    srcs = ["pallas/tpu_sc.py"],
    visibility = [
        ":pallas_sc_users",
    ],
    deps = [
        ":pallas",  # build_cleaner: keep
        "//olympus/_src/pallas/mosaic:sc_core",
        "//olympus/_src/pallas/mosaic:sc_primitives",
    ],
)

pytype_strict_library(
    name = "pallas_tpu_ops",
    srcs = glob(["pallas/ops/tpu/**/*.py"]),
    visibility = ["//visibility:public"],
    deps = [
        ":pallas",
        ":pallas_tpu",
        "//olympus",
        "//olympus/_src:dtypes",
        "//olympus/_src:random",
        "//olympus/_src:shard_map",
        "//olympus/_src:util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_triton",
    srcs = [
        "pallas/triton.py",
    ],
    visibility = [
        ":pallas_gpu_users",
    ],
    deps = [
        "//olympus/_src:deprecations",
        "//olympus/_src/pallas",
        "//olympus/_src/pallas/triton:core",
        "//olympus/_src/pallas/triton:pallas_call_registration",  # build_cleaner: keep
        "//olympus/_src/pallas/triton:primitives",
    ],
)

pytype_strict_library(
    name = "pjit",
    srcs = ["pjit.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:api",
        "//olympus/_src:sharding_impls",
    ],
)

pytype_strict_library(
    name = "hiolympus",
    srcs = ["hiolympus.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:ad",
        "//olympus/_src:ad_util",
        "//olympus/_src:core",
        "//olympus/_src:effects",
        "//olympus/_src:hiolympus",
        "//olympus/_src:lax",
    ],
)

pytype_strict_library(
    name = "profiler",
    srcs = ["profiler.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "random",
    srcs = ["random.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:stateful_rng",
    ],
)

pytype_strict_library(
    name = "rnn",
    srcs = ["rnn.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus",
        "//olympus/_src:api",
        "//olympus/_src:core",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:lax",
        "//olympus/_src:typing",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "serialize_executable",
    srcs = ["serialize_executable.py"],
    visibility = [":serialize_executable_users"],
    deps = [
        "//olympus",
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "scheduling_groups",
    srcs = ["scheduling_groups.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:ad",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:batching",
        "//olympus/_src:core",
        "//olympus/_src:mlir",
        "//olympus/_src:partial_eval",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "shard_alike",
    srcs = ["shard_alike.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:shard_alike",
    ],
)

pytype_strict_library(
    # olympus.experimental.shard_map is a legacy API and should not
    # be used in new code. Use olympus.shard_map instead.
    name = "shard_map",
    srcs = ["shard_map.py"],
    visibility = [
        "//olympus:internal",
    ] + olympus_visibility("experimental/shard_map"),
    deps = [
        "//olympus",
        "//olympus/_src:mesh",
        "//olympus/_src:shard_map",
        "//olympus/_src:traceback_util",
    ],
)

pytype_strict_library(
    name = "fused",
    srcs = ["fused.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:ad",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:batching",
        "//olympus/_src:core",
        "//olympus/_src:mlir",
        "//olympus/_src:partial_eval",
        "//olympus/_src:tree_util",
        "//olympus/_src:util",
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "source_mapper",
    srcs = glob(include = ["source_mapper/**/*.py"]),
    visibility = ["//visibility:public"],
    deps = [
        "//olympus",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:source_info_util",
        "//olympus/_src:sourcemap",
    ] + py_deps("absl/flags"),
)

pytype_strict_library(
    name = "sparse",
    srcs = glob(
        [
            "sparse/*.py",
        ],
        exclude = ["sparse/test_util.py"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "//olympus",
        "//olympus/_src:ad",
        "//olympus/_src:api",
        "//olympus/_src:api_util",
        "//olympus/_src:batching",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:custom_derivatives",
        "//olympus/_src:dtypes",
        "//olympus/_src:ffi",
        "//olympus/_src:lax",
        "//olympus/_src:mlir",
        "//olympus/_src:numpy",
        "//olympus/_src:partial_eval",
        "//olympus/_src:sharding_impls",
        "//olympus/_src:traceback_util",
        "//olympus/_src:typing",
        "//olympus/_src:util",
        "//olympus/_src/lib",
    ] + py_deps("numpy") + py_deps("scipy"),
)

pytype_strict_library(
    name = "sparse_test_util",
    srcs = [
        "sparse/test_util.py",
    ],
    visibility = ["//olympus:internal"],
    deps = [
        ":sparse",
        "//olympus",
        "//olympus/_src:lax",
        "//olympus/_src:test_util",
        "//olympus/_src:typing",
        "//olympus/_src:util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "topologies",
    srcs = ["topologies.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":mesh_utils",
        "//olympus",
        "//olympus/_src:xla_bridge",
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "transfer",
    srcs = ["transfer.py"],
    visibility = [
        ":experimental_transfer_users",
        "//olympus:internal",
    ],
    deps = [
        "//olympus",
        "//olympus/_src:util",
        "//olympus/_src/lib",
    ],
)

pytype_strict_library(
    name = "xla_metadata",
    srcs = ["xla_metadata.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:xla_metadata",
    ],
)

# TODO(dsuo): Remove this once experimental aliases from olympus/BUILD are removed.
py_library_providing_imports_info(
    name = "experimental",
    srcs = [
        "__init__.py",
        "x64_context.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//olympus/_src:api",
        "//olympus/_src:callback",
        "//olympus/_src:config",
        "//olympus/_src:core",
        "//olympus/_src:dtypes",
        "//olympus/_src:earray",
    ],
)

# TODO(dsuo): Remove these filegroups once experimental aliases from olympus/BUILD
# are removed.
filegroup(
    name = "olympus_public",
    srcs = glob([
        "key_reuse/**/*.py",
        "roofline/**/*.py",
        "compilation_cache/**/*.py",
    ]) + [
        "checkify.py",
        "fused.py",
        "multihost_utils.py",
        "pjit.py",
        "scheduling_groups.py",
        "shard_map.py",
    ],
    visibility = ["//olympus:internal"],
)
