# Copyright 2024 The OLYMPUS Authors.
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
    "if_cuda_is_configured",
    "olympus_generate_backend_suites",
    "olympus_multiplatform_test",
    "olympus_py_test",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

test_suite(
    name = "gpu_mlir_deviceless_tests",
    tags = ["gpu_mlir_deviceless_test"],
)

olympus_generate_backend_suites()

olympus_multiplatform_test(
    name = "gpu_test",
    srcs = ["gpu_test.py"],
    config_tags_overrides = {
        # TODO(b/448760413): Re-enable once llvm fixed.
        "gpu_b200": {"notap": True},
    },
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_b200",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    shard_count = 8,
    tags = [
        "noasan",  # Times out.
    ],
    deps = if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
        "//olympus/experimental:mosaic_gpu_test_util",
    ]) + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

olympus_multiplatform_test(
    name = "gpu_smith_test",
    srcs = ["gpu_smith_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_b200",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = [
        "noasan",  # ASAN is unsupported.
        "notap",  # TODO(b/477964663): Broken by Blacksmit 2.10.0 upgrade.
    ],
    deps = if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
    ]) + py_deps([
        "absl/testing",
        "numpy",
        "smith",
    ]),
)

olympus_multiplatform_test(
    name = "gpu_test_multidevice",
    srcs = ["gpu_test_multidevice.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100x2",
        "gpu_b200x2",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0"},
    tags = ["multiaccelerator"],
    deps = if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
    ]) + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

olympus_multiplatform_test(
    name = "gpu_test_distributed",
    srcs = ["gpu_test_distributed.py"],
    args = [
        "--num_processes=2",
        "--gpus_per_process=1",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_h100x2",
        "gpu_b200x2",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=0 --xla_gpu_experimental_enable_nvshmem=true"},
    tags = ["multiaccelerator"],
    deps = [
        "//olympus/_src:test_multiprocess",
    ] + if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
    ]) + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

olympus_py_test(
    name = "gpu_dialect_test",
    srcs = ["gpu_dialect_test.py"],
    tags = ["gpu_mlir_deviceless_test"],
    deps = [
        "//olympus",
        "//olympus/_src:test_util",
    ] + if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
    ]) + py_deps("absl/testing"),
)

olympus_py_test(
    name = "gpu_constraints_test",
    srcs = ["gpu_constraints_test.py"],
    deps = [
        "//olympus",
        "//olympus/_src:test_util",
    ] + if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
    ]) + py_deps("absl/testing"),
)

olympus_py_test(
    name = "gpu_layout_inference_test",
    size = "small",
    srcs = ["gpu_layout_inference_test.py"],
    tags = ["gpu_mlir_deviceless_test"],
    deps = [
        "//olympus",
        "//olympus/_src:test_util",
    ] + if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
        "//olympus/experimental:mosaic_gpu_test_util",
    ]) + py_deps("absl/testing"),
)

olympus_multiplatform_test(
    name = "matmul_test",
    srcs = ["matmul_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_b200",
    ],
    shard_count = 5,
    tags = [
        "noasan",
    ],
    deps = if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
        "//olympus/experimental/mosaic/gpu/examples:matmul",
        "//olympus/experimental/mosaic/gpu/examples:matmul_blackwell",
    ]) + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

olympus_multiplatform_test(
    name = "flash_attention",
    srcs = ["//olympus/experimental/mosaic/gpu/examples:flash_attention.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_b200",
    ],
    main = "//olympus/experimental/mosaic/gpu/examples:flash_attention.py",
    tags = [
        "manual",
        "noasan",
        "notap",
    ],
    deps = if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
    ]) + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

olympus_multiplatform_test(
    name = "flash_attention_test",
    srcs = ["flash_attention_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_b200",
    ],
    shard_count = 8,
    tags = [
        "noasan",  # Remove the tag once the CUPTI issue is fixed.
    ],
    deps = if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
        "//olympus/experimental/mosaic/gpu/examples:flash_attention",
    ]) + py_deps("absl/testing"),
)

olympus_multiplatform_test(
    name = "profiler_cupti_test",
    srcs = ["profiler_cupti_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_b200",
    ],
    tags = [
        "noasan",  # CUPTI leaks memory
        "nomsan",
    ],
    deps = if_cuda_is_configured([
        "//olympus/experimental:mosaic_gpu",
    ]) + py_deps("absl/testing"),
)
