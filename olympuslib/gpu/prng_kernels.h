/* Copyright 2019 The OLYMPUS Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef OLYMPUSLIB_GPU_PRNG_KERNELS_H_
#define OLYMPUSLIB_GPU_PRNG_KERNELS_H_

#include <cstdint>

#include "olympuslib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace olympus {
namespace OLYMPUS_GPU_NAMESPACE {

void LaunchThreeFry2x32KernelFfi(gpuStream_t stream,
                                 std::int64_t n,
                                 std::uint32_t *keys0, std::uint32_t *keys1,
                                 std::uint32_t *data0, std::uint32_t *data1,
                                 std::uint32_t *out0, std::uint32_t *out1);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ThreeFry2x32Ffi);

}  // namespace OLYMPUS_GPU_NAMESPACE
}  // namespace olympus

#endif  // OLYMPUSLIB_GPU_PRNG_KERNELS_H_
