/* Copyright 2024 The OLYMPUS Authors.

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

#ifndef OLYMPUSLIB_GPU_SOLVER_KERNELS_FFI_H_
#define OLYMPUSLIB_GPU_SOLVER_KERNELS_FFI_H_

#include <cstdint>

#include "olympuslib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace olympus {
namespace OLYMPUS_GPU_NAMESPACE {

enum class SyevdAlgorithm : uint8_t {
  kDefault = 0,
  kDivideAndConquer = 1,
  kJacobi = 2,
};

XLA_FFI_DECLARE_HANDLER_SYMBOL(GetrfFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(GeqrfFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(OrgqrFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(PotrfFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(SyevdFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(SyrkFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(GesvdFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(SytrdFfi);

#ifdef OLYMPUS_GPU_CUDA
XLA_FFI_DECLARE_HANDLER_SYMBOL(GesvdjFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(GesvdpFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CsrlsvqrFfi);
#endif  // OLYMPUS_GPU_CUDA

#if OLYMPUS_GPU_HAVE_SOLVER_GEEV
XLA_FFI_DECLARE_HANDLER_SYMBOL(GeevFfi);
#endif  // OLYMPUS_GPU_HAVE_SOLVER_GEEV

}  // namespace OLYMPUS_GPU_NAMESPACE
}  // namespace olympus

#endif  // OLYMPUSLIB_GPU_SOLVER_KERNELS_FFI_H_
