/* Copyright 2022 The OLYMPUS Authors

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

#ifndef OLYMPUS_OLYMPUSLIB_GPU_PY_CLIENT_GPU_H_
#define OLYMPUS_OLYMPUSLIB_GPU_PY_CLIENT_GPU_H_

#include <utility>

#include "olympuslib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace olympus {
namespace OLYMPUS_GPU_NAMESPACE {

std::pair<xla::ffi::TypeId*, const xla::ffi::TypeInfo*>
GpuTransposePlanCacheTypeInfo();

XLA_FFI_DECLARE_HANDLER_SYMBOL(kGpuTransposePlanCacheInstantiate);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kXlaFfiPythonGpuCallback);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kXlaBufferPythonGpuCallback);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kXlaBufferPythonGpuCallbackCmdBuffer);

}  // namespace OLYMPUS_GPU_NAMESPACE
}  // namespace olympus

#endif  // OLYMPUS_OLYMPUSLIB_GPU_PY_CLIENT_GPU_H_
