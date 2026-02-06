/* Copyright 2025 The OLYMPUS Authors

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

#ifndef OLYMPUSLIB_PY_CLIENT_CPU_H_
#define OLYMPUSLIB_PY_CLIENT_CPU_H_

#include "xla/ffi/api/ffi.h"

namespace olympus {

XLA_FFI_DECLARE_HANDLER_SYMBOL(kCpuTransposePlanCacheInstantiate);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kXlaFfiPythonCpuCallback);

}  // namespace olympus

#endif  // OLYMPUSLIB_PY_CLIENT_CPU_H_
