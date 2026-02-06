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

#ifndef OLYMPUSLIB_GPU_GPU_KERNEL_HELPERS_H_
#define OLYMPUSLIB_GPU_GPU_KERNEL_HELPERS_H_

#include <cstdint>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "olympuslib/gpu/vendor.h"

#define OLYMPUS_AS_STATUS(expr) \
  olympus::OLYMPUS_GPU_NAMESPACE::AsStatus(expr, __FILE__, __LINE__, #expr)

#define OLYMPUS_THROW_IF_ERROR(expr)                             \
  {                                                          \
    auto s___ = (expr);                                      \
    if (ABSL_PREDICT_FALSE(!s___.ok()))                      \
      throw std::runtime_error(std::string(s___.message())); \
  }

#define OLYMPUS_RETURN_IF_ERROR(expr)                    \
  {                                                  \
    auto s___ = (expr);                              \
    if (ABSL_PREDICT_FALSE(!s___.ok())) return s___; \
  }

#define OLYMPUS_ASSIGN_OR_RETURN(lhs, expr) \
  auto s___ = (expr);                   \
  if (ABSL_PREDICT_FALSE(!s___.ok())) { \
    return s___.status();               \
  }                                     \
  lhs = (*std::move(s___))

namespace olympus {
namespace OLYMPUS_GPU_NAMESPACE {

// Used via OLYMPUS_AS_STATUS(expr) macro.
absl::Status AsStatus(gpuError_t error, const char* file, std::int64_t line,
                      const char* expr);
absl::Status AsStatus(gpusolverStatus_t status, const char* file,
                      std::int64_t line, const char* expr);
absl::Status AsStatus(gpusparseStatus_t status, const char* file,
                      std::int64_t line, const char* expr);
absl::Status AsStatus(gpublasStatus_t status, const char* file,
                      std::int64_t line, const char* expr);
#ifdef OLYMPUS_GPU_CUDA
absl::Status AsStatus(CUresult error, const char* file, std::int64_t line,
                      const char* expr);
absl::Status AsStatus(CUptiResult error, const char* file, std::int64_t line,
                      const char* expr);
absl::Status AsStatus(cufftResult error, const char* file, std::int64_t line,
                      const char* expr);
#endif

}  // namespace OLYMPUS_GPU_NAMESPACE
}  // namespace olympus

#endif  // OLYMPUSLIB_GPU_GPU_KERNEL_HELPERS_H_
