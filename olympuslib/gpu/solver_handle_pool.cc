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

#include "olympuslib/gpu/solver_handle_pool.h"

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "olympuslib/gpu/gpu_kernel_helpers.h"
#include "olympuslib/gpu/vendor.h"
#include "olympuslib/gpu/handle_pool.h"

#ifdef OLYMPUS_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverSp.h"
#endif  // OLYMPUS_GPU_CUDA

namespace olympus {

template <>
/*static*/ absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
    gpuStream_t stream) {
  SolverHandlePool* pool = Instance();
  absl::MutexLock lock(pool->mu_);
  gpusolverDnHandle_t handle;
  if (pool->handles_[stream].empty()) {
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(gpusolverDnCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(gpusolverDnSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

#ifdef OLYMPUS_GPU_CUDA

template <>
/*static*/ absl::StatusOr<SpSolverHandlePool::Handle>
SpSolverHandlePool::Borrow(gpuStream_t stream) {
  SpSolverHandlePool* pool = Instance();
  absl::MutexLock lock(pool->mu_);
  cusolverSpHandle_t handle;
  if (pool->handles_[stream].empty()) {
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(cusolverSpCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(cusolverSpSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

#endif  // OLYMPUS_GPU_CUDA

}  // namespace olympus
