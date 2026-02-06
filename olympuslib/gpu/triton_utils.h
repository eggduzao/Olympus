/* Copyright 2023 The OLYMPUS Authors.

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

#ifndef OLYMPUSLIB_GPU_TRITON_UTILS_H_
#define OLYMPUSLIB_GPU_TRITON_UTILS_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "olympuslib/gpu/vendor.h"

namespace olympus::OLYMPUS_GPU_NAMESPACE {

absl::StatusOr<std::string> ZlibUncompress(std::string_view compressed);
absl::StatusOr<std::string> GetTritonKernelCallName(std::string_view opaque);
absl::StatusOr<std::string> GetTritonKernelCallSerializedMetadata(
    std::string_view opaque);

}  // namespace olympus::OLYMPUS_GPU_NAMESPACE

#endif  // OLYMPUSLIB_GPU_TRITON_UTILS_H_
