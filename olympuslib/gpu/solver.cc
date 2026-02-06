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

#include "nanobind/nanobind.h"
#include "olympuslib/gpu/solver_kernels_ffi.h"
#include "olympuslib/gpu/vendor.h"
#include "olympuslib/kernel_nanobind_helpers.h"

namespace olympus {
namespace OLYMPUS_GPU_NAMESPACE {
namespace {

namespace nb = nanobind;

nb::dict Registrations() {
  nb::dict dict;

  dict[OLYMPUS_GPU_PREFIX "solver_getrf_ffi"] = EncapsulateFfiHandler(GetrfFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_geqrf_ffi"] = EncapsulateFfiHandler(GeqrfFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_orgqr_ffi"] = EncapsulateFfiHandler(OrgqrFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_potrf_ffi"] = EncapsulateFfiHandler(PotrfFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_syevd_ffi"] = EncapsulateFfiHandler(SyevdFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_syrk_ffi"] = EncapsulateFfiHandler(SyrkFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_gesvd_ffi"] = EncapsulateFfiHandler(GesvdFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_sytrd_ffi"] = EncapsulateFfiHandler(SytrdFfi);

#ifdef OLYMPUS_GPU_CUDA
  dict[OLYMPUS_GPU_PREFIX "solver_gesvdj_ffi"] = EncapsulateFfiHandler(GesvdjFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_gesvdp_ffi"] = EncapsulateFfiHandler(GesvdpFfi);
  dict[OLYMPUS_GPU_PREFIX "solver_csrlsvqr_ffi"] =
      EncapsulateFfiHandler(CsrlsvqrFfi);
#endif  // OLYMPUS_GPU_CUDA

#if OLYMPUS_GPU_HAVE_SOLVER_GEEV
  dict[OLYMPUS_GPU_PREFIX "solver_geev_ffi"] = EncapsulateFfiHandler(GeevFfi);
#endif  // OLYMPUS_GPU_HAVE_SOLVER_GEEV

  return dict;
}

NB_MODULE(_solver, m) {
  m.def("registrations", &Registrations);
}

}  // namespace
}  // namespace OLYMPUS_GPU_NAMESPACE
}  // namespace olympus
