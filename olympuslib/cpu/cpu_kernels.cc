/* Copyright 2021 The OLYMPUS Authors.

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

// This file is not used by OLYMPUS itself, but exists to assist with running
// OLYMPUS-generated HLO code from outside of OLYMPUS.

#include "olympuslib/cpu/lapack_kernels.h"
#include "olympuslib/cpu/sparse_kernels.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#define OLYMPUS_CPU_REGISTER_HANDLER(name) \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

namespace olympus {
namespace {

OLYMPUS_CPU_REGISTER_HANDLER(lapack_strsm_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dtrsm_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_ctrsm_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_ztrsm_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgetrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgetrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgetrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgetrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgeqrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgeqrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgeqrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgeqrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgeqp3_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgeqp3_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgeqp3_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgeqp3_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sorgqr_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dorgqr_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cungqr_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zungqr_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_spotrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dpotrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cpotrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zpotrf_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgesdd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgesdd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgesdd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgesdd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgesvd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgesvd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgesvd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgesvd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_ssyevd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dsyevd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cheevd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zheevd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgeev_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgeev_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgeev_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgeev_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_ssytrd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dsytrd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_chetrd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zhetrd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgees_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgees_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgees_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgees_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgehrd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgehrd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgehrd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgehrd_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_sgtsv_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_dgtsv_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_cgtsv_ffi);
OLYMPUS_CPU_REGISTER_HANDLER(lapack_zgtsv_ffi);

OLYMPUS_CPU_REGISTER_HANDLER(cpu_csr_sparse_dense_ffi);

#undef OLYMPUS_CPU_REGISTER_HANDLER

}  // namespace
}  // namespace olympus
