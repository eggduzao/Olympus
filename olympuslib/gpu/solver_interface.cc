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

#include "olympuslib/gpu/solver_interface.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "olympuslib/gpu/gpu_kernel_helpers.h"
#include "olympuslib/gpu/vendor.h"

#ifdef OLYMPUS_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverSp.h"
#endif

namespace olympus {
namespace OLYMPUS_GPU_NAMESPACE {
namespace solver {

// LU decomposition: getrf

#define OLYMPUS_GPU_DEFINE_GETRF(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> GetrfBufferSize<Type>(gpusolverDnHandle_t handle, int m, \
                                            int n) {                           \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(                                         \
        Name##_bufferSize(handle, m, n, /*A=*/nullptr, m, &lwork)));           \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Getrf<Type>(gpusolverDnHandle_t handle, int m, int n, Type *a,  \
                           Type *workspace, int lwork, int *ipiv, int *info) { \
    return OLYMPUS_AS_STATUS(                                                      \
        Name(handle, m, n, a, m, workspace, lwork, ipiv, info));               \
  }

OLYMPUS_GPU_DEFINE_GETRF(float, gpusolverDnSgetrf);
OLYMPUS_GPU_DEFINE_GETRF(double, gpusolverDnDgetrf);
OLYMPUS_GPU_DEFINE_GETRF(gpuComplex, gpusolverDnCgetrf);
OLYMPUS_GPU_DEFINE_GETRF(gpuDoubleComplex, gpusolverDnZgetrf);
#undef OLYMPUS_GPU_DEFINE_GETRF

#define OLYMPUS_GPU_DEFINE_GETRF_BATCHED(Type, Name)                              \
  template <>                                                                 \
  absl::Status GetrfBatched<Type>(gpublasHandle_t handle, int n, Type **a,    \
                                  int lda, int *ipiv, int *info, int batch) { \
    return OLYMPUS_AS_STATUS(Name(handle, n, a, lda, ipiv, info, batch));         \
  }

OLYMPUS_GPU_DEFINE_GETRF_BATCHED(float, gpublasSgetrfBatched);
OLYMPUS_GPU_DEFINE_GETRF_BATCHED(double, gpublasDgetrfBatched);
OLYMPUS_GPU_DEFINE_GETRF_BATCHED(gpublasComplex, gpublasCgetrfBatched);
OLYMPUS_GPU_DEFINE_GETRF_BATCHED(gpublasDoubleComplex, gpublasZgetrfBatched);
#undef OLYMPUS_GPU_DEFINE_GETRF_BATCHED

// QR decomposition: geqrf

#define OLYMPUS_GPU_DEFINE_GEQRF(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> GeqrfBufferSize<Type>(gpusolverDnHandle_t handle, int m, \
                                            int n) {                           \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(                                         \
        Name##_bufferSize(handle, m, n, /*A=*/nullptr, m, &lwork)));           \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Geqrf<Type>(gpusolverDnHandle_t handle, int m, int n, Type *a,  \
                           Type *tau, Type *workspace, int lwork, int *info) { \
    return OLYMPUS_AS_STATUS(                                                      \
        Name(handle, m, n, a, m, tau, workspace, lwork, info));                \
  }

OLYMPUS_GPU_DEFINE_GEQRF(float, gpusolverDnSgeqrf);
OLYMPUS_GPU_DEFINE_GEQRF(double, gpusolverDnDgeqrf);
OLYMPUS_GPU_DEFINE_GEQRF(gpuComplex, gpusolverDnCgeqrf);
OLYMPUS_GPU_DEFINE_GEQRF(gpuDoubleComplex, gpusolverDnZgeqrf);
#undef OLYMPUS_GPU_DEFINE_GEQRF

#define OLYMPUS_GPU_DEFINE_GEQRF_BATCHED(Type, Name)                        \
  template <>                                                           \
  absl::Status GeqrfBatched<Type>(gpublasHandle_t handle, int m, int n, \
                                  Type **a, Type **tau, int *info,      \
                                  int batch) {                          \
    return OLYMPUS_AS_STATUS(Name(handle, m, n, a, m, tau, info, batch));   \
  }

OLYMPUS_GPU_DEFINE_GEQRF_BATCHED(float, gpublasSgeqrfBatched);
OLYMPUS_GPU_DEFINE_GEQRF_BATCHED(double, gpublasDgeqrfBatched);
OLYMPUS_GPU_DEFINE_GEQRF_BATCHED(gpublasComplex, gpublasCgeqrfBatched);
OLYMPUS_GPU_DEFINE_GEQRF_BATCHED(gpublasDoubleComplex, gpublasZgeqrfBatched);
#undef OLYMPUS_GPU_DEFINE_GEQRF_BATCHED

// Householder transformations: orgqr

#define OLYMPUS_GPU_DEFINE_ORGQR(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> OrgqrBufferSize<Type>(gpusolverDnHandle_t handle, int m, \
                                            int n, int k) {                    \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(Name##_bufferSize(                       \
        handle, m, n, k, /*A=*/nullptr, /*lda=*/m, /*tau=*/nullptr, &lwork))); \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Orgqr<Type>(gpusolverDnHandle_t handle, int m, int n, int k,    \
                           Type *a, Type *tau, Type *workspace, int lwork,     \
                           int *info) {                                        \
    return OLYMPUS_AS_STATUS(                                                      \
        Name(handle, m, n, k, a, m, tau, workspace, lwork, info));             \
  }

OLYMPUS_GPU_DEFINE_ORGQR(float, gpusolverDnSorgqr);
OLYMPUS_GPU_DEFINE_ORGQR(double, gpusolverDnDorgqr);
OLYMPUS_GPU_DEFINE_ORGQR(gpuComplex, gpusolverDnCungqr);
OLYMPUS_GPU_DEFINE_ORGQR(gpuDoubleComplex, gpusolverDnZungqr);
#undef OLYMPUS_GPU_DEFINE_ORGQR

// Cholesky decomposition: potrf

#define OLYMPUS_GPU_DEFINE_POTRF(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> PotrfBufferSize<Type>(gpusolverDnHandle_t handle,        \
                                            gpusolverFillMode_t uplo, int n) { \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(                                         \
        Name##_bufferSize(handle, uplo, n, /*A=*/nullptr, n, &lwork)));        \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Potrf<Type>(gpusolverDnHandle_t handle,                         \
                           gpusolverFillMode_t uplo, int n, Type *a,           \
                           Type *workspace, int lwork, int *info) {            \
    return OLYMPUS_AS_STATUS(                                                      \
        Name(handle, uplo, n, a, n, workspace, lwork, info));                  \
  }

OLYMPUS_GPU_DEFINE_POTRF(float, gpusolverDnSpotrf);
OLYMPUS_GPU_DEFINE_POTRF(double, gpusolverDnDpotrf);
OLYMPUS_GPU_DEFINE_POTRF(gpuComplex, gpusolverDnCpotrf);
OLYMPUS_GPU_DEFINE_POTRF(gpuDoubleComplex, gpusolverDnZpotrf);
#undef OLYMPUS_GPU_DEFINE_POTRF

#define OLYMPUS_GPU_DEFINE_POTRF_BATCHED(Type, Name)                               \
  template <>                                                                  \
  absl::Status PotrfBatched<Type>(gpusolverDnHandle_t handle,                  \
                                  gpusolverFillMode_t uplo, int n, Type **a,   \
                                  int lda, int *info, int batch) {             \
    return OLYMPUS_AS_STATUS(Name(handle, uplo, n, a, lda, info, batch));         \
  }

OLYMPUS_GPU_DEFINE_POTRF_BATCHED(float, gpusolverDnSpotrfBatched);
OLYMPUS_GPU_DEFINE_POTRF_BATCHED(double, gpusolverDnDpotrfBatched);
OLYMPUS_GPU_DEFINE_POTRF_BATCHED(gpuComplex, gpusolverDnCpotrfBatched);
OLYMPUS_GPU_DEFINE_POTRF_BATCHED(gpuDoubleComplex, gpusolverDnZpotrfBatched);
#undef OLYMPUS_GPU_DEFINE_POTRF_BATCHED

// Symmetric (Hermitian) eigendecomposition:
// * Jacobi algorithm: syevj/heevj (batches of matrices up to 32)
// * QR algorithm: syevd/heevd

#define OLYMPUS_GPU_DEFINE_SYEVJ(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> SyevjBufferSize<Type>(                                   \
      gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                     \
      gpusolverFillMode_t uplo, int n, gpuSyevjInfo_t params) {                \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(                                         \
        Name##_bufferSize(handle, jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,     \
                          /*w=*/nullptr, &lwork, params)));                    \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Syevj<Type>(                                                    \
      gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                     \
      gpusolverFillMode_t uplo, int n, Type *a, RealType<Type>::value *w,      \
      Type *workspace, int lwork, int *info, gpuSyevjInfo_t params) {          \
    return OLYMPUS_AS_STATUS(                                                      \
        Name(handle, jobz, uplo, n, a, n, w, workspace, lwork, info, params)); \
  }

OLYMPUS_GPU_DEFINE_SYEVJ(float, gpusolverDnSsyevj);
OLYMPUS_GPU_DEFINE_SYEVJ(double, gpusolverDnDsyevj);
OLYMPUS_GPU_DEFINE_SYEVJ(gpuComplex, gpusolverDnCheevj);
OLYMPUS_GPU_DEFINE_SYEVJ(gpuDoubleComplex, gpusolverDnZheevj);
#undef OLYMPUS_GPU_DEFINE_SYEVJ

#define OLYMPUS_GPU_DEFINE_SYEVJ_BATCHED(Type, Name)                           \
  template <>                                                              \
  absl::StatusOr<int> SyevjBatchedBufferSize<Type>(                        \
      gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                 \
      gpusolverFillMode_t uplo, int n, gpuSyevjInfo_t params, int batch) { \
    int lwork;                                                             \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(                                     \
        Name##_bufferSize(handle, jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, \
                          /*w=*/nullptr, &lwork, params, batch)));         \
    return lwork;                                                          \
  }                                                                        \
                                                                           \
  template <>                                                              \
  absl::Status SyevjBatched<Type>(                                         \
      gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                 \
      gpusolverFillMode_t uplo, int n, Type *a, RealType<Type>::value *w,  \
      Type *workspace, int lwork, int *info, gpuSyevjInfo_t params,        \
      int batch) {                                                         \
    return OLYMPUS_AS_STATUS(Name(handle, jobz, uplo, n, a, n, w, workspace,   \
                              lwork, info, params, batch));                \
  }

OLYMPUS_GPU_DEFINE_SYEVJ_BATCHED(float, gpusolverDnSsyevjBatched);
OLYMPUS_GPU_DEFINE_SYEVJ_BATCHED(double, gpusolverDnDsyevjBatched);
OLYMPUS_GPU_DEFINE_SYEVJ_BATCHED(gpuComplex, gpusolverDnCheevjBatched);
OLYMPUS_GPU_DEFINE_SYEVJ_BATCHED(gpuDoubleComplex, gpusolverDnZheevjBatched);
#undef OLYMPUS_GPU_DEFINE_SYEVJ_BATCHED

#define OLYMPUS_GPU_DEFINE_SYEVD(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> SyevdBufferSize<Type>(gpusolverDnHandle_t handle,        \
                                            gpusolverEigMode_t jobz,           \
                                            gpusolverFillMode_t uplo, int n) { \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(                                                       \
        OLYMPUS_AS_STATUS(Name##_bufferSize(handle, jobz, uplo, n, /*A=*/nullptr,  \
                                        /*lda=*/n, /*w=*/nullptr, &lwork)));   \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Syevd<Type>(gpusolverDnHandle_t handle,                         \
                           gpusolverEigMode_t jobz, gpusolverFillMode_t uplo,  \
                           int n, Type *a, RealType<Type>::value *w,           \
                           Type *workspace, int lwork, int *info) {            \
    return OLYMPUS_AS_STATUS(                                                      \
        Name(handle, jobz, uplo, n, a, n, w, workspace, lwork, info));         \
  }

OLYMPUS_GPU_DEFINE_SYEVD(float, gpusolverDnSsyevd);
OLYMPUS_GPU_DEFINE_SYEVD(double, gpusolverDnDsyevd);
OLYMPUS_GPU_DEFINE_SYEVD(gpuComplex, gpusolverDnCheevd);
OLYMPUS_GPU_DEFINE_SYEVD(gpuDoubleComplex, gpusolverDnZheevd);
#undef OLYMPUS_GPU_DEFINE_SYEVD

// Symmetric rank-k update: syrk

#define OLYMPUS_GPU_DEFINE_SYRK(Type, Name)                                       \
  template <>                                                                 \
  absl::Status Syrk<Type>(gpublasHandle_t handle, gpublasFillMode_t uplo,     \
                          gpublasOperation_t trans, int n, int k,             \
                          const Type *alpha, const Type *a, const Type *beta, \
                          Type *c) {                                          \
    int lda = trans == GPUBLAS_OP_N ? n : k;                                  \
    return OLYMPUS_AS_STATUS(                                                     \
        Name(handle, uplo, trans, n, k, alpha, a, lda, beta, c, n));          \
  }

OLYMPUS_GPU_DEFINE_SYRK(float, gpublasSsyrk);
OLYMPUS_GPU_DEFINE_SYRK(double, gpublasDsyrk);
OLYMPUS_GPU_DEFINE_SYRK(gpublasComplex, gpublasCsyrk);
OLYMPUS_GPU_DEFINE_SYRK(gpublasDoubleComplex, gpublasZsyrk);
#undef OLYMPUS_GPU_DEFINE_SYRK

// Singular Value Decomposition: gesvd

#define OLYMPUS_GPU_DEFINE_GESVD(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> GesvdBufferSize<Type>(gpusolverDnHandle_t handle,        \
                                            signed char job, int m, int n) {   \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(                                                       \
        OLYMPUS_AS_STATUS(Name##_bufferSize(handle, job, job, m, n, &lwork)));     \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Gesvd<Type>(gpusolverDnHandle_t handle, signed char job, int m, \
                           int n, Type *a, RealType<Type>::value *s, Type *u,  \
                           Type *vt, Type *workspace, int lwork, int *info) {  \
    return OLYMPUS_AS_STATUS(Name(handle, job, job, m, n, a, m, s, u, m, vt, n,    \
                              workspace, lwork, /*rwork=*/nullptr, info));     \
  }

OLYMPUS_GPU_DEFINE_GESVD(float, gpusolverDnSgesvd);
OLYMPUS_GPU_DEFINE_GESVD(double, gpusolverDnDgesvd);
OLYMPUS_GPU_DEFINE_GESVD(gpuComplex, gpusolverDnCgesvd);
OLYMPUS_GPU_DEFINE_GESVD(gpuDoubleComplex, gpusolverDnZgesvd);
#undef OLYMPUS_GPU_DEFINE_GESVD

#ifdef OLYMPUS_GPU_CUDA

#define OLYMPUS_GPU_DEFINE_GESVDJ(Type, Name)                                      \
  template <>                                                                  \
  absl::StatusOr<int> GesvdjBufferSize<Type>(                                  \
      gpusolverDnHandle_t handle, gpusolverEigMode_t job, int econ, int m,     \
      int n, gpuGesvdjInfo_t params) {                                         \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(Name##_bufferSize(                       \
        handle, job, econ, m, n, /*a=*/nullptr, /*lda=*/m, /*s=*/nullptr,      \
        /*u=*/nullptr, /*ldu=*/m, /*v=*/nullptr, /*ldv=*/n, &lwork, params))); \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Gesvdj<Type>(                                                   \
      gpusolverDnHandle_t handle, gpusolverEigMode_t job, int econ, int m,     \
      int n, Type *a, RealType<Type>::value *s, Type *u, Type *v,              \
      Type *workspace, int lwork, int *info, gpuGesvdjInfo_t params) {         \
    return OLYMPUS_AS_STATUS(Name(handle, job, econ, m, n, a, m, s, u, m, v, n,    \
                              workspace, lwork, info, params));                \
  }

OLYMPUS_GPU_DEFINE_GESVDJ(float, gpusolverDnSgesvdj);
OLYMPUS_GPU_DEFINE_GESVDJ(double, gpusolverDnDgesvdj);
OLYMPUS_GPU_DEFINE_GESVDJ(gpuComplex, gpusolverDnCgesvdj);
OLYMPUS_GPU_DEFINE_GESVDJ(gpuDoubleComplex, gpusolverDnZgesvdj);
#undef OLYMPUS_GPU_DEFINE_GESVDJ

#define OLYMPUS_GPU_DEFINE_GESVDJ_BATCHED(Type, Name)                             \
  template <>                                                                 \
  absl::StatusOr<int> GesvdjBatchedBufferSize<Type>(                          \
      gpusolverDnHandle_t handle, gpusolverEigMode_t job, int m, int n,       \
      gpuGesvdjInfo_t params, int batch) {                                    \
    int lwork;                                                                \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(                                        \
        Name##_bufferSize(handle, job, m, n, /*a=*/nullptr, /*lda=*/m,        \
                          /*s=*/nullptr, /*u=*/nullptr, /*ldu=*/m,            \
                          /*v=*/nullptr, /*ldv=*/n, &lwork, params, batch))); \
    return lwork;                                                             \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  absl::Status GesvdjBatched<Type>(                                           \
      gpusolverDnHandle_t handle, gpusolverEigMode_t job, int m, int n,       \
      Type *a, RealType<Type>::value *s, Type *u, Type *v, Type *workspace,   \
      int lwork, int *info, gpuGesvdjInfo_t params, int batch) {              \
    return OLYMPUS_AS_STATUS(Name(handle, job, m, n, a, m, s, u, m, v, n,         \
                              workspace, lwork, info, params, batch));        \
  }

OLYMPUS_GPU_DEFINE_GESVDJ_BATCHED(float, gpusolverDnSgesvdjBatched);
OLYMPUS_GPU_DEFINE_GESVDJ_BATCHED(double, gpusolverDnDgesvdjBatched);
OLYMPUS_GPU_DEFINE_GESVDJ_BATCHED(gpuComplex, gpusolverDnCgesvdjBatched);
OLYMPUS_GPU_DEFINE_GESVDJ_BATCHED(gpuDoubleComplex, gpusolverDnZgesvdjBatched);
#undef OLYMPUS_GPU_DEFINE_GESVDJ_BATCHED

#define OLYMPUS_GPU_DEFINE_CSRLSVQR(Type, Scalar, Name)                          \
  template <>                                                                \
  absl::Status Csrlsvqr<Type>(                                               \
      cusolverSpHandle_t handle, int n, int nnz, cusparseMatDescr_t matdesc, \
      const Type *csrValA, const int *csrRowPtrA, const int *csrColIndA,     \
      const Type *b, double tol, int reorder, Type *x, int *singularity) {   \
    return OLYMPUS_AS_STATUS(Name(handle, n, nnz, matdesc, csrValA, csrRowPtrA,  \
                              csrColIndA, b, static_cast<Scalar>(tol),       \
                              reorder, x, singularity));                     \
  }

OLYMPUS_GPU_DEFINE_CSRLSVQR(float, float, cusolverSpScsrlsvqr);
OLYMPUS_GPU_DEFINE_CSRLSVQR(double, double, cusolverSpDcsrlsvqr);
OLYMPUS_GPU_DEFINE_CSRLSVQR(gpuComplex, float, cusolverSpCcsrlsvqr);
OLYMPUS_GPU_DEFINE_CSRLSVQR(gpuDoubleComplex, double, cusolverSpZcsrlsvqr);
#undef OLYMPUS_GPU_DEFINE_CSRLSVQR

#endif  // OLYMPUS_GPU_CUDA

// Symmetric tridiagonal reduction: sytrd

#define OLYMPUS_GPU_DEFINE_SYTRD(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> SytrdBufferSize<Type>(gpusolverDnHandle_t handle,        \
                                            gpusolverFillMode_t uplo, int n) { \
    int lwork;                                                                 \
    OLYMPUS_RETURN_IF_ERROR(OLYMPUS_AS_STATUS(Name##_bufferSize(                       \
        handle, uplo, n, /*A=*/nullptr, /*lda=*/n, /*D=*/nullptr,              \
        /*E=*/nullptr, /*tau=*/nullptr, &lwork)));                             \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Sytrd<Type>(gpusolverDnHandle_t handle,                         \
                           gpusolverFillMode_t uplo, int n, Type *a,           \
                           RealType<Type>::value *d, RealType<Type>::value *e, \
                           Type *tau, Type *workspace, int lwork, int *info) { \
    return OLYMPUS_AS_STATUS(                                                      \
        Name(handle, uplo, n, a, n, d, e, tau, workspace, lwork, info));       \
  }

OLYMPUS_GPU_DEFINE_SYTRD(float, gpusolverDnSsytrd);
OLYMPUS_GPU_DEFINE_SYTRD(double, gpusolverDnDsytrd);
OLYMPUS_GPU_DEFINE_SYTRD(gpuComplex, gpusolverDnChetrd);
OLYMPUS_GPU_DEFINE_SYTRD(gpuDoubleComplex, gpusolverDnZhetrd);
#undef OLYMPUS_GPU_DEFINE_SYTRD

}  // namespace solver
}  // namespace OLYMPUS_GPU_NAMESPACE
}  // namespace olympus
