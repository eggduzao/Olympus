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

// This file defines a standard interface to the GPU linear algebra libraries.

#ifndef OLYMPUSLIB_GPU_SOLVER_INTERFACE_H_
#define OLYMPUSLIB_GPU_SOLVER_INTERFACE_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "olympuslib/gpu/vendor.h"

#ifdef OLYMPUS_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverSp.h"
#endif

namespace olympus {
namespace OLYMPUS_GPU_NAMESPACE {
namespace solver {

template <typename T>
struct RealType {
  using value = T;
};

template <>
struct RealType<gpuComplex> {
  using value = float;
};

template <>
struct RealType<gpuDoubleComplex> {
  using value = double;
};

#define OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(ReturnType, FunctionName)            \
  template <typename T>                                                       \
  ReturnType FunctionName(                                                    \
      OLYMPUS_GPU_SOLVER_##FunctionName##_ARGS(T, typename RealType<T>::value)) { \
    return absl::UnimplementedError(absl::StrFormat(                          \
        #FunctionName " not implemented for type %s", typeid(T).name()));     \
  }                                                                           \
  template <>                                                                 \
  ReturnType FunctionName<float>(                                             \
      OLYMPUS_GPU_SOLVER_##FunctionName##_ARGS(float, float));                    \
  template <>                                                                 \
  ReturnType FunctionName<double>(                                            \
      OLYMPUS_GPU_SOLVER_##FunctionName##_ARGS(double, double));                  \
  template <>                                                                 \
  ReturnType FunctionName<gpuComplex>(                                        \
      OLYMPUS_GPU_SOLVER_##FunctionName##_ARGS(gpuComplex, float));               \
  template <>                                                                 \
  ReturnType FunctionName<gpuDoubleComplex>(                                  \
      OLYMPUS_GPU_SOLVER_##FunctionName##_ARGS(gpuDoubleComplex, double))

// LU decomposition: getrf

#define OLYMPUS_GPU_SOLVER_GetrfBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, int m, int n
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, GetrfBufferSize);
#undef OLYMPUS_GPU_SOLVER_GetrfBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Getrf_ARGS(Type, ...)                          \
  gpusolverDnHandle_t handle, int m, int n, Type *a, Type *workspace, \
      int lwork, int *ipiv, int *info
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Getrf);
#undef OLYMPUS_GPU_SOLVER_Getrf_ARGS

#define OLYMPUS_GPU_SOLVER_GetrfBatched_ARGS(Type, ...)                       \
  gpublasHandle_t handle, int n, Type **a, int lda, int *ipiv, int *info, \
      int batch
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, GetrfBatched);
#undef OLYMPUS_GPU_SOLVER_GetrfBatched_ARGS

// QR decomposition: geqrf

#define OLYMPUS_GPU_SOLVER_GeqrfBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, int m, int n
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, GeqrfBufferSize);
#undef OLYMPUS_GPU_SOLVER_GeqrfBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Geqrf_ARGS(Type, ...)                    \
  gpusolverDnHandle_t handle, int m, int n, Type *a, Type *tau, \
      Type *workspace, int lwork, int *info
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Geqrf);
#undef OLYMPUS_GPU_SOLVER_Geqrf_ARGS

#define OLYMPUS_GPU_SOLVER_GeqrfBatched_ARGS(Type, ...)                      \
  gpublasHandle_t handle, int m, int n, Type **a, Type **tau, int *info, \
      int batch
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, GeqrfBatched);
#undef OLYMPUS_GPU_SOLVER_GeqrfBatched_ARGS

// Householder transformations: orgqr

#define OLYMPUS_GPU_SOLVER_OrgqrBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, int m, int n, int k
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, OrgqrBufferSize);
#undef OLYMPUS_GPU_SOLVER_OrgqrBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Orgqr_ARGS(Type, ...)                           \
  gpusolverDnHandle_t handle, int m, int n, int k, Type *a, Type *tau, \
      Type *workspace, int lwork, int *info
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Orgqr);
#undef OLYMPUS_GPU_SOLVER_Orgqr_ARGS

// Cholesky decomposition: potrf

#define OLYMPUS_GPU_SOLVER_PotrfBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, gpusolverFillMode_t uplo, int n
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, PotrfBufferSize);
#undef OLYMPUS_GPU_SOLVER_PotrfBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Potrf_ARGS(Type, ...)                          \
  gpusolverDnHandle_t handle, gpusolverFillMode_t uplo, int n, Type *a, \
      Type *workspace, int lwork, int *info
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Potrf);
#undef OLYMPUS_GPU_SOLVER_Potrf_ARGS

#define OLYMPUS_GPU_SOLVER_PotrfBatched_ARGS(Type, ...)                       \
  gpusolverDnHandle_t handle, gpusolverFillMode_t uplo, int n, Type **a, \
      int lda, int *info, int batch
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, PotrfBatched);
#undef OLYMPUS_GPU_SOLVER_PotrfBatched_ARGS

// Symmetric (Hermitian) eigendecomposition:
// * Jacobi algorithm: syevj/heevj (batches of matrices up to 32)
// * QR algorithm: syevd/heevd

#define OLYMPUS_GPU_SOLVER_SyevjBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, \
      gpusolverFillMode_t uplo, int n, gpuSyevjInfo_t params
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, SyevjBufferSize);
#undef OLYMPUS_GPU_SOLVER_SyevjBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Syevj_ARGS(Type, Real)                             \
  gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                    \
      gpusolverFillMode_t uplo, int n, Type *a, Real *w, Type *workspace, \
      int lwork, int *info, gpuSyevjInfo_t params
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Syevj);
#undef OLYMPUS_GPU_SOLVER_Syevj_ARGS

#define OLYMPUS_GPU_SOLVER_SyevjBatchedBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,        \
      gpusolverFillMode_t uplo, int n, gpuSyevjInfo_t params, int batch
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, SyevjBatchedBufferSize);
#undef OLYMPUS_GPU_SOLVER_SyevjBatchedBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_SyevjBatched_ARGS(Type, Real)                      \
  gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                    \
      gpusolverFillMode_t uplo, int n, Type *a, Real *w, Type *workspace, \
      int lwork, int *info, gpuSyevjInfo_t params, int batch
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, SyevjBatched);
#undef OLYMPUS_GPU_SOLVER_SyevjBatched_ARGS

#define OLYMPUS_GPU_SOLVER_SyevdBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, \
      gpusolverFillMode_t uplo, int n
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, SyevdBufferSize);
#undef OLYMPUS_GPU_SOLVER_SyevdBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Syevd_ARGS(Type, Real)                             \
  gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                    \
      gpusolverFillMode_t uplo, int n, Type *a, Real *w, Type *workspace, \
      int lwork, int *info
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Syevd);
#undef OLYMPUS_GPU_SOLVER_Syevd_ARGS

// Symmetric rank-k update: syrk

#define OLYMPUS_GPU_SOLVER_Syrk_ARGS(Type, ...)                                 \
  gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans, \
      int n, int k, const Type *alpha, const Type *a, const Type *beta,     \
      Type *c
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Syrk);
#undef OLYMPUS_GPU_SOLVER_Syrk_ARGS

// Singular Value Decomposition: gesvd

#define OLYMPUS_GPU_SOLVER_GesvdBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, signed char job, int m, int n
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, GesvdBufferSize);
#undef OLYMPUS_GPU_SOLVER_GesvdBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Gesvd_ARGS(Type, Real)                                  \
  gpusolverDnHandle_t handle, signed char job, int m, int n, Type *a, Real *s, \
      Type *u, Type *vt, Type *workspace, int lwork, int *info
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Gesvd);
#undef OLYMPUS_GPU_SOLVER_Gesvd_ARGS

#ifdef OLYMPUS_GPU_CUDA

#define OLYMPUS_GPU_SOLVER_GesvdjBufferSize_ARGS(Type, ...)                       \
  gpusolverDnHandle_t handle, gpusolverEigMode_t job, int econ, int m, int n, \
      gesvdjInfo_t params
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, GesvdjBufferSize);
#undef OLYMPUS_GPU_SOLVER_GesvdjBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Gesvdj_ARGS(Type, Real)                                \
  gpusolverDnHandle_t handle, gpusolverEigMode_t job, int econ, int m, int n, \
      Type *a, Real *s, Type *u, Type *v, Type *workspace, int lwork,         \
      int *info, gesvdjInfo_t params
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Gesvdj);
#undef OLYMPUS_GPU_SOLVER_Gesvdj_ARGS

#define OLYMPUS_GPU_SOLVER_GesvdjBatchedBufferSize_ARGS(Type, ...)      \
  gpusolverDnHandle_t handle, gpusolverEigMode_t job, int m, int n, \
      gpuGesvdjInfo_t params, int batch
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, GesvdjBatchedBufferSize);
#undef OLYMPUS_GPU_SOLVER_GesvdjBatchedBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_GesvdjBatched_ARGS(Type, Real)                        \
  gpusolverDnHandle_t handle, gpusolverEigMode_t job, int m, int n, Type *a, \
      Real *s, Type *u, Type *v, Type *workspace, int lwork, int *info,      \
      gpuGesvdjInfo_t params, int batch
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, GesvdjBatched);
#undef OLYMPUS_GPU_SOLVER_GesvdjBatched_ARGS

#define OLYMPUS_GPU_SOLVER_Csrlsvqr_ARGS(Type, ...)                          \
  cusolverSpHandle_t handle, int n, int nnz, cusparseMatDescr_t matdesc, \
      const Type *csrValA, const int *csrRowPtrA, const int *csrColIndA, \
      const Type *b, double tol, int reorder, Type *x, int *singularity
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Csrlsvqr);
#undef OLYMPUS_GPU_SOLVER_Csrlsvqr_ARGS

#endif  // OLYMPUS_GPU_CUDA

// Symmetric tridiagonal reduction: sytrd

#define OLYMPUS_GPU_SOLVER_SytrdBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, SytrdBufferSize);
#undef OLYMPUS_GPU_SOLVER_SytrdBufferSize_ARGS

#define OLYMPUS_GPU_SOLVER_Sytrd_ARGS(Type, Real)                                  \
  gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, Type *a, Real *d, \
      Real *e, Type *tau, Type *workspace, int lwork, int *info
OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Sytrd);
#undef OLYMPUS_GPU_SOLVER_Sytrd_ARGS

#undef OLYMPUS_GPU_SOLVER_EXPAND_DEFINITION

}  // namespace solver
}  // namespace OLYMPUS_GPU_NAMESPACE
}  // namespace olympus

#endif  // OLYMPUSLIB_GPU_SOLVER_INTERFACE_H_
