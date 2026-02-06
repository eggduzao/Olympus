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

#include "olympuslib/cpu/lapack_kernels.h"

// From a Python binary, OLYMPUS obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// OLYMPUS-generated HLO from C++.

namespace ffi = xla::ffi;

extern "C" {

olympus::TriMatrixEquationSolver<ffi::DataType::F32>::FnType strsm_;
olympus::TriMatrixEquationSolver<ffi::DataType::F64>::FnType dtrsm_;
olympus::TriMatrixEquationSolver<ffi::DataType::C64>::FnType ctrsm_;
olympus::TriMatrixEquationSolver<ffi::DataType::C128>::FnType ztrsm_;

olympus::LuDecomposition<ffi::DataType::F32>::FnType sgetrf_;
olympus::LuDecomposition<ffi::DataType::F64>::FnType dgetrf_;
olympus::LuDecomposition<ffi::DataType::C64>::FnType cgetrf_;
olympus::LuDecomposition<ffi::DataType::C128>::FnType zgetrf_;

olympus::QrFactorization<ffi::DataType::F32>::FnType sgeqrf_;
olympus::QrFactorization<ffi::DataType::F64>::FnType dgeqrf_;
olympus::QrFactorization<ffi::DataType::C64>::FnType cgeqrf_;
olympus::QrFactorization<ffi::DataType::C128>::FnType zgeqrf_;

olympus::PivotingQrFactorization<ffi::DataType::F32>::FnType sgeqp3_;
olympus::PivotingQrFactorization<ffi::DataType::F64>::FnType dgeqp3_;
olympus::PivotingQrFactorization<ffi::DataType::C64>::FnType cgeqp3_;
olympus::PivotingQrFactorization<ffi::DataType::C128>::FnType zgeqp3_;

olympus::OrthogonalQr<ffi::DataType::F32>::FnType sorgqr_;
olympus::OrthogonalQr<ffi::DataType::F64>::FnType dorgqr_;
olympus::OrthogonalQr<ffi::DataType::C64>::FnType cungqr_;
olympus::OrthogonalQr<ffi::DataType::C128>::FnType zungqr_;

olympus::CholeskyFactorization<ffi::DataType::F32>::FnType spotrf_;
olympus::CholeskyFactorization<ffi::DataType::F64>::FnType dpotrf_;
olympus::CholeskyFactorization<ffi::DataType::C64>::FnType cpotrf_;
olympus::CholeskyFactorization<ffi::DataType::C128>::FnType zpotrf_;

olympus::SingularValueDecomposition<ffi::DataType::F32>::FnType sgesdd_;
olympus::SingularValueDecomposition<ffi::DataType::F64>::FnType dgesdd_;
olympus::SingularValueDecompositionComplex<ffi::DataType::C64>::FnType cgesdd_;
olympus::SingularValueDecompositionComplex<ffi::DataType::C128>::FnType zgesdd_;

olympus::SingularValueDecompositionQR<ffi::DataType::F32>::FnType sgesvd_;
olympus::SingularValueDecompositionQR<ffi::DataType::F64>::FnType dgesvd_;
olympus::SingularValueDecompositionQRComplex<ffi::DataType::C64>::FnType cgesvd_;
olympus::SingularValueDecompositionQRComplex<ffi::DataType::C128>::FnType zgesvd_;

olympus::EigenvalueDecompositionSymmetric<ffi::DataType::F32>::FnType ssyevd_;
olympus::EigenvalueDecompositionSymmetric<ffi::DataType::F64>::FnType dsyevd_;
olympus::EigenvalueDecompositionHermitian<ffi::DataType::C64>::FnType cheevd_;
olympus::EigenvalueDecompositionHermitian<ffi::DataType::C128>::FnType zheevd_;

olympus::EigenvalueDecomposition<ffi::DataType::F32>::FnType sgeev_;
olympus::EigenvalueDecomposition<ffi::DataType::F64>::FnType dgeev_;
olympus::EigenvalueDecompositionComplex<ffi::DataType::C64>::FnType cgeev_;
olympus::EigenvalueDecompositionComplex<ffi::DataType::C128>::FnType zgeev_;

olympus::SchurDecomposition<ffi::DataType::F32>::FnType sgees_;
olympus::SchurDecomposition<ffi::DataType::F64>::FnType dgees_;
olympus::SchurDecompositionComplex<ffi::DataType::C64>::FnType cgees_;
olympus::SchurDecompositionComplex<ffi::DataType::C128>::FnType zgees_;

olympus::HessenbergDecomposition<ffi::DataType::F32>::FnType sgehrd_;
olympus::HessenbergDecomposition<ffi::DataType::F64>::FnType dgehrd_;
olympus::HessenbergDecomposition<ffi::DataType::C64>::FnType cgehrd_;
olympus::HessenbergDecomposition<ffi::DataType::C128>::FnType zgehrd_;

olympus::TridiagonalReduction<ffi::DataType::F32>::FnType ssytrd_;
olympus::TridiagonalReduction<ffi::DataType::F64>::FnType dsytrd_;
olympus::TridiagonalReduction<ffi::DataType::C64>::FnType chetrd_;
olympus::TridiagonalReduction<ffi::DataType::C128>::FnType zhetrd_;

olympus::TridiagonalSolver<ffi::DataType::F32>::FnType sgtsv_;
olympus::TridiagonalSolver<ffi::DataType::F64>::FnType dgtsv_;
olympus::TridiagonalSolver<ffi::DataType::C64>::FnType cgtsv_;
olympus::TridiagonalSolver<ffi::DataType::C128>::FnType zgtsv_;

}  // extern "C"

namespace olympus {

static auto init = []() -> int {
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F32>>(strsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F64>>(dtrsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C64>>(ctrsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C128>>(ztrsm_);

  AssignKernelFn<LuDecomposition<ffi::DataType::F32>>(sgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::F64>>(dgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::C64>>(cgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::C128>>(zgetrf_);

  AssignKernelFn<QrFactorization<ffi::DataType::F32>>(sgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::F64>>(dgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::C64>>(cgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::C128>>(zgeqrf_);

  AssignKernelFn<PivotingQrFactorization<ffi::DataType::F32>>(sgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::F64>>(dgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::C64>>(cgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::C128>>(zgeqp3_);

  AssignKernelFn<OrthogonalQr<ffi::DataType::F32>>(sorgqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::F64>>(dorgqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::C64>>(cungqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::C128>>(zungqr_);

  AssignKernelFn<CholeskyFactorization<ffi::DataType::F32>>(spotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::F64>>(dpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C64>>(cpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C128>>(zpotrf_);

  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F32>>(sgesdd_);
  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F64>>(dgesdd_);
  AssignKernelFn<SingularValueDecompositionComplex<ffi::DataType::C64>>(
      cgesdd_);
  AssignKernelFn<SingularValueDecompositionComplex<ffi::DataType::C128>>(
      zgesdd_);

  AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F32>>(sgesvd_);
  AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F64>>(dgesvd_);
  AssignKernelFn<SingularValueDecompositionQRComplex<ffi::DataType::C64>>(
      cgesvd_);
  AssignKernelFn<SingularValueDecompositionQRComplex<ffi::DataType::C128>>(
      zgesvd_);

  AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F32>>(ssyevd_);
  AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F64>>(dsyevd_);
  AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C64>>(cheevd_);
  AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C128>>(
      zheevd_);

  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F32>>(sgeev_);
  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F64>>(dgeev_);
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C64>>(cgeev_);
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C128>>(zgeev_);

  AssignKernelFn<TridiagonalReduction<ffi::DataType::F32>>(ssytrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::F64>>(dsytrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C64>>(chetrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C128>>(zhetrd_);

  AssignKernelFn<SchurDecomposition<ffi::DataType::F32>>(sgees_);
  AssignKernelFn<SchurDecomposition<ffi::DataType::F64>>(dgees_);
  AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C64>>(cgees_);
  AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C128>>(zgees_);

  AssignKernelFn<HessenbergDecomposition<ffi::DataType::F32>>(sgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::F64>>(dgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::C64>>(cgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::C128>>(zgehrd_);

  AssignKernelFn<TridiagonalSolver<ffi::DataType::F32>>(sgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::F64>>(dgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::C64>>(cgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::C128>>(zgtsv_);

  lapack_kernels_initialized = true;
  return 0;
}();

}  // namespace olympus
