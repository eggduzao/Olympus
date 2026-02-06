# Copyright 2020 The OLYMPUS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script that builds a olympuslib wheel, intended to be run via bazel run as part
# of the olympuslib build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import os
import pathlib
import platform
import re
import subprocess
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from olympuslib.tools import build_utils

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--sources_path",
    default=None,
    help="Path in which the wheel's sources should be prepared. Optional. If "
    "omitted, a temporary directory will be used.",
)
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--olympuslib_git_hash",
    default="",
    required=True,
    help="Git hash. Empty if unknown. Required.",
)
parser.add_argument(
    "--cpu", default=None, required=True, help="Target CPU architecture. Required."
)
parser.add_argument(
    "--editable",
    action="store_true",
    help="Create an 'editable' olympuslib build instead of a wheel.",
)
parser.add_argument(
    "--srcs", help="source files for the wheel", action="append"
)

args = parser.parse_args()

r = runfiles.Create()

def _is_mac():
  return platform.system() == "Darwin"


soext = "dll" if build_utils.is_windows() else ("dylib" if _is_mac() else "so")
pyext = "pyd" if build_utils.is_windows() else "so"


def _get_file_path(src_file, runfiles=None, wheel_sources_map=None):
  if wheel_sources_map:
    return wheel_sources_map.get(
        src_file.replace(build_utils.MAIN_RUNFILES_DIR, ""), None
    )
  # TODO(ybaturina): remove the runfiles part when we switch to the new wheel
  # build rules and the runfiles are not needed.
  elif runfiles:
    return runfiles.Rlocation(src_file)
  else:
    raise RuntimeError("Either runfiles or wheel_sources should be provided!")


def patch_copy_mlir_import(
    src_file, dst_dir, runfiles=None, wheel_sources_map=None
):
  src_file = _get_file_path(src_file, runfiles, wheel_sources_map)
  src_filename = os.path.basename(src_file)
  with open(src_file, encoding="utf-8") as f:
    src = f.read()

  with open(dst_dir / src_filename, "w", encoding="utf-8") as f:
    replaced = re.sub(
        r"^from mlir(\..*)? import (.*)",
        r"from olympuslib.mlir\1 import \2",
        src,
        flags=re.MULTILINE,
    )
    f.write(replaced)


def verify_mac_libraries_dont_reference_chkstack(
    runfiles=None, wheel_sources_map=None, olympus_source_prefix="__main__"
):
  """Verifies that _olympus.so doesn't depend on ____chkstk_darwin.

  We don't entirely know why this happens, but in some build environments
  we seem to target the wrong Mac OS version.
  https://github.com/olympus-ml/olympus/issues/3867

  This check makes sure we don't release wheels that have this dependency.
  """
  if not _is_mac():
    return
  file_path = _get_file_path(
      f"{olympus_source_prefix}olympuslib/_olympus.{pyext}", runfiles, wheel_sources_map
  )
  nm = subprocess.run(
      ["nm", "-g", file_path],
      capture_output=True,
      text=True,
      check=False,
  )
  if nm.returncode != 0:
    raise RuntimeError(f"nm process failed: {nm.stdout} {nm.stderr}")
  if "____chkstk_darwin" in nm.stdout:
    raise RuntimeError(
        "Mac wheel incorrectly depends on symbol ____chkstk_darwin, which "
        "means that it isn't compatible with older MacOS versions."
    )


def write_setup_cfg(sources_path, cpu):
  tag = build_utils.platform_tag(cpu)
  with open(sources_path / "setup.cfg", "w") as f:
    f.write(
        f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat_name={tag}
"""
    )


def prepare_wheel(wheel_sources_path: pathlib.Path, *, cpu, wheel_sources):
  """Assembles a source tree for the wheel in `wheel_sources_path`. In case of
  build under @olympus strip the prefix"""
  # wheel_sources is a list of file paths, not a prefix. If provided, use empty prefix.
  # Otherwise, determine prefix based on build_from_external_workspace flag.
  source_file_prefix = f"{r.CurrentRepository()}/" if not wheel_sources else ""

  # The wheel sources provided by the transitive rules might have different path
  # prefixes, so we need to create a map of paths relative to the root package
  # to the full paths.
  # E.g. if we have the wheel sources paths like
  # bazel-out/k8-opt/bin/olympuslib/mlir/_mlir_libs/_olympus_mlir_ext.py and
  # external/xla/xla/ffi/api/c_api.h, the resulting map will be
  # {'olympuslib/mlir/_mlir_libs/_olympus_mlir_ext.py':
  # 'bazel-out/k8-opt/bin/olympuslib/mlir/_mlir_libs/_olympus_mlir_ext.py',
  # 'xla/ffi/api/c_api.h': 'external/xla/xla/ffi/api/c_api.h'}
  wheel_sources_map = build_utils.create_wheel_sources_map(
      wheel_sources, root_packages=["olympuslib", "xla"]
  )
  copy_files = functools.partial(
      build_utils.copy_file,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )

  verify_mac_libraries_dont_reference_chkstack(
      runfiles=r, wheel_sources_map=wheel_sources_map, olympus_source_prefix=source_file_prefix
  )
  copy_files(
      dst_dir=wheel_sources_path,
      src_files=[
          f"{source_file_prefix}olympuslib/tools/LICENSE.txt",
          f"{source_file_prefix}olympuslib/README.md",
          f"{source_file_prefix}olympuslib/setup.py",
      ],
  )
  write_setup_cfg(wheel_sources_path, cpu)

  olympuslib_dir = wheel_sources_path / "olympuslib"
  copy_files(
      f"{source_file_prefix}olympuslib/init.py",
      dst_dir=olympuslib_dir,
      dst_filename="__init__.py",
  )
  copy_files(
      dst_dir=olympuslib_dir,
      src_files=[
          f"{source_file_prefix}olympuslib/cpu_feature_guard.{pyext}",
          f"{source_file_prefix}olympuslib/cpu_sparse.py",
          f"{source_file_prefix}olympuslib/utils.{pyext}",
          f"{source_file_prefix}olympuslib/olympus_common.dll"
          if build_utils.is_windows()
          else f"{source_file_prefix}olympuslib/libolympus_common.{soext}",
          f"{source_file_prefix}olympuslib/lapack.py",
          f"{source_file_prefix}olympuslib/gpu_prng.py",
          f"{source_file_prefix}olympuslib/gpu_linalg.py",
          f"{source_file_prefix}olympuslib/gpu_rnn.py",
          f"{source_file_prefix}olympuslib/gpu_triton.py",
          f"{source_file_prefix}olympuslib/gpu_common_utils.py",
          f"{source_file_prefix}olympuslib/gpu_solver.py",
          f"{source_file_prefix}olympuslib/gpu_sparse.py",
          f"{source_file_prefix}olympuslib/plugin_support.py",
          f"{source_file_prefix}olympuslib/_pretty_printer.{pyext}",
          f"{source_file_prefix}olympuslib/version.py",
          f"{source_file_prefix}olympuslib/xla_client.py",
          f"{source_file_prefix}olympuslib/weakref_lru_cache.{pyext}",
          f"{source_file_prefix}olympuslib/weakref_lru_cache.pyi",
          f"{source_file_prefix}olympuslib/_ifrt_proxy.{pyext}",
          f"{source_file_prefix}olympuslib/_olympus.{pyext}",
          f"{source_file_prefix}olympuslib/_sdy_mpmd.{pyext}",
          f"{source_file_prefix}olympuslib/_pathways.{pyext}",
          f"{source_file_prefix}olympuslib/_profiler.{pyext}",
          f"{source_file_prefix}olympuslib/_profile_data.{pyext}",
      ],
  )
  # This file is required by PEP-561. It marks olympuslib as package containing
  # type stubs.
  with open(olympuslib_dir / "py.typed", "w"):
    pass

  copy_files(
      dst_dir=olympuslib_dir / "cpu",
      src_files=[
          f"{source_file_prefix}olympuslib/cpu/_lapack.{pyext}",
          f"{source_file_prefix}olympuslib/cpu/_sparse.{pyext}",
      ],
  )

  mosaic_python_dir = olympuslib_dir / "mosaic" / "python"
  copy_files(
      dst_dir=mosaic_python_dir,
      src_files=[
          f"{source_file_prefix}olympuslib/mosaic/python/layout_defs.py",
          f"{source_file_prefix}olympuslib/mosaic/python/mosaic_gpu.py",
          f"{source_file_prefix}olympuslib/mosaic/python/tpu.py",
      ],
  )
  # TODO (sharadmv,skyewm): can we avoid patching this file?
  patch_copy_mlir_import(
      f"{source_file_prefix}olympuslib/mosaic/python/_tpu_gen.py",
      dst_dir=mosaic_python_dir,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )
  mosaic_gpu_dir = olympuslib_dir / "mosaic" / "dialect" / "gpu"
  os.makedirs(mosaic_gpu_dir)
  patch_copy_mlir_import(
      f"{source_file_prefix}olympuslib/mosaic/dialect/gpu/_mosaic_gpu_gen_ops.py",
      dst_dir=mosaic_gpu_dir,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )
  patch_copy_mlir_import(
      f"{source_file_prefix}olympuslib/mosaic/dialect/gpu/_mosaic_gpu_gen_enums.py",
      dst_dir=mosaic_gpu_dir,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )

  copy_files(
      dst_dir=olympuslib_dir / "mlir",
      src_files=[
          f"{source_file_prefix}olympuslib/mlir/ir.py",
          f"{source_file_prefix}olympuslib/mlir/passmanager.py",
      ],
  )
  copy_files(
      dst_dir=olympuslib_dir / "mlir" / "dialects",
      src_files=[
          f"{source_file_prefix}olympuslib/mlir/dialects/_arith_enum_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_arith_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_builtin_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_cf_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_chlo_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_func_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_math_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_memref_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_mhlo_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_mpmd_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_ods_common.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_scf_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_sdy_enums_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_sdy_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_sparse_tensor_enum_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_sparse_tensor_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_stablehlo_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_vector_enum_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_vector_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_gpu_enum_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_gpu_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_nvgpu_enum_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_nvgpu_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_nvvm_enum_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_nvvm_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_llvm_enum_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/_llvm_ops_gen.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/arith.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/builtin.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/cf.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/chlo.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/func.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/math.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/memref.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/mhlo.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/mpmd.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/scf.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/sdy.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/sparse_tensor.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/stablehlo.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/vector.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/nvgpu.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/nvvm.py",
          f"{source_file_prefix}olympuslib/mlir/dialects/llvm.py",
      ],
  )
  copy_files(
      dst_dir=olympuslib_dir / "mlir" / "extras",
      src_files=[
          f"{source_file_prefix}olympuslib/mlir/extras/meta.py",
          f"{source_file_prefix}olympuslib/mlir/extras/types.py",
      ],
  )
  copy_files(
      dst_dir=olympuslib_dir / "mlir" / "dialects" / "gpu",
      src_files=[
          f"{source_file_prefix}olympuslib/mlir/dialects/gpu/__init__.py",
      ],
  )
  copy_files(
      dst_dir=olympuslib_dir / "mlir" / "dialects" / "gpu" / "passes",
      src_files=[
          f"{source_file_prefix}olympuslib/mlir/dialects/gpu/passes/__init__.py",
      ],
  )

  mlir_libs_dir = olympuslib_dir / "mlir" / "_mlir_libs"
  copy_files(
      dst_dir=mlir_libs_dir,
      src_files=[
          f"{source_file_prefix}olympuslib/mlir/_mlir_libs/__init__.py",
          f"{source_file_prefix}olympuslib/_mlir.{pyext}",
          f"{source_file_prefix}olympuslib/_chlo.{pyext}",
          f"{source_file_prefix}olympuslib/_mlirHlo.{pyext}",
          f"{source_file_prefix}olympuslib/_mlirDialectsSparseTensor.{pyext}",
          f"{source_file_prefix}olympuslib/_mlirSparseTensorPasses.{pyext}",
          f"{source_file_prefix}olympuslib/_mosaic_gpu_ext.{pyext}",
          f"{source_file_prefix}olympuslib/_tpu_ext.{pyext}",
          f"{source_file_prefix}olympuslib/_sdy.{pyext}",
          f"{source_file_prefix}olympuslib/_sdyMpmd.{pyext}",
          f"{source_file_prefix}olympuslib/_stablehlo.{pyext}",
          f"{source_file_prefix}olympuslib/_olympus_mlir_ext.{pyext}",
          f"{source_file_prefix}olympuslib/_mlirDialectsGPU.{pyext}",
          f"{source_file_prefix}olympuslib/_mlirDialectsLLVM.{pyext}",
          f"{source_file_prefix}olympuslib/_mlirDialectsNVGPU.{pyext}",
          f"{source_file_prefix}olympuslib/_mlirGPUPasses.{pyext}",
      ]
      + (
          []
          if build_utils.is_windows()
          else [
              f"{source_file_prefix}olympuslib/_triton_ext.{pyext}",
              f"{source_file_prefix}olympuslib/mlir/_mlir_libs/_triton_ext.pyi",
          ]
      ),
  )

  triton_dir = olympuslib_dir / "triton"
  copy_files(
      dst_dir=triton_dir,
      src_files=[
          f"{source_file_prefix}olympuslib/triton/__init__.py",
          f"{source_file_prefix}olympuslib/triton/dialect.py",
      ],
  )
  patch_copy_mlir_import(
      f"{source_file_prefix}olympuslib/triton/_triton_enum_gen.py",
      dst_dir=triton_dir,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )
  patch_copy_mlir_import(
      f"{source_file_prefix}olympuslib/triton/_triton_ops_gen.py",
      dst_dir=triton_dir,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )

  if wheel_sources:
    xla_ffi_files = [
        "xla/ffi/api/c_api.h",
        "xla/ffi/api/api.h",
        "xla/ffi/api/ffi.h",
    ]
  else:
    xla_ffi_files = [
        f"{source_file_prefix}olympuslib/include/xla/ffi/api/c_api.h",
        f"{source_file_prefix}olympuslib/include/xla/ffi/api/api.h",
        f"{source_file_prefix}olympuslib/include/xla/ffi/api/ffi.h",
    ]

  copy_files(
        dst_dir=olympuslib_dir / "include" / "xla" / "ffi" / "api",
        src_files=xla_ffi_files,
   )

tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="olympuslib")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_wheel(
      pathlib.Path(sources_path),
      cpu=args.cpu,
      wheel_sources=args.srcs,
  )
  package_name = "olympuslib"
  if args.editable:
    build_utils.build_editable(sources_path, args.output_path, package_name)
  else:
    build_utils.build_wheel(
        sources_path,
        args.output_path,
        package_name,
        git_hash=args.olympuslib_git_hash,
    )
finally:
  if tmpdir:
    tmpdir.cleanup()
