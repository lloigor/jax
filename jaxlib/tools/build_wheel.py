# Copyright 2020 The JAX Authors.
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

# Script that builds a jaxlib wheel, intended to be run via bazel run as part
# of the jaxlib build process.

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
from jax.tools import build_utils

parser = argparse.ArgumentParser()
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
    "--jaxlib_git_hash",
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
    help="Create an 'editable' jaxlib build instead of a wheel.",
)
args = parser.parse_args()

r = runfiles.Create()


def _is_mac():
  return platform.system() == "Darwin"


pyext = "pyd" if build_utils.is_windows() else "so"


def exists(src_file):
  path = r.Rlocation(src_file)
  if path is None:
    return False
  return os.path.exists(path)


def patch_copy_mlir_import(src_file, dst_dir):
  src_file = r.Rlocation(src_file)
  src_filename = os.path.basename(src_file)
  with open(src_file) as f:
    src = f.read()

  with open(dst_dir / src_filename, "w") as f:
    replaced = re.sub(
        r"^from mlir(\..*)? import (.*)",
        r"from jaxlib.mlir\1 import \2",
        src,
        flags=re.MULTILINE,
    )
    f.write(replaced)


_XLA_EXTENSION_STUBS = [
    "__init__.pyi",
    "guard_lib.pyi",
    "ifrt_programs.pyi",
    "ifrt_proxy.pyi",
    "jax_jit.pyi",
    "ops.pyi",
    "pmap_lib.pyi",
    "profiler.pyi",
    "pytree.pyi",
    "transfer_guard_lib.pyi",
]
_OPTIONAL_XLA_EXTENSION_STUBS = []


def patch_copy_xla_extension_stubs(dst_dir):
  xla_extension_dir = os.path.join(dst_dir, "xla_extension")
  os.makedirs(xla_extension_dir)
  for stub_name in _XLA_EXTENSION_STUBS:
    stub_path = r.Rlocation("xla/xla/python/xla_extension/" + stub_name)
    stub_path = str(stub_path)  # Make pytype accept os.path.exists(stub_path).
    if stub_name in _OPTIONAL_XLA_EXTENSION_STUBS and not os.path.exists(stub_path):
      continue
    with open(stub_path) as f:
      src = f.read()
    src = src.replace(
        "from xla.python import xla_extension", "from .. import xla_extension"
    )
    with open(os.path.join(xla_extension_dir, stub_name), "w") as f:
      f.write(src)


def verify_mac_libraries_dont_reference_chkstack():
  """Verifies that xla_extension.so doesn't depend on ____chkstk_darwin.

  We don't entirely know why this happens, but in some build environments
  we seem to target the wrong Mac OS version.
  https://github.com/jax-ml/jax/issues/3867

  This check makes sure we don't release wheels that have this dependency.
  """
  if not _is_mac():
    return
  nm = subprocess.run(
      ["nm", "-g", r.Rlocation("xla/xla/python/xla_extension.so")],
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


def prepare_wheel(sources_path: pathlib.Path, *, cpu):
  """Assembles a source tree for the wheel in `sources_path`."""
  copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

  verify_mac_libraries_dont_reference_chkstack()
  print("sources_path=", sources_path)
  copy_runfiles(
      dst_dir=sources_path,
      src_files=[
          "_main/jaxlib/tools/LICENSE.txt",
          "_main/jaxlib/README.md",
          "_main/jaxlib/setup.py",
      ],
  )
  write_setup_cfg(sources_path, cpu)

  jaxlib_dir = sources_path / "jaxlib"
  copy_runfiles(
      "_main/jaxlib/init.py", dst_dir=jaxlib_dir, dst_filename="__init__.py"
  )
  copy_runfiles(
      dst_dir=jaxlib_dir,
      src_files=[
          f"_main/jaxlib/cpu_feature_guard.{pyext}",
          f"_main/jaxlib/utils.{pyext}",
          "_main/jaxlib/lapack.py",
          "_main/jaxlib/hlo_helpers.py",
          "_main/jaxlib/gpu_prng.py",
          "_main/jaxlib/gpu_linalg.py",
          "_main/jaxlib/gpu_rnn.py",
          "_main/jaxlib/gpu_triton.py",
          "_main/jaxlib/gpu_common_utils.py",
          "_main/jaxlib/gpu_solver.py",
          "_main/jaxlib/gpu_sparse.py",
          "_main/jaxlib/version.py",
          "_main/jaxlib/xla_client.py",
          f"xla/xla/python/xla_extension.{pyext}",
      ],
  )
  # This file is required by PEP-561. It marks jaxlib as package containing
  # type stubs.
  with open(jaxlib_dir / "py.typed", "w"):
    pass
  patch_copy_xla_extension_stubs(jaxlib_dir)

  copy_runfiles(
      dst_dir=jaxlib_dir / "cpu",
      src_files=[
          f"_main/jaxlib/cpu/_lapack.{pyext}",
      ],
  )

  mosaic_python_dir = jaxlib_dir / "mosaic" / "python"
  copy_runfiles(
      dst_dir=mosaic_python_dir,
      src_files=[
          "_main/jaxlib/mosaic/python/layout_defs.py",
          "_main/jaxlib/mosaic/python/mosaic_gpu.py",
          "_main/jaxlib/mosaic/python/tpu.py",
      ],
  )
  # TODO (sharadmv,skyewm): can we avoid patching this file?
  patch_copy_mlir_import(
      "_main/jaxlib/mosaic/python/_tpu_gen.py", dst_dir=mosaic_python_dir
  )
  mosaic_gpu_dir = jaxlib_dir / "mosaic" / "dialect" / "gpu"
  os.makedirs(mosaic_gpu_dir)
  patch_copy_mlir_import(
      "_main/jaxlib/mosaic/dialect/gpu/_mosaic_gpu_gen_ops.py",
      dst_dir=mosaic_gpu_dir,
  )
  patch_copy_mlir_import(
      "_main/jaxlib/mosaic/dialect/gpu/_mosaic_gpu_gen_enums.py",
      dst_dir=mosaic_gpu_dir,
  )

  copy_runfiles(
      dst_dir=jaxlib_dir / "mlir",
      src_files=[
          "_main/jaxlib/mlir/ir.py",
          "_main/jaxlib/mlir/ir.pyi",
          "_main/jaxlib/mlir/passmanager.py",
          "_main/jaxlib/mlir/passmanager.pyi",
      ],
  )
  copy_runfiles(
      dst_dir=jaxlib_dir / "mlir" / "dialects",
      src_files=[
          "_main/jaxlib/mlir/dialects/_arith_enum_gen.py",
          "_main/jaxlib/mlir/dialects/_arith_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_builtin_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_chlo_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_func_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_math_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_memref_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_mhlo_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_ods_common.py",
          "_main/jaxlib/mlir/dialects/_scf_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_sdy_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_sparse_tensor_enum_gen.py",
          "_main/jaxlib/mlir/dialects/_sparse_tensor_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_stablehlo_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_vector_enum_gen.py",
          "_main/jaxlib/mlir/dialects/_vector_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_gpu_enum_gen.py",
          "_main/jaxlib/mlir/dialects/_gpu_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_nvgpu_enum_gen.py",
          "_main/jaxlib/mlir/dialects/_nvgpu_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_nvvm_enum_gen.py",
          "_main/jaxlib/mlir/dialects/_nvvm_ops_gen.py",
          "_main/jaxlib/mlir/dialects/_llvm_enum_gen.py",
          "_main/jaxlib/mlir/dialects/_llvm_ops_gen.py",
          "_main/jaxlib/mlir/dialects/arith.py",
          "_main/jaxlib/mlir/dialects/builtin.py",
          "_main/jaxlib/mlir/dialects/chlo.py",
          "_main/jaxlib/mlir/dialects/func.py",
          "_main/jaxlib/mlir/dialects/math.py",
          "_main/jaxlib/mlir/dialects/memref.py",
          "_main/jaxlib/mlir/dialects/mhlo.py",
          "_main/jaxlib/mlir/dialects/scf.py",
          "_main/jaxlib/mlir/dialects/sdy.py",
          "_main/jaxlib/mlir/dialects/sparse_tensor.py",
          "_main/jaxlib/mlir/dialects/stablehlo.py",
          "_main/jaxlib/mlir/dialects/vector.py",
          "_main/jaxlib/mlir/dialects/nvgpu.py",
          "_main/jaxlib/mlir/dialects/nvvm.py",
          "_main/jaxlib/mlir/dialects/llvm.py",
      ],
  )
  copy_runfiles(
      dst_dir=jaxlib_dir / "mlir" / "extras",
      src_files=[
          "_main/jaxlib/mlir/extras/meta.py",
      ],
  )
  copy_runfiles(
      dst_dir=jaxlib_dir / "mlir" / "dialects" / "gpu",
      src_files=[
          "_main/jaxlib/mlir/dialects/gpu/__init__.py",
      ],
  )
  copy_runfiles(
      dst_dir=jaxlib_dir / "mlir" / "dialects" / "gpu" / "passes",
      src_files=[
          "_main/jaxlib/mlir/dialects/gpu/passes/__init__.py",
      ],
  )


  if build_utils.is_windows():
    capi_so = "_main/jaxlib/mlir/_mlir_libs/jaxlib_mlir_capi.dll"
  else:
    so_ext = "dylib" if _is_mac() else "so"
    capi_so = f"_main/jaxlib/mlir/_mlir_libs/libjaxlib_mlir_capi.{so_ext}"

  mlir_libs_dir = jaxlib_dir / "mlir" / "_mlir_libs"
  copy_runfiles(
      dst_dir=mlir_libs_dir,
      src_files=[
          capi_so,
          "_main/jaxlib/mlir/_mlir_libs/__init__.py",
          f"_main/jaxlib/mlir/_mlir_libs/_mlir.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_chlo.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_mlirHlo.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_mlirDialectsSparseTensor.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_mlirSparseTensorPasses.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_mosaic_gpu_ext.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_tpu_ext.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_sdy.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_stablehlo.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/register_jax_dialects.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_mlirDialectsGPU.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_mlirDialectsLLVM.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_mlirDialectsNVGPU.{pyext}",
          f"_main/jaxlib/mlir/_mlir_libs/_mlirGPUPasses.{pyext}",
      ]
      + (
          []
          if build_utils.is_windows()
          else [
              f"_main/jaxlib/mlir/_mlir_libs/_triton_ext.{pyext}",
              "_main/jaxlib/mlir/_mlir_libs/_triton_ext.pyi",
          ]
      ),
  )

  triton_dir = jaxlib_dir / "triton"
  copy_runfiles(
      dst_dir=triton_dir,
      src_files=[
          "_main/jaxlib/triton/__init__.py",
          "_main/jaxlib/triton/dialect.py",
      ],
  )
  patch_copy_mlir_import(
      "_main/jaxlib/triton/_triton_enum_gen.py", dst_dir=triton_dir
  )
  patch_copy_mlir_import(
      "_main/jaxlib/triton/_triton_ops_gen.py", dst_dir=triton_dir
  )

  copy_runfiles(
    dst_dir=jaxlib_dir / "include" / "xla" / "ffi" / "api",
    src_files=[
        "xla/xla/ffi/api/c_api.h",
        "xla/xla/ffi/api/api.h",
        "xla/xla/ffi/api/ffi.h",
    ],
  )

tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="jaxlib")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_wheel(
      pathlib.Path(sources_path),
      cpu=args.cpu,
  )
  package_name = "jaxlib"
  if args.editable:
    build_utils.build_editable(sources_path, args.output_path, package_name)
  else:
    git_hash = build_utils.get_githash(args.jaxlib_git_hash)
    build_utils.build_wheel(sources_path, args.output_path, package_name, git_hash=git_hash)
finally:
  if tmpdir:
    tmpdir.cleanup()
