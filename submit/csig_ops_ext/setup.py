from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)


def _cmake_cuda_architectures(torch_arch_list: str) -> str:
    """Translate Torch arch list syntax such as 7.0+PTX to CMake CUDA_ARCHITECTURES."""
    out: list[str] = []
    for raw in torch_arch_list.replace(";", " ").split():
        item = raw.strip()
        if not item:
            continue
        with_ptx = item.upper().endswith("+PTX")
        if with_ptx:
            item = item[:-4]
        digits = item.replace(".", "")
        if not digits.isdigit():
            raise ValueError(f"Unsupported CSIG_CUDA_ARCH_LIST entry: {raw!r}")
        out.append(f"{digits}-real")
        if with_ptx:
            out.append(f"{digits}-virtual")
    if not out:
        raise ValueError("CSIG_CUDA_ARCH_LIST produced no CUDA architectures")
    return ";".join(out)


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        outdir = ext_fullpath.parent
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        srcdir = Path(__file__).parent.resolve()
        builddir = Path(self.build_temp).resolve()
        builddir.mkdir(parents=True, exist_ok=True)
        cache_file = builddir / "CMakeCache.txt"
        cache_dir = builddir / "CMakeFiles"
        if cache_file.exists():
            cache_file.unlink()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        env = os.environ.copy()
        # 服务器是 V100(sm70)；本地 sm86 覆盖用 CSIG_CUDA_ARCH_LIST="8.6+PTX"。
        torch_arch_list = env.get("CSIG_CUDA_ARCH_LIST", "7.0+PTX")
        env["TORCH_CUDA_ARCH_LIST"] = torch_arch_list
        cmake_args = [
            "cmake",
            "-S",
            str(srcdir),
            "-B",
            str(builddir),
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={outdir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DEXT_SUFFIX={ext_suffix}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={_cmake_cuda_architectures(torch_arch_list)}",
        ]
        # nvcc 不在 PATH 时（服务器就是这种情况）由 CMAKE_CUDA_COMPILER env 指定绝对路径。
        nvcc = env.get("CMAKE_CUDA_COMPILER", "").strip()
        if nvcc:
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
        subprocess.check_call(cmake_args, env=env)
        jobs = env.get("CSIG_BUILD_PARALLEL", "16").strip()
        subprocess.check_call(
            ["cmake", "--build", str(builddir), *(["-j", jobs] if jobs else ["-j"])], env=env
        )


setup(
    name="custom_op",
    version="0.0.1",
    packages=["custom_op"],
    ext_modules=[CMakeExtension("custom_op._C")],
    cmdclass={"build_ext": CMakeBuild},
)
