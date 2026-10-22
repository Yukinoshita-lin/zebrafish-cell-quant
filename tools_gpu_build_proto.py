#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 Yukinoshita-lin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""原型：用便携 clang+CUDA 工具链手工构建 darknet_gpu.dll"""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build", "gpu_clang")

CLANG = r"C:\clang_portable\bin\clang++.exe"
CUDA = r"C:\cuda_portable"
MINGW = r"C:\mingw64\bin"
CUDA_ARCH = os.environ.get("DARKNET_CUDA_ARCH", "sm_89")

ENV_PATH = ";".join([
    os.path.join(CUDA, "bin"), os.path.join(CUDA, "dll"), MINGW,
    os.environ.get("PATH", ""),
])

ENV = dict(os.environ)
ENV["PATH"] = ENV_PATH

# GCC 14+ 把下列诊断从警告提升为错误；darknet 的 GPU 代码路径
# （cudaHostAlloc(&float_ptr, ...) 等）依赖旧的宽松行为，显式关掉。
GCC14_COMPAT = ["-Wno-incompatible-pointer-types",
                "-Wno-implicit-function-declaration",
                "-Wno-implicit-int", "-Wno-int-conversion"]

HOST_CFLAGS = ["-Ofast", "-DNDEBUG", "-DGPU", "-w"] + GCC14_COMPAT + [
    "-Isrc", "-I.", "-Iinclude", "-I3rdparty/stb/include",
    "-I" + os.path.join(CUDA, "include")]
CUDA_CFLAGS = ["-O3", "-DNDEBUG", "-DGPU", "-w",
               "-Isrc", "-I.", "-Iinclude", "-I3rdparty/stb/include",
               "-I" + os.path.join(CUDA, "include")]


def run(cmd):
    print("    $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit("FAILED: " + " ".join(cmd))
    return r


def main():
    os.makedirs(BUILD, exist_ok=True)
    cu_files = sorted(glob.glob(os.path.join("src", "*.cu")))
    c_files = [f for f in sorted(glob.glob(os.path.join("src", "*.c")))
               if os.path.basename(f) not in ("darknet.c", "gettimeofday.c")]
    cpp_files = ["src/http_stream.cpp", "src/image_opencv.cpp",
                 "src/yolo_v2_class.cpp"]
    cpp_files = [f for f in cpp_files if os.path.exists(os.path.join(ROOT, f))]

    objs = []
    print("=== CUDA sources (%d) ===" % len(cu_files))
    for f in cu_files:
        o = os.path.join(BUILD, os.path.basename(f) + ".o")
        run([CLANG, "-x", "cuda", "--cuda-path=" + CUDA.replace("\\", "/"),
             "--cuda-gpu-arch=" + CUDA_ARCH] + CUDA_CFLAGS +
            ["-c", "-o", o, f])
        objs.append(o)
    print("=== C sources (%d) ===" % len(c_files))
    for f in c_files:
        o = os.path.join(BUILD, os.path.basename(f) + ".o")
        run([os.path.join(MINGW, "gcc.exe")] + HOST_CFLAGS +
            ["-c", "-o", o, f])
        objs.append(o)
    print("=== C++ sources (%d) ===" % len(cpp_files))
    for f in cpp_files:
        o = os.path.join(BUILD, os.path.basename(f) + ".o")
        run([os.path.join(MINGW, "g++.exe")] + HOST_CFLAGS +
            ["-c", "-o", o, f])
        objs.append(o)

    print("=== LINK ===")
    out = os.path.join(ROOT, "darknet_gpu.dll")
    # -static：把 libgcc/libstdc++/libwinpthread 静态进 DLL。
    # 否则产物依赖 C:/mingw64/bin 下的 libgcc_s_seh-1.dll 等，
    # 而运行时 PATH 里没有该目录，ctypes.CDLL 直接加载失败
    # （"or one of its dependencies"），CPU/GPU 双路径全部不可用。
    link = [os.path.join(MINGW, "g++.exe"), "-shared", "-o", out] + objs + [
        "-static", "-static-libgcc", "-static-libstdc++",
        "-L" + os.path.join(CUDA, "lib", "x64"),
        # cuda = 驱动 API（dark_cuda.c 用 cuCtxGetCurrent 等），
        # 导入库由 nvcuda.dll 经 gendef + dlltool 生成
        "-lcuda", "-lcudart", "-lcublas", "-lcurand",
        "-lws2_32", "-lpthread", "-lm",
        "-Wl,--out-implib," + os.path.join(BUILD, "libdarknet_gpu.dll.a"),
    ]
    run(link)
    print("\nOK -> " + out)

    # 运行时依赖 DLL 拷到产物同目录（_preload_dll_dir 会将该目录加入 PATH）。
    # cublasLt64_12.dll 容易被漏掉：cublas64_12.dll 的导入表里有它，
    # 缺了它 LoadLibrary 整条链断在本目录之外，报模糊的
    # "or one of its dependencies"。
    for dll in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                "curand64_10.dll"):
        src = os.path.join(CUDA, "dll", dll)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(ROOT, dll))
            print("  copied runtime dll: " + dll)


if __name__ == "__main__":
    sys.exit(main())
