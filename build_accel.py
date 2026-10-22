#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_accel.py - 构建双加速路径（C++ CPU 版 + GPU 版）

对应审计 P0-1（PERF-1）：
    原生 Makefile / CMake 默认 GPU=0 CUDNN=0 OPENCV=0 AVX=0 OPENMP=0，
    开箱即用的二进制既没有 SIMD 也没有多线程，还读不了常规图像格式。
    本脚本显式开启正确开关，并把产物命名为后端可识别的名字：

        darknet_cpu.dll / libdarknet_cpu.so   <- C++ CPU 版（AVX + OpenMP）
        darknet_gpu.dll / libdarknet_gpu.so   <- GPU 版（CUDA）

    命名区分是关键：这样两个版本可以同时存在，
    backends.py 才能在运行时选择/回退。

构建策略（均经真机验证，细节见各函数内注释）：
    CPU 版   CMake 编译（MinGW Makefiles，OpenMP/AVX）→ 手工静态链接。
             为什么不能直接用 CMake 的链接：OpenMP::OpenMP_CXX 在链接行
             展开为 libgomp.dll.a 导入库【全路径】且排在静态归档之前，
             ld 先从导入库满足 GOMP 符号，任何后补的静态归档都再无符号
             可拉，产物仍依赖 libgomp-1.dll 等运行时 DLL（无 MinGW 的
             机器加载即失败）。手工链接不传 -fopenmp，驱动不会追加
             -lgomp -lpthread，libgomp.a/libwinpthread.a 直接满足符号
             ——PE 链接器只对实际拉到符号的导入库生成 DLL 依赖，产物
             最终只依赖系统 DLL（KERNEL32 / api-ms-win-crt-* / WS2_32）。
    GPU 版   两条路线，自动择优：
              a) 有 nvcc：CMake 标准 CUDA 构建（CuDNN 视探测结果开关）；
              b) 无 nvcc 但有便携工具链（默认 C:/clang_portable +
                 C:/cuda_portable，可用 DARKNET_CLANG_DIR /
                 DARKNET_CUDA_DIR 环境变量覆盖）：clang --cuda 逐文件
                 编译 + 手工链接，CUDA 运行时 DLL 随产物分发到项目根目录
                 （native.py 的 _preload_dll_dir 会把该目录加入 PATH）。
    OpenCV   一律 OFF。图像 I/O 由 zebrafish_quant.imaging 在 Python 侧
             完成（opencv-python wheel 即可满足）；C++ 侧链 OpenCV 只会
             引入 opencv_*.dll 运行时依赖，违背"便携 drop-in"目标。
             注意：opencv-python 只是 Python wheel，不提供
             OpenCVConfig.cmake，本就不是 -DENABLE_OPENCV=ON 的依据。

用法：
    python build_accel.py --dry-run      # 只显示构建计划与环境探测
    python build_accel.py                # 构建当前机器支持的全部变体
    python build_accel.py --only cpu     # 只构建 CPU 版
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))

# 构建所需的最小文件集（用于中文路径下的 ASCII 暂存）
# 注：LICENSE 必须包含 —— CMakeLists 末尾 include(CPack) 会校验
# CPACK_RESOURCE_FILE_LICENSE，缺失则 configure 直接失败。
_STAGE_ITEMS = ("src", "include", "3rdparty", "cmake", "scripts",
                "CMakeLists.txt", "DarknetConfig.cmake.in", "vcpkg.json",
                "LICENSE")

MINGW_BIN = r"C:\mingw64\bin"
MINGW_LIB = r"C:\mingw64\lib"

# 便携工具链（无 CUDA Toolkit/nvcc 时也能构建 GPU 版）。
# 生成方法：libcuda.a 等导入库由 nvcuda.dll 经 gendef + dlltool 制作。
PORTABLE_CLANG_DIR = os.environ.get("DARKNET_CLANG_DIR", r"C:\clang_portable")
PORTABLE_CUDA_DIR = os.environ.get("DARKNET_CUDA_DIR", r"C:\cuda_portable")

# GPU 运行时需要随产物分发的 DLL。cublasLt64_12.dll 最易漏：cublas64_12.dll
# 的导入表里有它，缺了它 LoadLibrary 整条链断在本目录之外，报模糊的
# "or one of its dependencies"。
GPU_RUNTIME_DLLS = ("cudart64_12.dll", "cublas64_12.dll",
                    "cublasLt64_12.dll", "curand64_10.dll")

# GCC 14+ 把下列诊断从警告提升为错误；darknet 的 GPU 代码路径
# （cudaHostAlloc(&float_ptr, ...) 等）依赖旧的宽松行为，显式关掉。
GCC14_COMPAT = ["-Wno-incompatible-pointer-types",
                "-Wno-implicit-function-declaration",
                "-Wno-implicit-int", "-Wno-int-conversion"]


def run(cmd: List[str], *, check: bool = True, timeout: int = 3600,
        cwd: Optional[str] = None,
        env: Optional[dict] = None) -> subprocess.CompletedProcess:
    print("    $ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd or ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def has_opencv_cpp() -> bool:
    """检测 OpenCV **C++ SDK**（CMake find_package(OpenCV) 需要）。

    仅用于环境报告：加速 DLL 恒定以 -DENABLE_OPENCV=OFF 构建
    （图像 I/O 全在 Python 侧完成），C++ SDK 不是构建依据。
    """
    import glob as _g
    d = os.environ.get("OpenCV_DIR", "")
    if d and os.path.exists(os.path.join(d, "OpenCVConfig.cmake")):
        return True
    try:
        r = subprocess.run(["pkg-config", "--modversion", "opencv4"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return True
    except Exception:                                     # noqa: BLE001
        pass
    pats = [r"C:\opencv\build\*\OpenCVConfig.cmake",
            r"C:\tools\opencv\build\*\OpenCVConfig.cmake",
            "/usr/local/lib/cmake/opencv4/OpenCVConfig.cmake",
            "/usr/lib/*/cmake/opencv4/OpenCVConfig.cmake"]
    return any(_g.glob(p) for p in pats)


def detect_cuda_arch() -> str:
    """GPU 编译架构：环境变量优先，否则从 nvidia-smi 探测计算能力。"""
    if os.environ.get("DARKNET_CUDA_ARCH"):
        return os.environ["DARKNET_CUDA_ARCH"]
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            cap = r.stdout.strip().splitlines()[0].strip()
            major, minor = cap.split(".")[:2]
            return f"sm_{major}{minor}"
    except Exception:                                     # noqa: BLE001
        pass
    return "sm_89"          # 默认按 RTX 4060 Laptop（sm_89）


def detect_portable_toolchain() -> dict:
    """探测便携 clang + CUDA 工具链（无 nvcc 时的 GPU 构建路线）。"""
    clang = os.path.join(PORTABLE_CLANG_DIR, "bin", "clang++.exe")
    cuda_inc = os.path.join(PORTABLE_CUDA_DIR, "include")
    cuda_lib = os.path.join(PORTABLE_CUDA_DIR, "lib", "x64")
    ok = (os.path.exists(clang) and os.path.isdir(cuda_inc)
          and os.path.isdir(cuda_lib))
    return {"ok": ok, "clang": clang, "cuda": PORTABLE_CUDA_DIR,
            "cuda_lib": cuda_lib, "arch": detect_cuda_arch()}


def has_cudnn() -> bool:
    """探测 cuDNN 头文件（有 nvcc 的 CMake GPU 路线据此开关 CUDNN）。"""
    cands = []
    nvcc = which("nvcc")
    if nvcc:
        cands.append(os.path.dirname(os.path.dirname(nvcc)))
    if os.environ.get("CUDA_PATH"):
        cands.append(os.environ["CUDA_PATH"])
    return any(os.path.exists(os.path.join(c, "include", "cudnn.h"))
               for c in cands)


def detect_env() -> dict:
    env = {"system": platform.system()}
    env["cmake"] = which("cmake")
    env["mingw32_make"] = which("mingw32-make")
    env["ninja"] = which("ninja")
    env["gcc"] = which("gcc")
    env["nvcc"] = which("nvcc")
    env["opencv_cpp"] = has_opencv_cpp()      # 仅报告，不作为构建依据
    try:
        import cv2  # noqa: F401
        env["opencv_python"] = True            # 仅用于 Python 侧读图
    except Exception:                                     # noqa: BLE001
        env["opencv_python"] = False
    env["gpu"] = False
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        env["gpu"] = (r.returncode == 0 and bool(r.stdout.strip()))
        env["gpu_name"] = (r.stdout.strip().splitlines() or [""])[0]
    except Exception:                                     # noqa: BLE001
        env["gpu_name"] = ""
    env["portable_gpu"] = detect_portable_toolchain()
    env["cudnn"] = has_cudnn()
    env["mingw_root"] = os.path.exists(os.path.join(MINGW_BIN, "g++.exe"))
    return env


def pick_generator(env: dict) -> Optional[str]:
    if env["mingw32_make"]:
        return "MinGW Makefiles"
    if env["ninja"]:
        return "Ninja"
    return None          # 交给 CMake 默认


def stage_ascii_workspace() -> Tuple[str, Optional[str]]:
    """若源码路径含非 ASCII（如中文），复制到纯 ASCII 临时目录再构建。

    实测：MinGW 的 mingw32-make 在中文路径下报
        stat: .../src/darknet.c: Illegal byte sequence
    这是 make 使用 C locale 无法处理多字节路径导致的，与本项目
    「darknet.py 用 .encode('ascii') 处理模型路径」属于同一类问题（缺陷 B-1）。

    返回 (workspace_root, 临时目录 or None)。None 表示无需暂存。
    """
    if ROOT.isascii():
        return ROOT, None

    tmp = tempfile.mkdtemp(prefix="dn_build_")
    dst = os.path.join(tmp, "darknet")
    os.makedirs(dst, exist_ok=True)
    for item in _STAGE_ITEMS:
        s, d = os.path.join(ROOT, item), os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        elif os.path.isfile(s):
            shutil.copy2(s, d)
    print(f"  源码路径含非 ASCII 字符，已暂存到: {dst}")
    return dst, tmp


def lib_name_for(variant: str) -> str:
    if platform.system() == "Windows":
        return f"darknet_{variant}.dll"
    return f"libdarknet_{variant}.so"


def locate_built_lib(build_dir: str) -> Optional[str]:
    """在构建目录里找编译产物（不同生成器/平台命名不同）。

    命名差异：
        MSVC            -> darknet.dll
        MinGW/GCC(Win)  -> libdarknet.dll   （注意有 lib 前缀！）
        Linux           -> libdarknet.so
        macOS           -> libdarknet.dylib
    同时排除 MinGW 的导入库 libdarknet.dll.a（不是运行时库）。
    """
    names = ("darknet.dll", "libdarknet.dll", "libdarknet.so", "libdarknet.dylib")
    cands = []
    for root, _dirs, files in os.walk(build_dir):
        for f in files:
            if f in names and not f.endswith(".dll.a"):
                cands.append(os.path.join(root, f))
    return cands[0] if cands else None


# ---------------------------------------------------------------- 静态链接
def find_mingw_static_archives() -> List[str]:
    """定位 libgomp.a / libwinpthread.a 静态归档。

    两个路径陷阱（都踩过）：
    1) C:/mingw64/lib/gcc/x86_64-w64-mingw32/ 下除版本目录（如 15.2.0）
       外还有一个名为 "lib" 的目录——取排序最后一项会命中 lib/libgomp.a
       （不存在），归档列表静默变空。直接 glob libgomp.a 本身最稳。
    2) libwinpthread.a 在 sysroot 的 x86_64-w64-mingw32/lib/ 下，
       不在 lib/x86_64-w64-mingw32/（后者是 gcc 内部库目录）。
    """
    root = os.path.dirname(MINGW_LIB.rstrip("\\/"))
    gomp = _glob.glob(os.path.join(MINGW_LIB, "gcc", "x86_64-w64-mingw32",
                                   "*", "libgomp.a"))
    archives = [sorted(gomp)[0]] if gomp else []
    for cand in (os.path.join(MINGW_LIB, "x86_64-w64-mingw32",
                              "libwinpthread.a"),
                 os.path.join(root, "x86_64-w64-mingw32", "lib",
                              "libwinpthread.a")):
        if os.path.exists(cand):
            archives.append(cand)      # libwinpthread.a 是叶依赖，排最后
            break
    return [p.replace("\\", "/") for p in archives]


def collect_target_objects(build_dir: str) -> List[str]:
    """收集主库目标（dark）的对象文件。

    优先读 objects*.rsp 权威清单——MinGW Makefiles 的对象扩展名是
    .obj 而非 .o，直接按 .o 扫目标目录会全落空（踩过）；Ninja 不产生
    rsp，退回按 .o/.obj 扫。目标目录通过 link.txt 指向 libdarknet.dll
    识别，避免误收可执行目标（darknet.dir）里的 main.o。
    """
    obj_dir = None
    for d in sorted(_glob.glob(os.path.join(build_dir, "CMakeFiles",
                                            "*.dir"))):
        lt = os.path.join(d, "link.txt")
        if os.path.exists(lt) and "libdarknet.dll" in open(
                lt, encoding="utf-8", errors="replace").read():
            obj_dir = d
            break
    if not obj_dir:
        return []
    objs: List[str] = []
    for rf in sorted(_glob.glob(os.path.join(obj_dir, "objects*.rsp"))):
        with open(rf, encoding="utf-8", errors="replace") as fh:
            objs += [os.path.join(build_dir, p) for p in fh.read().split()]
    if not objs:
        for root_, _d, files in os.walk(obj_dir):
            objs += [os.path.join(root_, f) for f in files
                     if f.endswith((".o", ".obj"))]
    return [o for o in objs if os.path.exists(o)]


def hand_link(out: str, objs: List[str], libs: List[str],
              archives: Optional[List[str]] = None) -> subprocess.CompletedProcess:
    """手工链接：完全控制静态归档的位置。

    为什么需要它（CMake 链接做不到）：
      - CMake 的 OpenMP::OpenMP_CXX 会展开成 libgomp.dll.a 导入库的
        【全路径】且排在目标库之后、语言标准库之前——ld 先从导入库
        满足 GOMP 符号，追加在后面的静态归档再无符号可拉；
      - 驱动 spec（libgomp.spec）还会把 -lgomp -lpthread 追加到行尾。
      手工链接不传 -fopenmp，两个动态来源都不存在，静态归档直接满足
      GOMP_/pthread_ 符号；PE 链接器只对实际拉到符号的导入库生成 DLL
      依赖，于是产物不依赖 libgomp-1.dll / libwinpthread-1.dll。

    归档顺序：libgomp.a 与 libmingwthrd.a 都引用 pthread_*，叶依赖
    libwinpthread.a 必须排最后。
    """
    archives = archives or []
    mingwthrd = os.path.join(os.path.dirname(MINGW_LIB.rstrip("\\/")),
                             "x86_64-w64-mingw32", "lib", "libmingwthrd.a")
    ordered: List[str] = [a for a in archives if "libgomp.a" in a]
    if os.path.exists(mingwthrd):
        ordered.append(mingwthrd)
    ordered += [a for a in archives if "winpthread" in a]
    cmd = [os.path.join(MINGW_BIN, "g++.exe"), "-shared", "-o", out] + objs + \
        ["-static", "-static-libgcc", "-static-libstdc++"] + ordered + libs
    print(f"    $ g++ -shared ... （对象 {len(objs)} 个，归档 "
          f"{len(ordered)} 个，库 {len(libs)} 项）")
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 构建流程
def build_variant(env: dict, variant: str, *, jobs: int,
                  dry_run: bool, workspace: str) -> dict:
    """variant: 'cpu' | 'gpu'

    workspace: 实际执行构建的源码根目录（中文路径时为 ASCII 暂存目录）。
    """
    out = {"variant": variant, "ok": False, "skipped": False,
           "reason": "", "output": None}

    if variant == "gpu":
        if not env["gpu"]:
            out.update(skipped=True, reason="未检测到 NVIDIA GPU，跳过 GPU 构建")
            return out
        # 无 nvcc 但便携工具链可用 -> clang 路线（真机验证路线）
        if not env["nvcc"]:
            if not env["portable_gpu"]["ok"]:
                out.update(skipped=True,
                           reason=("未检测到 nvcc，便携工具链也不可用"
                                   f"（{PORTABLE_CLANG_DIR} / {PORTABLE_CUDA_DIR}），"
                                   "跳过 GPU 构建"))
                return out
            return build_gpu_portable(env, dry_run=dry_run, workspace=workspace)

    # ---------------- CPU：CMake 编译 + 手工静态链接 ----------------
    # ---------------- GPU（有 nvcc）：CMake 标准路线 ----------------
    if variant == "cpu" and not env["mingw_root"]:
        out.update(skipped=True,
                   reason=f"未找到 MinGW 工具链（{MINGW_BIN}），跳过 CPU 构建")
        return out

    gen = pick_generator(env)
    if variant == "cpu":
        gen = "MinGW Makefiles" if env["mingw32_make"] else gen
    build_dir = os.path.join(workspace, "build", f"accel_{variant}")
    cfg = ["cmake", "-S", workspace, "-B", build_dir]
    if gen:
        cfg += ["-G", gen]
    cfg += [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DENABLE_CUDA=" + ("ON" if variant == "gpu" else "OFF"),
        "-DENABLE_CUDNN=" + ("ON" if (variant == "gpu" and env["cudnn"]) else "OFF"),
        "-DENABLE_OPENCV=OFF",          # 图像 I/O 在 Python 侧，见模块 docstring
        "-DENABLE_CUDNN_HALF=" + ("ON" if variant == "gpu" else "OFF"),
    ]
    if variant == "cpu":
        cfg += [
            "-DCMAKE_C_COMPILER=" + os.path.join(MINGW_BIN, "gcc.exe"),
            "-DCMAKE_CXX_COMPILER=" + os.path.join(MINGW_BIN, "g++.exe"),
            "-DCMAKE_MAKE_PROGRAM=" + os.path.join(MINGW_BIN, "mingw32-make.exe"),
            # CMake 自己的链接产物会被弃用（改手工静态链接），此 flag 仅作
            # 双保险：万一手工链接未执行，至少 libgcc/libstdc++ 是静态的
            "-DCMAKE_SHARED_LINKER_FLAGS=-static -static-libgcc -static-libstdc++",
        ]
    if dry_run:
        print(f"  [{variant}] 配置命令: {' '.join(cfg)}")
        if variant == "cpu":
            print(f"  [{variant}] 链接方式: 手工静态链接"
                  f"（归档 {find_mingw_static_archives()}）")
        out.update(ok=True, reason="dry-run")
        return out

    print(f"\n=== 构建 {variant} 版 ===")
    r = run(cfg, check=False, cwd=workspace)
    if r.returncode != 0:
        out["reason"] = "cmake configure 失败:\n" + (r.stderr or r.stdout)[-1500:]
        return out

    r = run(["cmake", "--build", build_dir, "--config", "Release",
             "-j", str(jobs)], check=False, cwd=workspace)
    if r.returncode != 0 and variant == "gpu":
        out["reason"] = "编译失败:\n" + (r.stderr or r.stdout)[-1500:]
        return out
    if r.returncode != 0:
        print("  （CMake 链接步骤失败无碍——CPU 版改用手工静态链接）")

    if variant == "cpu":
        # CMake 只当编译驱动；其链接走动态 libgomp 导入库，产物弃用
        objs = collect_target_objects(build_dir)
        if not objs:
            out["reason"] = ("未找到对象文件（编译失败？）:\n"
                             + (r.stderr or r.stdout)[-1500:])
            return out
        archives = find_mingw_static_archives()
        if not archives:
            out["reason"] = "未找到 MinGW 静态归档（libgomp.a/libwinpthread.a）"
            return out
        print(f"  对象文件 {len(objs)} 个 -> 手工静态链接")
        dst = os.path.join(ROOT, lib_name_for("cpu"))
        r = hand_link(dst, objs, ["-lws2_32", "-lm"], archives)
        if r.returncode != 0:
            out["reason"] = "手工链接失败:\n" + (r.stderr or r.stdout)[-1500:]
            return out
        out.update(ok=True, output=dst,
                   reason="ok（静态链接 MinGW 运行时，零运行时 DLL 依赖）")
        return out

    lib = locate_built_lib(build_dir)
    if not lib:
        out["reason"] = "未找到编译产物"
        return out
    dst = os.path.join(ROOT, lib_name_for("gpu"))
    shutil.copy2(lib, dst)
    out.update(ok=True, output=dst, reason="ok")
    return out


def build_gpu_portable(env: dict, *, dry_run: bool, workspace: str) -> dict:
    """GPU 便携路线：便携 clang + CUDA，无 nvcc 也能构建。

    即 tools_gpu_build_proto.py 的同款路径（产物 darknet_gpu.dll 已真机
    验证：加载 / 对拍 100% / 9.5ms 每图）。要点：
      - clang --cuda 编译 .cu（--cuda-path 指向便携 CUDA，ptxas 等从
        CUDA/bin 经 PATH 提供）；
      - C/C++ 宿主代码用 MinGW gcc/g++ 编译（GCC14_COMPAT 关掉被提升为
        错误的诊断）；
      - 手工链接：-lcuda（驱动 API，导入库由 nvcuda.dll 经 gendef+dlltool
        生成）+ -lcudart -lcublas -lcurand；
      - CUDA 运行时 DLL 拷到项目根目录，_preload_dll_dir 会将其加入 PATH。
    """
    out = {"variant": "gpu", "ok": False, "skipped": False,
           "reason": "", "output": None}
    pt = env["portable_gpu"]
    clang, cuda, cuda_lib, arch = (pt["clang"], pt["cuda"], pt["cuda_lib"],
                                   pt["arch"])
    build_dir = os.path.join(workspace, "build", "gpu_clang")
    os.makedirs(build_dir, exist_ok=True)

    envp = dict(os.environ)
    envp["PATH"] = ";".join([os.path.join(cuda, "bin"),
                             os.path.join(cuda, "dll"), MINGW_BIN,
                             envp.get("PATH", "")])

    inc = ["-Isrc", "-I.", "-Iinclude", "-I3rdparty/stb/include",
           "-I" + os.path.join(cuda, "include")]
    host_flags = ["-Ofast", "-DNDEBUG", "-DGPU", "-w"] + GCC14_COMPAT + inc
    cuda_flags = ["-O3", "-DNDEBUG", "-DGPU", "-w"] + inc

    cu_files = sorted(_glob.glob(os.path.join(workspace, "src", "*.cu")))
    c_files = [f for f in sorted(_glob.glob(os.path.join(workspace, "src",
                                                         "*.c")))
               if os.path.basename(f) not in ("darknet.c", "gettimeofday.c")]
    cpp_files = [os.path.join(workspace, "src", f)
                 for f in ("http_stream.cpp", "image_opencv.cpp",
                           "yolo_v2_class.cpp")
                 if os.path.exists(os.path.join(workspace, "src", f))]

    if dry_run:
        print(f"  [gpu] 便携工具链: clang={clang}")
        print(f"  [gpu]            cuda={cuda} arch={arch}")
        print(f"  [gpu] 源文件: cu={len(cu_files)} c={len(c_files)} "
              f"cpp={len(cpp_files)}")
        out.update(ok=True, reason="dry-run (portable clang+cuda)")
        return out

    def compile_one(cmd: List[str], src: str) -> str:
        o = os.path.join(build_dir, os.path.basename(src) + ".o")
        r = run(cmd + ["-c", "-o", o, src], check=False, cwd=workspace,
                env=envp)
        if r.returncode != 0:
            raise RuntimeError("编译失败: " + src + "\n"
                               + (r.stderr or r.stdout)[-1500:])
        return o

    print(f"\n=== 构建 gpu 版（便携 clang+CUDA，arch={arch}） ===")
    try:
        objs: List[str] = []
        for f in cu_files:
            objs.append(compile_one(
                [clang, "-x", "cuda",
                 "--cuda-path=" + cuda.replace("\\", "/"),
                 "--cuda-gpu-arch=" + arch] + cuda_flags, f))
        for f in c_files:
            objs.append(compile_one(
                [os.path.join(MINGW_BIN, "gcc.exe")] + host_flags, f))
        for f in cpp_files:
            objs.append(compile_one(
                [os.path.join(MINGW_BIN, "g++.exe")] + host_flags, f))
    except RuntimeError as e:
        out["reason"] = str(e)
        return out

    dst = os.path.join(ROOT, lib_name_for("gpu"))
    implib = os.path.join(build_dir, "libdarknet_gpu.dll.a")
    libs = ["-L" + cuda_lib,
            "-lcuda", "-lcudart", "-lcublas", "-lcurand",
            "-lws2_32", "-lpthread", "-lm",
            "-Wl,--out-implib," + implib]
    r = hand_link(dst, objs, libs)
    if r.returncode != 0:
        out["reason"] = "链接失败:\n" + (r.stderr or r.stdout)[-1500:]
        return out

    for dll in GPU_RUNTIME_DLLS:
        src = os.path.join(cuda, "dll", dll)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ROOT, dll))
            print(f"  拷贝运行时 DLL: {dll}")
        else:
            print(f"  [警告] 运行时 DLL 不存在: {src}")

    out.update(ok=True, output=dst, reason="ok（便携 clang+CUDA）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 darknet 双加速路径")
    ap.add_argument("--only", choices=["cpu", "gpu"], help="只构建指定变体")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-workspace", action="store_true",
                    help="构建后保留 ASCII 暂存目录，便于排查")
    args = ap.parse_args()

    env = detect_env()
    print("=== 环境探测 ===")
    for k in ("system", "cmake", "gcc", "mingw32_make", "ninja",
              "nvcc", "cudnn", "opencv_cpp", "opencv_python", "gpu",
              "gpu_name"):
        print(f"  {k:15s}: {env.get(k)}")
    pt = env["portable_gpu"]
    print(f"  {'portable_gpu':15s}: ok={pt['ok']} clang={pt['clang']}")
    print(f"  {'cuda_arch':15s}: {pt['arch']}")

    if not env["cmake"]:
        print("\n[错误] 未检测到 cmake，无法构建。")
        return 2
    if not env["mingw_root"]:
        print(f"\n[警告] 未找到 MinGW 工具链（{MINGW_BIN}），CPU 版将跳过。")

    print("  （OpenCV 恒定 OFF：图像 I/O 由 zebrafish_quant.imaging 在 Python")
    print("    侧完成；opencv-python 只是 wheel，不满足 find_package(OpenCV)，")
    print("    C++ 侧链 OpenCV 只会引入 opencv_*.dll 运行时依赖。）")

    workspace, staged_tmp = stage_ascii_workspace()
    variants = [args.only] if args.only else ["cpu", "gpu"]
    results = []
    try:
        for v in variants:
            results.append(build_variant(env, v, jobs=args.jobs,
                                         dry_run=args.dry_run,
                                         workspace=workspace))
    finally:
        all_ok = all(r["ok"] for r in results)
        if staged_tmp and (all_ok and not args.keep_workspace):
            shutil.rmtree(staged_tmp, ignore_errors=True)
            print(f"  （已清理暂存目录 {staged_tmp}）")
        elif staged_tmp:
            # 失败时保留现场，方便人工排查编译问题
            print(f"  （保留暂存目录以便排查: {staged_tmp}）")

    print("\n=== 构建结果 ===")
    for r in results:
        tag = "OK  " if r["ok"] else ("跳过" if r["skipped"] else "失败")
        print(f"  [{tag}] {r['variant']:4s} -> {r.get('output') or r['reason']}")

    return 0 if any(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
