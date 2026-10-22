#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_cpu_build_static.py - 用静态链接重建 darknet_cpu.dll

背景：原 darknet_cpu.dll 动态依赖 4 个 MinGW 运行时 DLL
（libgcc_s_seh-1 / libstdc++-6 / libgomp-1 / libwinpthread-1），
这些 DLL 位于 C:/mingw64/bin，目标机器 PATH 中没有 → CPU 回退路径
加载即失败（"Could not find module ... or one of its dependencies"）。

修复：编译走 CMake（复用其 include/define/flag），链接手工完成——
加 -static -static-libgcc -static-libstdc++ 并把 libgomp.a /
libwinpthread.a 静态归档直接放对象之后，MinGW 运行时全部打进
DLL 本体，产物只依赖系统 DLL（KERNEL32 / api-ms-win-crt-* /
WS2_32）。不能依赖 CMake 链接：OpenMP::OpenMP_CXX 会展开为
libgomp.dll.a 导入库全路径且排在归档之前（详见下方注释）。

安全性：已扫描 venv site-packages 全部 DLL 导入表，没有任何 DLL
导入 libgomp/libwinpthread/libgcc，静态 OpenMP 不会与 numpy 的
OpenBLAS 形成双 OpenMP 运行时冲突。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
_STAGE_ITEMS = ("src", "include", "3rdparty", "cmake", "scripts",
                "CMakeLists.txt", "DarknetConfig.cmake.in", "vcpkg.json",
                "LICENSE")

MINGW_BIN = r"C:\mingw64\bin"
MINGW_LIB = r"C:\mingw64\lib"

import glob as _glob

# MinGW 静态归档定位。
# 坑1：C:/mingw64/lib/gcc/x86_64-w64-mingw32/ 下除版本目录（15.2.0）
# 外还有一个名为 "lib" 的目录，取排序最后一项会命中 lib/libgomp.a
# （不存在）→ 归档列表为空。直接 glob libgomp.a 本身最稳。
# 坑2：归档必须放链接行【对象之后】。驱动 spec（libgomp.spec）会把
# -lgomp -lpthread 追加到行尾（动态导入库）；若静态 .a 在对象之前
# （CMAKE_SHARED_LINKER_FLAGS 的位置），无待决符号可满足，成员根本
# 不会被拉入。正确载体是 CMAKE_CXX_STANDARD_LIBRARIES（行尾）：
# 归档先满足 GOMP_/pthread_ 符号，随后导入库无符号可拉，PE 链接器
# 不生成 libgomp-1/libwinpthread-1 依赖。
_GCC_GLOB = _glob.glob(os.path.join(MINGW_LIB, "gcc", "x86_64-w64-mingw32",
                                    "*", "libgomp.a"))
_STATIC_ARCHIVES = [sorted(_GCC_GLOB)[0]] if _GCC_GLOB else []
# libwinpthread.a 在 sysroot 的 x86_64-w64-mingw32/lib/ 下，
# 不在 lib/x86_64-w64-mingw32/（后者是 gcc 内部库目录）
_MINGW_ROOT = os.path.dirname(MINGW_LIB.rstrip("\\/"))
_WINPTHREAD_CANDS = [
    os.path.join(MINGW_LIB, "x86_64-w64-mingw32", "libwinpthread.a"),
    os.path.join(_MINGW_ROOT, "x86_64-w64-mingw32", "lib", "libwinpthread.a"),
]
for _wp in _WINPTHREAD_CANDS:
    if os.path.exists(_wp):
        _STATIC_ARCHIVES.append(_wp)   # 顺序：libgomp.a 在前
        break
_STATIC_ARCHIVES = [p.replace("\\", "/") for p in _STATIC_ARCHIVES]
print(f"静态归档: {_STATIC_ARCHIVES}")

# 链接策略（经 rsp 解剖验证，见下）：
# 1) -static -static-libgcc -static-libstdc++ 消除 libgcc_s_seh-1/libstdc++-6。
# 2) libgomp/libwinpthread：CMake 的 OpenMP::OpenMP_CXX 在链接行展开为
#    libgomp.dll.a / libpthread.dll.a 导入库【全路径】，且排在目标库之后、
#    语言标准库之前——ld 先从导入库满足 GOMP 符号，任何追加在后面的静态
#    归档都再无符号可拉。CMAKE_CXX_STANDARD_LIBRARIES 路线因此无效
#    （且 ; 分隔在 MinGW 响应文件里不被解析，直接构 built 失败）。
#    解法：CMake 只当编译驱动（复用其 include/define/flag），链接完全自主：
#    不传 -fopenmp → 驱动不追加 -lgomp -lpthread → 静态归档直接满足符号。
#    这与 tools_gpu_build_proto.py 的 GPU 手工链接是同一条已验证路径。
STATIC_LINK_FLAGS = "-static -static-libgcc -static-libstdc++"


def run(cmd, cwd=None, timeout=3600):
    print("    $ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="dn_cpu_static_")
    dst = os.path.join(tmp, "darknet")
    os.makedirs(dst, exist_ok=True)
    try:
        for item in _STAGE_ITEMS:
            s, d = os.path.join(ROOT, item), os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            elif os.path.isfile(s):
                shutil.copy2(s, d)
        print(f"暂存源码 -> {dst}")

        build_dir = os.path.join(dst, "build", "accel_cpu")
        cfg = [
            "cmake", "-S", dst, "-B", build_dir,
            "-G", "MinGW Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_C_COMPILER=" + os.path.join(MINGW_BIN, "gcc.exe"),
            "-DCMAKE_CXX_COMPILER=" + os.path.join(MINGW_BIN, "g++.exe"),
            "-DCMAKE_MAKE_PROGRAM=" + os.path.join(MINGW_BIN, "mingw32-make.exe"),
            "-DENABLE_CUDA=OFF",
            "-DENABLE_CUDNN=OFF",
            "-DENABLE_OPENCV=OFF",
            "-DENABLE_CUDNN_HALF=OFF",
            # 关键：静态链接 MinGW 运行时，消除运行时 DLL 依赖
            "-DCMAKE_SHARED_LINKER_FLAGS=" + STATIC_LINK_FLAGS,
        ]
        r = run(cfg)
        if r.returncode != 0:
            print("configure 失败:\n" + (r.stderr or r.stdout)[-3000:])
            return 1
        print("configure OK")

        # 诊断：打印实际链接行，确认归档位置在对象之后、-lgomp 之前
        for root, _d, files in os.walk(build_dir):
            if "link.txt" in files:
                lt = os.path.join(root, "link.txt")
                print(f"\n链接行 ({os.path.relpath(lt, build_dir)}):")
                print(open(lt, encoding="utf-8", errors="replace").read()[:3000])
                break

        # CMake 只当编译驱动；其链接走动态 libgomp 导入库，产物弃用。
        # 容忍链接步骤失败——只要对象文件齐全，手工重链即可。
        r = run(["cmake", "--build", build_dir, "--config", "Release",
                 "-j", str(os.cpu_count() or 4)])
        if r.returncode != 0:
            print("（CMake 链接步骤失败无碍——改用手工静态链接）")

        # 定位主库目标的对象目录：含 link.txt 且指向 libdarknet.dll
        # 的那个 .dir（避免误收可执行目标里的 main.o）
        obj_dir = None
        for d in sorted(_glob.glob(os.path.join(build_dir, "CMakeFiles",
                                                "*.dir"))):
            lt = os.path.join(d, "link.txt")
            if os.path.exists(lt) and "libdarknet.dll" in open(
                    lt, encoding="utf-8", errors="replace").read():
                obj_dir = d
                break
        objs = []
        if obj_dir:
            # 对象清单以 objects*.rsp 为权威（MinGW Makefiles 的对象
            # 扩展名是 .obj 而非 .o，直接按扩展名扫会全落空）
            for rf in sorted(_glob.glob(os.path.join(obj_dir,
                                                     "objects*.rsp"))):
                with open(rf, encoding="utf-8", errors="replace") as fh:
                    objs += [os.path.join(build_dir, p)
                             for p in fh.read().split()]
            if not objs:      # 兜底：直接扫（兼容两种扩展名）
                for root, _d, files in os.walk(obj_dir):
                    objs += [os.path.join(root, f) for f in files
                             if f.endswith((".o", ".obj"))]
        objs = [o for o in objs if os.path.exists(o)]
        objs.sort()
        if not objs:
            print("未找到对象文件（编译可能失败）:\n" +
                  (r.stderr or r.stdout)[-3000:])
            return 1
        print(f"对象文件 {len(objs)} 个 -> 手工静态链接")

        # 手工链接：不传 -fopenmp，驱动不会追加 -lgomp -lpthread；
        # 静态归档直接满足 GOMP_/pthread_ 符号。归档顺序有讲究：
        # libgomp.a 与 libmingwthrd.a 都引用 pthread_*，叶依赖
        # libwinpthread.a 必须排最后。
        out = os.path.join(ROOT, "darknet_cpu.dll")
        _MINGWTHRD = os.path.join(_MINGW_ROOT, "x86_64-w64-mingw32", "lib",
                                  "libmingwthrd.a")
        _gomp = next((a for a in _STATIC_ARCHIVES if "libgomp.a" in a), None)
        _wp = next((a for a in _STATIC_ARCHIVES if "winpthread" in a), None)
        archives = ([_gomp] if _gomp else [])
        if os.path.exists(_MINGWTHRD):
            archives.append(_MINGWTHRD)
        if _wp:
            archives.append(_wp)
        link = [os.path.join(MINGW_BIN, "g++.exe"), "-shared", "-o", out] + \
            objs + ["-static", "-static-libgcc", "-static-libstdc++"] + \
            archives + ["-lws2_32", "-lm"]
        r = run(link)
        if r.returncode != 0:
            print("手工链接失败:\n" + (r.stderr or r.stdout)[-3000:])
            return 1
        print(f"产物 -> {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    finally:
        if os.environ.get("DN_KEEP_TMP"):
            print(f"保留暂存目录: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    # 依赖检查（objdump 不在 PATH 时跳过，由外部验证）
    import shutil as _sh
    objdump = _sh.which("objdump")
    if not objdump:
        print("\n[警告] 未找到 objdump，跳过依赖检查")
        return 0
    r = subprocess.run([objdump, "-p", out], capture_output=True, text=True,
                       errors="replace")
    deps = [l.split("DLL Name:")[1].strip()
            for l in (r.stdout or "").splitlines() if "DLL Name:" in l]
    print("\n依赖 DLL:")
    for d in deps:
        print(f"  {d}")
    bad = [d for d in deps if d.startswith(("libgcc", "libstdc++", "libgomp",
                                            "libwinpthread"))]
    if bad:
        print(f"\n[失败] 仍依赖 MinGW 运行时: {bad}")
        return 1
    print("\n[OK] 仅依赖系统 DLL，MinGW 运行时依赖已消除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
