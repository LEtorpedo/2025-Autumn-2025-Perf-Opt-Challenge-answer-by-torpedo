"""
C++ 模块编译脚本

使用pybind11将C++代码编译为Python可调用的扩展模块

编译命令:
    python setup.py build_ext --inplace

这将生成:
    - random_walk_cpp_omp.*.so (OpenMP版本)
    - random_walk_cpp_simd.*.so (SIMD优化版本)
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import sys

# 魔鬼级编译选项（来自参考报告）
cpp_args_aggressive = [
    '-O3',              # 最高优化级别
    '-march=native',    # 使用本机CPU所有指令集
    '-flto',            # 链接时优化
    '-ffast-math',      # 快速数学（牺牲一些IEEE标准）
    '-fopenmp',         # 启用OpenMP
    '-std=c++17',       # 使用C++17标准
]

# 保守编译选项（如果激进选项失败，使用这个）
cpp_args_conservative = [
    '-O3',
    '-fopenmp',
    '-std=c++17',
]

# SIMD特定选项
cpp_args_simd = cpp_args_aggressive + [
    '-mavx2',           # 明确启用AVX2
    '-mfma',            # 启用FMA指令
]

# 链接选项
link_args = [
    '-fopenmp',
]

# 检测系统是否支持-march=native
import subprocess
def supports_march_native():
    try:
        result = subprocess.run(
            ['g++', '-march=native', '-x', 'c++', '-', '-o', '/dev/null'],
            input=b'int main() { return 0; }',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except:
        return False

# 根据系统选择编译选项
if not supports_march_native():
    print("Warning: -march=native not supported, using conservative options")
    cpp_args_aggressive = cpp_args_conservative
    cpp_args_simd = cpp_args_conservative + ['-mavx2']

ext_modules = [
    # OpenMP版本
    Pybind11Extension(
        'random_walk_cpp_omp',
        ['simulate_cpp_omp.cpp'],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=cpp_args_aggressive,
        extra_link_args=link_args,
    ),
    
    # SIMD版本
    Pybind11Extension(
        'random_walk_cpp_simd',
        ['simulate_cpp_simd.cpp'],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=cpp_args_simd,
        extra_link_args=link_args,
    ),
]

setup(
    name='random_walk_cpp',
    version='1.0',
    author='Your Name',
    description='Random Walk Simulation - C++ Optimized Implementations',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.7',
)

