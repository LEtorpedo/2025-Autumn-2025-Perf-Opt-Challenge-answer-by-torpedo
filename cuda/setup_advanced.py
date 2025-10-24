"""
Setup script for CUDA Advanced Random Walk Simulation
编译高级优化的CUDA扩展模块
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess

class CUDAExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['nvcc', '--version'])
        except OSError:
            raise RuntimeError("NVCC not found! Please install CUDA Toolkit")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import pybind11
        
        # 获取Python和pybind11的include路径
        python_include = subprocess.check_output(
            [sys.executable, '-c', 'from distutils.sysconfig import get_python_inc; print(get_python_inc())']
        ).decode().strip()
        
        pybind11_include = pybind11.get_include()
        
        # 源文件
        source_file = 'random_walk_cuda_advanced.cu'
        output_file = 'random_walk_cuda_advanced' + ('.pyd' if sys.platform == 'win32' else '.so')
        
        # 构建nvcc命令（高级优化选项）
        nvcc_cmd = [
            'nvcc',
            '--compiler-options', '-fPIC',
            '-O3',
            '-std=c++17',
            '--shared',
            '-o', output_file,
            source_file,
            f'-I{python_include}',
            f'-I{pybind11_include}',
            '-lcudart',
            '-L/usr/lib/x86_64-linux-gnu',
            '-Xlinker', '-rpath,/usr/lib/x86_64-linux-gnu',
            '--expt-relaxed-constexpr',
            # 高级优化选项
            '-use_fast_math',
            '--generate-code', 'arch=compute_86,code=sm_86',  # RTX 4080
            '--ptxas-options=-v',
            '-maxrregcount=64',  # 限制寄存器使用以提高占用率
            '--extra-device-vectorization',  # 额外的向量化
        ]
        
        print("Compiling CUDA Advanced extension...")
        print(f"Command: {' '.join(nvcc_cmd)}")
        
        try:
            subprocess.check_call(nvcc_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            print(f"✓ Successfully compiled {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Compilation failed: {e}")
            raise

setup(
    name='random_walk_cuda_advanced',
    version='1.0',
    author='Torpedo',
    description='CUDA-accelerated Random Walk (Advanced Optimizations)',
    ext_modules=[CUDAExtension('random_walk_cuda_advanced')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.7',
)

