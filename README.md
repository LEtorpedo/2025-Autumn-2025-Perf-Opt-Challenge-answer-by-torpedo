# 随机游走模拟优化考核

## 考核简介

随机游走模型常在计算机领域用于研究系统动态，随着问题规模增大，如何提升模拟效率成为突破计算优化的关键。

本考核要求在 Ubuntu 22.04 环境下，对 L×L 网格上 N 个粒子进行 T 步随机游走模拟（L=512，N=100000，T=1000）。粒子随机选方向、允许多粒子同格、网格采用周期性边界。需用纯 Python 实现串行基线，并通过不限手段优化，缩短运行时间。最终输出粒子在中心区域平均停留比例与模拟时间，可视化可加分，考查并行计算与优化能力。

## 考核目标

本考核要求您在虚拟机中安装 Ubuntu 22.04 并配置，在 L×L 网格上模拟 N 个相互独立的粒子进行 T 步随机游走，统计所有粒子在中心区域的平均停留比例，并优化模拟速度。

## 考核题流程

### 1. 安装虚拟机
- 安装 Ubuntu 22.04 LTS
- 配置网络连接

### 2. 配置编程环境
- 选择合适版本的Python解释器（要求Python 3.8+）
- 安装必要的开发工具

### 3. 考核题场景

模拟规则：
- 每个粒子每步独立随机选择上下左右四个方向之一（等概率1/4）
- 允许多个粒子占据同一格子（无碰撞约束）
- 网格采用周期性边界条件（从边界走出会从对面边界进入）
- 中心区域定义为：x ∈ [L//4, 3*L//4) 且 y ∈ [L//4, 3*L//4)
- 粒子初始位置随机分布在整个L×L网格上
- 坐标范围：x, y ∈ [0, L-1]

### 4. 考核要求

#### 基础要求
- 用纯 Python 实现串行版本（baseline）
- 优化目标：固定 L=512, N=100000, T=1000，总运行时间最短
- 不限优化手段

#### 可选加分项
- **可视化**：绘制最终粒子分布热力图
- **深度优化**：展示多种优化思路和实现
- **性能分析**：详细的性能对比和分析

#### 核心能力考查
- 理解并行计算思想和优化理念
- 掌握Python性能优化技术

### 5. 优化方向建议

- **算法优化**：向量化计算、减少循环嵌套
- **数据结构优化**：选择合适的数据结构
- **并行计算**：多进程、多线程
- **编译优化**：Numba JIT、Cython
- **使用其他语言重写**：C/C++、Rust等
- **调用高速计算库**：NumPy、CuPy等

### 6. 输入输出格式

#### 输入
```bash
python simulate.py 512 100000 1000
#可以不严格遵守，你同样可以使用 bash 脚本来辅助运行
```

参数说明：
- `<L>`：网格边长（512）
- `<N>`：粒子数量（100000）
- `<T>`：模拟步数（1000）

#### 输出
```
Average dwell ratio: 0.2501
Simulation time: 2.34s
```

输出说明：
- **dwell ratio** = 所有粒子在中心区域的总步数 / (N × T)
- **Simulation time** = 模拟运行时间（秒）

### 7. 实验报告要求

本考核实验报告应应用 LaTeX 或 Markdown 语法撰写：

#### 报告内容要求
- **有图有真相**：希望你可视化的展示你的结果
- **行文流畅**：能正常的描述出你的整个探索过程
- **技术深度**：展示对优化技术的理解和应用

## 提交要求

### 必须提交
- **主程序**：`simulate.py` 或等效的可执行程序
- **基线实现**：纯Python串行版本
- **优化版本**：至少一个优化实现
- **实验报告**：PDF格式，包含算法说明、优化分析、性能对比
- **运行说明**：如何在Ubuntu 22.04环境下运行你的代码

### 可选提交
- 可视化结果图片
- 性能测试脚本
- requirements.txt（如使用第三方库）
- 其他辅助文件

## 参考资源

### Python优化相关
- [NumPy官方文档](https://numpy.org/doc/)
- [Numba用户手册](https://numba.pydata.org/)
- [Python性能优化指南](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)

### 并行计算相关
- [Python多进程编程](https://docs.python.org/3/library/multiprocessing.html)
- [Python多线程编程](https://docs.python.org/3/library/threading.html)
- [joblib并行计算库](https://joblib.readthedocs.io/)

### 可视化相关
- [Matplotlib用户指南](https://matplotlib.org/stable/users/index.html)
- [Seaborn统计可视化](https://seaborn.pydata.org/)

### 性能分析工具
- [cProfile性能分析](https://docs.python.org/3/library/profile.html)
- [line_profiler逐行分析](https://github.com/pyutils/line_profiler)

## 常见问题

### Q: 可以使用第三方库吗？
A: 可以，但需要在requirements.txt中声明依赖。鼓励使用各种优化库和工具。

### Q: 必须在Ubuntu 22.04上运行吗？
A: 是的，这是考核环境要求。最终提交的代码需要能在Ubuntu 22.04上正常运行。

### Q: 如何验证结果的正确性？
A: 重点关注算法逻辑的正确性，dwell ratio的数值会因随机性有所不同，但应该在合理范围内。建议验证方法：
- 检查边界条件是否正确处理（周期性边界）
- 验证中心区域判断逻辑是否准确
- 测试小规模参数确保基本逻辑正确
- 对于标准参数(L=512, N=100000, T=1000)，dwell ratio通常在0.22-0.28范围内

### Q: 优化程度有上限吗？
A: 没有上限，鼓励大胆尝试各种优化手段，包括但不限于算法优化、并行计算、编译优化等。

---

## 本项目参考实现

本仓库提供了一套**完整的多层次优化参考实现**，展示了从纯 Python 到 GPU 加速的性能优化路径。

### 已实现的优化方案

| 实现方案 | 文件位置 | 技术栈 | 实测时间 | 实测加速比 |
|---------|---------|--------|---------|-----------|
| **Baseline** | `src/baseline/baseline.py` | 纯 Python | 40.33s | 1.00× (基准) |
| **NumPy** | `src/numpy_process/numpy_process.py` | NumPy 向量化 | 1.84s | **21.96×** |
| **Numba** | `src/numba_jit/numba_jit.py` | Numba JIT + 并行 | 1.66s | **24.31×** |
| **C++ OpenMP** | `cpp/simulate_cpp_omp.cpp` | C++ + OpenMP | 2.57s | **15.70×** |
| **C++ SIMD** | `cpp/simulate_cpp_simd.cpp` | C++ + AVX2 + OpenMP | 0.82s | **49.47×** |
| **CUDA Basic (Kernel)** | `cuda/random_walk_cuda.cu` | CUDA GPU (仅核函数) | 0.0013s | **31,937×** |
| **CUDA Basic (Transfer)** | `cuda/random_walk_cuda.cu` | CUDA GPU (含传输) | 0.0026s | **15,571×** |
| **CUDA Advanced (Kernel)** | `cuda/random_walk_cuda_advanced.cu` | CUDA + 块级归约 (仅核函数) | 0.0006s | **68,355×** |
| **CUDA Advanced (Transfer)** | `cuda/random_walk_cuda_advanced.cu` | CUDA + 块级归约 (含传输) | 0.0019s | **21,512×** |

> 📊 **测试环境**: Ubuntu 22.04, AMD Ryzen (16核), NVIDIA RTX 4080  
> 📅 **测试时间**: 2025-10-24  
> 🎯 **测试参数**: L=512, N=100,000, T=1,000

### 核心特性

- ✅ **完整的优化路径**：从基础到极致的性能提升演示
- ✅ **统一测试框架**：`benchmark.py` 一键对比所有实现
- ✅ **公平的性能测试**：只测量核心计算时间，排除初始化开销
- ✅ **可视化支持**：自动生成性能对比图表和粒子分布热力图
- ✅ **详细注释**：每个实现都有清晰的代码注释和说明

---

## 快速开始

### 环境要求

**基础环境**（必需）：
- Ubuntu 22.04 LTS（推荐）或其他 Linux 发行版
- Python 3.8+
- pip 包管理器

**可选环境**（用于高级优化）：
- GCC/G++ 编译器（C++ 实现）
- CUDA Toolkit 12.0+（GPU 加速）
- NVIDIA GPU with Compute Capability 7.0+（GPU 加速）

### 安装依赖

```bash
# 克隆或下载本仓库
git clone https://github.com/LEtorpedo/2025-Autumn-2025-Perf-Opt-Challenge-answer-by-torpedo.git
cd 2025-Autumn-2025-Perf-Opt-Challenge-answer-by-torpedo

# 安装 Python 依赖
pip install -r requirements.txt
```

### 运行示例

#### 1. 运行单个实现

**Baseline（纯 Python）**：
```bash
python src/baseline/baseline.py
```

**NumPy 优化版本**：
```bash
python src/numpy_process/numpy_process.py
```

**Numba 优化版本**：
```bash
python src/numba_jit/numba_jit.py
```

#### 2. 运行完整性能测试

```bash
python benchmark.py
```

这将测试所有实现并生成：
- 终端输出：性能统计表格
- `performance_benchmark.png`：性能对比图表（PNG 格式）
- `performance_benchmark.pdf`：性能对比图表（PDF 格式）
- `particle_heatmap_*.png`：粒子分布热力图

#### 3. 编译 C++ 实现（可选）

```bash
cd cpp
python setup.py build_ext --inplace
cd ..
```

编译成功后会生成 `.so` 文件，可被 Python 调用。

#### 4. 编译 CUDA 实现（可选）

```bash
cd cuda
python setup.py build_ext --inplace
cd ..
```

**注意**：需要安装 CUDA Toolkit 和 pybind11。

### 修改模拟参数

编辑 `src/config.py` 文件：

```python
L = 512       # 网格边长
N = 100000    # 粒子数量
T = 1000      # 时间步数

# 测试重复次数
BENCHMARK_REPEATS = {
    'baseline': 10,
    'numpy': 1000,
    'numba': 1000,
    'cpp': 1000,
    'cuda': 1000,
}
```

---

## 项目结构

```
2025-Autumn-2025-Perf-Opt-Challenge-answer-by-torpedo/
├── README.md                           # 本文档
├── LICENSE                             # MIT 许可证
├── requirements.txt                    # Python 依赖列表
├── benchmark.py                        # 统一性能测试脚本
│
├── src/                                # 源代码目录
│   ├── config.py                       # 统一参数配置
│   ├── visualization.py                # 可视化工具
│   ├── baseline/
│   │   └── baseline.py                 # Baseline 实现
│   ├── numpy_process/
│   │   └── numpy_process.py            # NumPy 优化实现
│   └── numba_jit/
│       └── numba_jit.py                # Numba JIT 实现
│
├── cpp/                                # C++ 实现
│   ├── simulate_cpp_omp.cpp            # C++ OpenMP 版本
│   ├── simulate_cpp_simd.cpp           # C++ SIMD 优化版本
│   └── setup.py                        # C++ 编译脚本
│
└── cuda/                               # CUDA GPU 实现
    ├── random_walk_cuda.cu             # CUDA 主实现（Python 绑定）
    ├── kernel.cu                       # CUDA 核函数（独立版本）
    ├── kernel_advanced.cu              # 高级 CUDA 优化
    ├── main.cpp                        # CUDA 主机端代码
    ├── Makefile                        # CUDA 编译脚本
    └── setup.py                        # CUDA Python 绑定编译脚本
```

---

## 复现方法

### 方法一：直接运行（推荐）

适用于只想测试 Python 实现（Baseline、NumPy、Numba）：

```bash
# 1. 安装依赖
pip install numpy numba matplotlib tqdm

# 2. 运行性能测试
python benchmark.py
```

**说明**：会自动跳过 C++ 和 CUDA 实现（如果未编译）。

### 方法二：完整复现（包含 C++）

```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y build-essential python3-dev

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 编译 C++ 模块
cd cpp
python setup.py build_ext --inplace
cd ..

# 4. 运行完整测试
python benchmark.py
```

### 方法三：完整复现（包含 CUDA）

**前置条件**：
- NVIDIA GPU（推荐 RTX 系列或更高）
- CUDA Toolkit 12.0+

```bash
# 1. 验证 CUDA 环境
nvidia-smi              # 查看 GPU 信息
nvcc --version          # 查看 CUDA 版本

# 2. 安装 Python 依赖（包含 pybind11）
pip install -r requirements.txt

# 3. 编译 C++ 模块
cd cpp
python setup.py build_ext --inplace
cd ..

# 4. 编译 CUDA 模块
cd cuda
python setup.py build_ext --inplace
cd ..

# 5. 运行完整测试
python benchmark.py
```

### 实际输出示例

```
✓ Loaded system libstdc++: /usr/lib/x86_64-linux-gnu/libstdc++.so.6
Benchmark started at: 2025-10-24 19:44:46
Log file: results/benchmark_20251024_194446.log

############################################################
# Performance Benchmark Suite
# Parameters: L=512, N=100000, T=1000
# Repeat configuration:
#   - Baseline: 10 times
#   - NumPy: 100 times
#   - Numba: 100 times
#   - C++: 100 times
#   - CUDA: 1000 times
############################################################

============================================================
Benchmarking: Baseline (Pure Python)
============================================================
Running benchmark with L=512, N=100000, T=1000
Repeats: 10 times
────────────────────────────────────────────────────────────
Results:
────────────────────────────────────────────────────────────
  Average execution time (core): 40.3266s
  Std deviation: ±2.0468s
  Min/Max: 37.9065s / 43.8201s
  Average dwell ratio: 0.2508 (±0.001322)
  Total test time: 403.27s

[... CUDA Advanced 的输出显示最佳性能 ...]

════════════════════════════════════════════════════════════════════════════════════════════════════════
                                            PERFORMANCE SUMMARY
════════════════════════════════════════════════════════════════════════════════════════════════════════
Implementation                    Repeats    Avg Time (s)    Speedup      vs Previous  Dwell Ratio
────────────────────────────────────────────────────────────────────────────────────────────────────────
Baseline (Pure Python)            10         40.3266         1.00×        -            0.250834
NumPy (Vectorized)                100        1.8362          21.96×       21.96×       0.250049
Numba (JIT + Parallel)            100        1.6592          24.31×       1.11×        0.250018
C++ OpenMP                        100        2.5688          15.70×       0.65×        0.250049
C++ SIMD                          100        0.8152          49.47×       3.15×        0.249010
CUDA (Kernel Only)                1000       0.0013          31,937×      646×         0.249744
CUDA (With Transfer)              1000       0.0026          15,571×      0.49×        0.249744
CUDA Advanced (Kernel Only)       1000       0.0006          68,355×      4.39×        0.249744
CUDA Advanced (With Transfer)     1000       0.0019          21,512×      0.31×        0.249744
════════════════════════════════════════════════════════════════════════════════════════════════════════

✓ Benchmark visualization saved to: results/benchmark_20251024_194446.png
✓ PDF version saved to: results/benchmark_20251024_194446.pdf

✓ Benchmark complete!
✓ Results saved to: results/
  - Log file: results/benchmark_20251024_194446.log
  - Figures: benchmark_20251024_194446.png/pdf
```

**🏆 最佳性能: CUDA Advanced (Kernel Only) - 0.6ms，68,355倍加速！**

---

## 性能优化技术说明

### 1. NumPy 向量化
- 使用 `np.ndarray` 替代 Python 列表
- 布尔索引筛选中心区域粒子
- 向量化随机数生成和位置更新

### 2. Numba JIT 编译
- `@jit(nopython=True, parallel=True)` 装饰器
- 自动并行化（`prange`）
- 编译为机器码执行

### 3. C++ OpenMP
- 原生 C++ 实现
- `#pragma omp parallel for` 多线程并行
- 编译期优化 `-O3 -march=native`

### 4. C++ SIMD
- AVX2 SIMD 指令集
- Xorshift 快速随机数生成
- 位运算优化取模操作

### 5. CUDA GPU 加速
- 大规模并行计算（数千 GPU 核心）
- 设备端随机数生成
- 局部累加减少原子操作
- **双重计时**：区分纯计算和数据传输时间

---

## 故障排除

### Q: 运行 benchmark.py 时提示某些模块不可用
A: 这是正常的。如果 C++ 或 CUDA 模块未编译，会自动跳过这些实现。只测试可用的 Python 实现。

### Q: C++ 编译失败
A: 确保安装了 GCC/G++ 编译器和 Python 开发头文件：
```bash
sudo apt install build-essential python3-dev
```

### Q: CUDA 编译失败
A: 检查：
1. 是否安装 CUDA Toolkit：`nvcc --version`
2. 是否有 NVIDIA GPU：`nvidia-smi`
3. 是否安装 pybind11：`pip install pybind11`

### Q: Numba 首次运行很慢
A: 正常现象。Numba 首次运行需要 JIT 编译，后续运行会快很多。`benchmark.py` 已包含预热阶段。

---

## 联系方式

如有问题，请联系超算俱乐部：
- 邮箱: Zhengyang_Li@email.ncu.edu.cn
- QQ群: 95878716159

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---
