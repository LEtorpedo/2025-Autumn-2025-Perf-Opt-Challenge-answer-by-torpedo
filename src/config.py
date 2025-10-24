import sys

# --- 参数定义 ---
# 检查是否从命令行传入参数
if len(sys.argv) == 4:
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    T = int(sys.argv[3])
else:
    # 考核固定的参数
    L = 512
    N = 100000
    T = 1000

# --- 中心区域定义 ---
CENTER_MIN = L // 4
CENTER_MAX = 3 * L // 4

# --- 基准测试配置 ---
# 不同实现的重复测试次数
BENCHMARK_REPEATS = {
    'baseline': 10,      # Baseline 运行 10 次
    'numpy': 100,       # NumPy 实现运行 100 次
    'numba': 100,       # Numba 实现运行 100 次
    'cpp': 100,         # C++ 实现运行 100 次
    'cuda': 1000,        # CUDA 实现运行 1000 次
}

