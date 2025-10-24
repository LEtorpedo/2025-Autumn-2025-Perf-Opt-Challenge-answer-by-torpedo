#!/bin/bash
# Benchmark 启动脚本
# 设置正确的库路径以加载 CUDA 模块

# 强制使用系统的 libstdc++（绕过 conda 环境的旧版本）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "=========================================="
echo "Random Walk Performance Benchmark"
echo "=========================================="
echo ""
echo "Environment setup:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠ nvidia-smi not found - CUDA tests may fail"
    echo ""
fi

# 运行 benchmark
python benchmark.py "$@"

