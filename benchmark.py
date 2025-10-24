#!/usr/bin/env python3
"""
性能基准测试和可视化脚本
对比不同实现的运行时间和加速比
"""
import os
import sys
import ctypes
import ctypes.util

# 🔧 强制加载系统的 libstdc++ (解决 CUDA 模块的 GLIBCXX_3.4.32 问题)
# 这必须在导入任何其他模块之前完成
try:
    # 找到系统的 libstdc++
    system_libstdcpp = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
    if os.path.exists(system_libstdcpp):
        # 使用 RTLD_GLOBAL 全局加载，让所有后续加载的模块都使用这个版本
        ctypes.CDLL(system_libstdcpp, mode=ctypes.RTLD_GLOBAL)
        print(f"✓ Loaded system libstdc++: {system_libstdcpp}")
except Exception as e:
    print(f"Warning: Could not preload system libstdc++: {e}")

import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入配置
from config import L, N, T, BENCHMARK_REPEATS, CENTER_MIN, CENTER_MAX

# 导入各个实现的核心函数
from baseline.baseline import run_baseline_core
from numpy_process.numpy_process import run_numpy_core
from numba_jit.numba_jit import run_numba_simulation_for_benchmark, run_numba_core, MOVE_DELTAS

# 尝试导入 C++ 实现
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp'))
    import random_walk_cpp_omp
    import random_walk_cpp_simd
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: C++ modules not available: {e}")
    print("Please compile C++ modules first: cd cpp && python setup.py build_ext --inplace")
    CPP_AVAILABLE = False

# 尝试导入 CUDA 实现
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cuda'))
    import random_walk_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CUDA module not available: {e}")
    print("Please compile CUDA module first: cd cuda && python setup.py build_ext --inplace")
    CUDA_AVAILABLE = False

# 尝试导入 CUDA Advanced 实现
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cuda'))
    import random_walk_cuda_advanced
    CUDA_ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CUDA Advanced module not available: {e}")
    print("Please compile CUDA Advanced module: cd cuda && python setup_advanced.py build_ext --inplace")
    CUDA_ADVANCED_AVAILABLE = False

def benchmark_implementation(name, func, L, N, T, repeats=1, warmup=False, impl_type='baseline'):
    """
    测试单个实现的性能（支持多次运行取平均）
    
    Args:
        name: 实现名称
        func: 要测试的函数
        L, N, T: 模拟参数
        repeats: 重复运行次数
        warmup: 是否需要预热（对 Numba 有用）
        impl_type: 实现类型 ('baseline', 'numpy', 'numba', 'cpp', 'cuda', 'cuda_with_transfer')
    
    Returns:
        (avg_execution_time, avg_dwell_ratio, repeats)
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    warmup_time = 0
    if warmup:
        print("Warming up (JIT compilation)...")
        warmup_start = time.time()
        try:
            if impl_type == 'numba':
                # Numba warmup: 调用一次核心函数来触发 JIT 编译
                particles = np.random.randint(0, L, size=(N, 2), dtype=np.int32)
                all_moves = np.random.randint(0, 4, size=(T, N), dtype=np.uint8)
                _ = run_numba_core(particles, all_moves, L, N, T, 
                                  CENTER_MIN, CENTER_MAX, MOVE_DELTAS)
            else:
                _ = func(L, N, T)
        except Exception as e:
            print(f"Warmup error (ignored): {e}")
        warmup_time = time.time() - warmup_start
        print(f"Warmup complete. Time: {warmup_time:.2f}s")
    
    print(f"Running benchmark with L={L}, N={N}, T={T}")
    print(f"Repeats: {repeats} times")
    
    execution_times = []
    dwell_ratios = []
    
    # 使用 tqdm 进度条（禁用动态列以提高性能）
    progress_bar = tqdm(range(repeats), desc=f"  {name[:20]:20s}", 
                       unit="run", leave=False, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for i in progress_bar:
        
        # 根据实现类型准备数据（不计时）
        if impl_type == 'baseline':
            import random
            particles = [[random.randint(0, L - 1) for _ in range(2)] for _ in range(N)]
            
            start_time = time.time()
            total_dwell_steps = func(particles, L, N, T)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
        elif impl_type == 'numpy':
            particles = np.random.randint(0, L, size=(N, 2), dtype=np.int32)
            
            start_time = time.time()
            result = func(particles, L, N, T)
            end_time = time.time()
            
            execution_time = end_time - start_time
            total_dwell_steps = result[0] if isinstance(result, tuple) else result
            
        elif impl_type == 'numba':
            # Numba: 调用不包含 warmup 的版本
            result = func(L, N, T)
            total_dwell_steps = result[0]
            execution_time = result[2]  # Numba 返回内部计时
            
        elif impl_type == 'cpp':
            # C++ 实现返回 (particles, dwell_ratio, execution_time)
            # 函数内部已经计时核心计算部分
            result = func(L, N, T, CENTER_MIN, CENTER_MAX)
            total_dwell_steps = int(result[1] * N * T)  # 从 dwell_ratio 反算
            execution_time = result[2]  # C++ 返回的计时
            
        elif impl_type == 'cuda':
            # CUDA 实现返回 (particles, dwell_ratio, kernel_time)
            # 只计时核函数执行
            result = func(L, N, T, CENTER_MIN, CENTER_MAX)
            total_dwell_steps = int(result[1] * N * T)  # 从 dwell_ratio 反算
            execution_time = result[2]  # CUDA 核函数时间
            
        elif impl_type == 'cuda_with_transfer':
            # CUDA 实现返回 (particles, dwell_ratio, kernel_time, total_time)
            # 包含数据传输时间
            result = func(L, N, T, CENTER_MIN, CENTER_MAX)
            total_dwell_steps = int(result[1] * N * T)  # 从 dwell_ratio 反算
            execution_time = result[3]  # CUDA 总时间（包含传输）
        
        dwell_ratio = total_dwell_steps / (N * T)
        
        execution_times.append(execution_time)
        dwell_ratios.append(dwell_ratio)
    
    progress_bar.close()  # 确保进度条关闭
    
    avg_execution_time = np.mean(execution_times)
    avg_dwell_ratio = np.mean(dwell_ratios)
    std_execution_time = np.std(execution_times)
    std_dwell_ratio = np.std(dwell_ratios)
    min_execution_time = np.min(execution_times)
    max_execution_time = np.max(execution_times)
    
    print(f"\n{'─'*60}")
    print(f"Results:")
    print(f"{'─'*60}")
    if warmup and warmup_time > 0:
        print(f"  Warmup time: {warmup_time:.4f}s")
    print(f"  Average execution time (core): {avg_execution_time:.4f}s")
    print(f"  Std deviation: ±{std_execution_time:.4f}s")
    print(f"  Min/Max: {min_execution_time:.4f}s / {max_execution_time:.4f}s")
    print(f"  Average dwell ratio: {avg_dwell_ratio:.4f} (±{std_dwell_ratio:.6f})")
    print(f"  Total test time: {np.sum(execution_times):.2f}s")
    if warmup and warmup_time > 0:
        print(f"  Total with warmup: {warmup_time + np.sum(execution_times):.2f}s")
    print(f"{'─'*60}")
    
    return avg_execution_time, avg_dwell_ratio, repeats

def run_benchmarks():
    """运行所有基准测试"""
    
    print(f"\n{'#'*60}")
    print(f"# Performance Benchmark Suite")
    print(f"# Parameters: L={L}, N={N}, T={T}")
    print(f"# Repeat configuration:")
    print(f"#   - Baseline: {BENCHMARK_REPEATS['baseline']} times")
    print(f"#   - NumPy: {BENCHMARK_REPEATS['numpy']} times")
    print(f"#   - Numba: {BENCHMARK_REPEATS['numba']} times")
    print(f"#   - C++: {BENCHMARK_REPEATS['cpp']} times")
    print(f"#   - CUDA: {BENCHMARK_REPEATS['cuda']} times")
    print(f"{'#'*60}")
    
    results = {}
    
    # 1. Baseline (纯 Python)
    try:
        time_baseline, ratio_baseline, repeats_baseline = benchmark_implementation(
            "Baseline (Pure Python)", 
            run_baseline_core, 
            L, N, T,
            repeats=BENCHMARK_REPEATS['baseline'],
            impl_type='baseline'
        )
        results['Baseline\n(Pure Python)'] = {
            'time': time_baseline, 
            'ratio': ratio_baseline,
            'speedup': 1.0,
            'repeats': repeats_baseline
        }
    except Exception as e:
        print(f"Error in Baseline: {e}")
        results['Baseline\n(Pure Python)'] = {
            'time': None, 'ratio': None, 'speedup': None, 'repeats': None
        }
    
    # 2. NumPy (向量化)
    try:
        time_numpy, ratio_numpy, repeats_numpy = benchmark_implementation(
            "NumPy (Vectorized)", 
            run_numpy_core, 
            L, N, T,
            repeats=BENCHMARK_REPEATS['numpy'],
            impl_type='numpy'
        )
        results['NumPy\n(Vectorized)'] = {
            'time': time_numpy, 
            'ratio': ratio_numpy,
            'speedup': time_baseline / time_numpy if time_baseline else None,
            'repeats': repeats_numpy
        }
    except Exception as e:
        print(f"Error in NumPy: {e}")
        results['NumPy\n(Vectorized)'] = {
            'time': None, 'ratio': None, 'speedup': None, 'repeats': None
        }
    
    # 3. Numba (JIT + 并行)
    try:
        time_numba, ratio_numba, repeats_numba = benchmark_implementation(
            "Numba (JIT + Parallel)", 
            run_numba_simulation_for_benchmark, 
            L, N, T,
            repeats=BENCHMARK_REPEATS['numba'],
            warmup=True,
            impl_type='numba'
        )
        results['Numba\n(JIT + Parallel)'] = {
            'time': time_numba, 
            'ratio': ratio_numba,
            'speedup': time_baseline / time_numba if time_baseline else None,
            'repeats': repeats_numba
        }
    except Exception as e:
        print(f"Error in Numba: {e}")
        results['Numba\n(JIT + Parallel)'] = {
            'time': None, 'ratio': None, 'speedup': None, 'repeats': None
        }
    
    # 4. C++ OpenMP (如果可用)
    if CPP_AVAILABLE:
        try:
            time_cpp_omp, ratio_cpp_omp, repeats_cpp_omp = benchmark_implementation(
                "C++ (OpenMP)", 
                random_walk_cpp_omp.run_simulation, 
                L, N, T,
                repeats=BENCHMARK_REPEATS['cpp'],
                impl_type='cpp'
            )
            results['C++ OpenMP'] = {
                'time': time_cpp_omp, 
                'ratio': ratio_cpp_omp,
                'speedup': time_baseline / time_cpp_omp if time_baseline else None,
                'repeats': repeats_cpp_omp
            }
        except Exception as e:
            print(f"Error in C++ OpenMP: {e}")
            results['C++ OpenMP'] = {
                'time': None, 'ratio': None, 'speedup': None, 'repeats': None
            }
        
        # 5. C++ SIMD (如果可用)
        try:
            time_cpp_simd, ratio_cpp_simd, repeats_cpp_simd = benchmark_implementation(
                "C++ (SIMD + OpenMP)", 
                random_walk_cpp_simd.run_simulation, 
                L, N, T,
                repeats=BENCHMARK_REPEATS['cpp'],
                impl_type='cpp'
            )
            results['C++ SIMD'] = {
                'time': time_cpp_simd, 
                'ratio': ratio_cpp_simd,
                'speedup': time_baseline / time_cpp_simd if time_baseline else None,
                'repeats': repeats_cpp_simd
            }
        except Exception as e:
            print(f"Error in C++ SIMD: {e}")
            results['C++ SIMD'] = {
                'time': None, 'ratio': None, 'speedup': None, 'repeats': None
            }
    
    # 6. CUDA - 仅核函数时间 (如果可用)
    if CUDA_AVAILABLE:
        try:
            time_cuda, ratio_cuda, repeats_cuda = benchmark_implementation(
                "CUDA (Kernel Only)", 
                random_walk_cuda.run_cuda_simulation, 
                L, N, T,
                repeats=BENCHMARK_REPEATS['cuda'],
                impl_type='cuda'
            )
            results['CUDA\n(Kernel Only)'] = {
                'time': time_cuda, 
                'ratio': ratio_cuda,
                'speedup': time_baseline / time_cuda if time_baseline else None,
                'repeats': repeats_cuda
            }
        except Exception as e:
            print(f"Error in CUDA (Kernel Only): {e}")
            import traceback
            traceback.print_exc()
            results['CUDA\n(Kernel Only)'] = {
                'time': None, 'ratio': None, 'speedup': None, 'repeats': None
            }
        
        # 7. CUDA - 包含数据传输 (如果可用)
        try:
            time_cuda_full, ratio_cuda_full, repeats_cuda_full = benchmark_implementation(
                "CUDA (With Transfer)", 
                random_walk_cuda.run_cuda_simulation_with_transfer, 
                L, N, T,
                repeats=BENCHMARK_REPEATS['cuda'],
                impl_type='cuda_with_transfer'
            )
            results['CUDA\n(With Transfer)'] = {
                'time': time_cuda_full, 
                'ratio': ratio_cuda_full,
                'speedup': time_baseline / time_cuda_full if time_baseline else None,
                'repeats': repeats_cuda_full
            }
        except Exception as e:
            print(f"Error in CUDA (With Transfer): {e}")
            import traceback
            traceback.print_exc()
            results['CUDA\n(With Transfer)'] = {
                'time': None, 'ratio': None, 'speedup': None, 'repeats': None
            }
    
    # 8. CUDA Advanced - 仅核函数时间 (如果可用)
    if CUDA_ADVANCED_AVAILABLE:
        try:
            time_cuda_adv, ratio_cuda_adv, repeats_cuda_adv = benchmark_implementation(
                "CUDA Advanced (Kernel)", 
                random_walk_cuda_advanced.run_cuda_simulation_advanced, 
                L, N, T,
                repeats=BENCHMARK_REPEATS['cuda'],
                impl_type='cuda'
            )
            results['CUDA Advanced\n(Kernel Only)'] = {
                'time': time_cuda_adv, 
                'ratio': ratio_cuda_adv,
                'speedup': time_baseline / time_cuda_adv if time_baseline else None,
                'repeats': repeats_cuda_adv
            }
        except Exception as e:
            print(f"Error in CUDA Advanced (Kernel Only): {e}")
            import traceback
            traceback.print_exc()
            results['CUDA Advanced\n(Kernel Only)'] = {
                'time': None, 'ratio': None, 'speedup': None, 'repeats': None
            }
        
        # 9. CUDA Advanced - 包含数据传输 (如果可用)
        try:
            time_cuda_adv_full, ratio_cuda_adv_full, repeats_cuda_adv_full = benchmark_implementation(
                "CUDA Advanced (With Transfer)", 
                random_walk_cuda_advanced.run_cuda_simulation_advanced_with_transfer, 
                L, N, T,
                repeats=BENCHMARK_REPEATS['cuda'],
                impl_type='cuda_with_transfer'
            )
            results['CUDA Advanced\n(With Transfer)'] = {
                'time': time_cuda_adv_full, 
                'ratio': ratio_cuda_adv_full,
                'speedup': time_baseline / time_cuda_adv_full if time_baseline else None,
                'repeats': repeats_cuda_adv_full
            }
        except Exception as e:
            print(f"Error in CUDA Advanced (With Transfer): {e}")
            import traceback
            traceback.print_exc()
            results['CUDA Advanced\n(With Transfer)'] = {
                'time': None, 'ratio': None, 'speedup': None, 'repeats': None
            }
    
    return results

def visualize_results(results, output_dir='results', timestamp=None):
    """绘制性能对比图表"""
    
    # 过滤掉失败的测试
    valid_results = {k: v for k, v in results.items() if v['time'] is not None}
    
    if not valid_results:
        print("No valid results to visualize!")
        return
    
    names = list(valid_results.keys())
    times = [valid_results[name]['time'] for name in names]
    speedups = [valid_results[name]['speedup'] for name in names]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 颜色方案（扩展以支持更多实现）
    colors = [
        '#e74c3c',  # Baseline (红)
        '#3498db',  # NumPy (蓝)
        '#2ecc71',  # Numba (绿)
        '#f39c12',  # C++ OpenMP (橙)
        '#9b59b6',  # C++ SIMD (紫)
        '#1abc9c',  # CUDA Basic Kernel (青)
        '#e67e22',  # CUDA Basic Transfer (深橙)
        '#16a085',  # CUDA Advanced Kernel (深青)
        '#27ae60',  # CUDA Advanced Transfer (深绿)
        '#c0392b',  # 备用
        '#8e44ad'   # 备用
    ]
    
    # 图1: 执行时间对比 (柱状图)
    bars1 = ax1.bar(range(len(names)), times, color=colors[:len(names)], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Implementation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Performance Comparison: Execution Time\n(L={L}, N={N}, T={T})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱子上添加数值标签
    for i, (bar, time_val) in enumerate(zip(bars1, times)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 图2: 加速比对比 (柱状图)
    bars2 = ax2.bar(range(len(names)), speedups, color=colors[:len(names)], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Implementation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Performance Comparison: Speedup Ratio\n(Baseline = 1.0×)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax2.legend()
    
    # 在柱子上添加加速比标签
    for i, (bar, speedup) in enumerate(zip(bars2, speedups)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}×',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 生成文件名
    if timestamp:
        base_filename = f'benchmark_{timestamp}'
    else:
        base_filename = 'performance_benchmark'
    
    # 保存图表
    png_path = os.path.join(output_dir, f'{base_filename}.png')
    pdf_path = os.path.join(output_dir, f'{base_filename}.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Benchmark visualization saved to: {png_path}")
    
    # 同时保存为 PDF
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ PDF version saved to: {pdf_path}")
    
    plt.close()

def print_summary_table(results):
    """打印性能汇总表格"""
    
    print(f"\n{'='*120}")
    print(f"{'PERFORMANCE SUMMARY':^120}")
    print(f"{'='*120}")
    print(f"{'Implementation':<25} {'Repeats':<10} {'Avg Time (s)':<15} {'Speedup':<12} {'vs Previous':<12} {'Dwell Ratio':<15}")
    print(f"{'-'*120}")
    
    prev_time = None
    for name, data in results.items():
        if data['time'] is not None:
            name_clean = name.replace('\n', ' ')
            repeats_str = f"{data['repeats']}"
            speedup_str = f"{data['speedup']:.2f}×" if data['speedup'] else 'N/A'
            
            # 计算相对上一个实现的加速比
            if prev_time and prev_time > 0:
                vs_prev = prev_time / data['time']
                vs_prev_str = f"{vs_prev:.2f}×"
            else:
                vs_prev_str = '-'
            
            print(f"{name_clean:<25} {repeats_str:<10} {data['time']:<15.4f} {speedup_str:<12} {vs_prev_str:<12} {data['ratio']:<15.6f}")
            prev_time = data['time']
        else:
            name_clean = name.replace('\n', ' ')
            print(f"{name_clean:<25} {'FAILED':<10} {'-':<15} {'-':<12} {'-':<12} {'-':<15}")
    
    print(f"{'='*120}\n")

def main():
    """主函数"""
    
    # 创建输出目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 设置日志文件
    log_file = os.path.join(results_dir, f'benchmark_{timestamp}.log')
    
    # 同时输出到终端和文件
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    
    print(f"Benchmark started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}\n")
    
    # 运行基准测试
    results = run_benchmarks()
    
    # 打印汇总表格
    print_summary_table(results)
    
    # 可视化结果
    visualize_results(results, results_dir, timestamp)
    
    print(f"\n✓ Benchmark complete!")
    print(f"✓ Results saved to: {results_dir}/")
    print(f"  - Log file: {log_file}")
    print(f"  - Figures: benchmark_{timestamp}.png/pdf")
    
    # 恢复标准输出
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()

