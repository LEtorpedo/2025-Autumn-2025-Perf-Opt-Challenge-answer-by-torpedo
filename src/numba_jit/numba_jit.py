import time
import numpy as np
import sys
import os
from numba import jit, prange # 导入 numba

# 导入配置文件和可视化模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import L, N, T, CENTER_MIN, CENTER_MAX
from visualization import visualize_heatmap

# --- 移动增量查找表 ---
MOVE_DELTAS = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.int32)

# --- Numba JIT 编译的核心函数 ---
# nopython=True: 强制 Numba 生成纯机器码，不回退到 Python 对象模式。
# parallel=True: 允许 Numba 使用 prange() 进行自动并行化。
@jit(nopython=True, parallel=True, fastmath=True)
def run_numba_core(particles, all_moves, L, N, T, center_min, center_max, move_deltas):
    """
    Numba 优化的核心循环。
    此函数内的所有操作都将被编译。
    """
    total_dwell_steps = 0
    
    # 临时的布尔数组，用于并行安全地计数
    is_in_center = np.zeros(N, dtype=np.bool_)

    for t in range(T):
        
        # prange: Numba 的并行 for 循环
        # Numba 会自动将这个循环分配到所有可用的 CPU 核心上
        for i in prange(N):
            # 1. 检查位置 (并行)
            x, y = particles[i, 0], particles[i, 1]
            is_in_center[i] = (center_min <= x < center_max) and \
                              (center_min <= y < center_max)
            
            # 2. 移动粒子 (并行)
            move_idx = all_moves[t, i]
            particles[i, 0] = (x + move_deltas[move_idx, 0] + L) % L
            particles[i, 1] = (y + move_deltas[move_idx, 1] + L) % L
        
        # 3. 串行汇总 (非常快)
        # 在并行循环之外安全地求和，避免竞争条件
        total_dwell_steps += np.sum(is_in_center)
            
    return total_dwell_steps, particles

def run_numba_simulation_for_benchmark(L, N, T):
    """
    专为 benchmark 设计的函数（不包含 warmup）
    假设已经完成了 JIT 编译
    """
    # 1. 在 Python 中准备所有数据
    particles = np.random.randint(0, L, size=(N, 2), dtype=np.int32)
    
    # 2. 预先生成所有随机移动
    all_moves = np.random.randint(0, 4, size=(T, N), dtype=np.uint8)
    
    # 直接运行已编译的代码（不 warmup）
    start_run_time = time.time()
    
    total_dwell_steps, final_particles = run_numba_core(
        particles, all_moves, L, N, T, 
        CENTER_MIN, CENTER_MAX, MOVE_DELTAS
    )
    
    end_run_time = time.time()
    
    return total_dwell_steps, final_particles, (end_run_time - start_run_time)

def run_numba_simulation(L, N, T):
    """Numba 模拟的 Python 包装器（包含 warmup）"""
    
    # 1. 在 Python 中准备所有数据
    particles = np.random.randint(0, L, size=(N, 2), dtype=np.int32)
    
    # 2. 预先生成所有随机移动
    # (T, N) 形状。这会占用内存 (T*N*1 byte ~ 100MB)，但速度极快。
    all_moves = np.random.randint(0, 4, size=(T, N), dtype=np.uint8)
    
    # --- JIT 编译和执行 ---
    # 第一次调用会触发 JIT 编译，这需要几秒钟。
    print("Numba JIT compilation (first run)...")
    # 传入 .copy() 以确保编译运行不影响实际数据
    _ = run_numba_core(particles.copy(), all_moves, L, N, T, 
                       CENTER_MIN, CENTER_MAX, MOVE_DELTAS)
    
    print("Compilation finished. Starting timed run...")
    
    # 第二次调用将运行已编译的机器码
    start_run_time = time.time()
    
    total_dwell_steps, final_particles = run_numba_core(
        particles, all_moves, L, N, T, 
        CENTER_MIN, CENTER_MAX, MOVE_DELTAS
    )
    
    end_run_time = time.time()
    
    return total_dwell_steps, final_particles, (end_run_time - start_run_time)

def main():
    print(f"Starting Numba (parallel) simulation: L={L}, N={N}, T={T}")
    start_total_time = time.time()
    
    total_dwell_steps, final_particles, sim_time = run_numba_simulation(L, N, T)
    
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    
    dwell_ratio = total_dwell_steps / (N * T)
    
    print(f"Average dwell ratio: {dwell_ratio:.4f}")
    # 我们只关心核心模拟时间
    print(f"Simulation time: {sim_time:.2f}s") 
    print(f"(Total time incl. data generation: {total_time:.2f}s)")

    # 可视化（不计时）
    visualize_heatmap(final_particles, L, N, T,
                     filename='particle_heatmap_numba.png',
                     title_prefix='Numba')

if __name__ == "__main__":
    main()