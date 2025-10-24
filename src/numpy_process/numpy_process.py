import time
import numpy as np
import sys
import os

# 导入配置文件和可视化模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import L, N, T, CENTER_MIN, CENTER_MAX
from visualization import visualize_heatmap

# --- 向量化实现的技巧 ---
# 创建一个“移动增量”查找表
# 0:上 [0, 1], 1:下 [0, -1], 2:左 [-1, 0], 3:右 [1, 0]
MOVE_DELTAS = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.int32)

def run_numpy_simulation(L, N, T):
    """使用 NumPy 向量化模拟（完整版本）"""
    
    # 1. 初始化粒子位置：使用 2D NumPy 数组
    # (N, 2) 形状的数组，[[x1, y1], [x2, y2], ...]
    # 使用 int32 节约内存并加速计算
    particles = np.random.randint(0, L, size=(N, 2), dtype=np.int32)
    
    total_dwell_steps = 0

    # 2. 开始模拟 (T 步循环仍然在 Python 中)
    for _ in range(T):
        
        # 2a. 检查中心区域 (向量化)
        x_pos = particles[:, 0]
        y_pos = particles[:, 1]
        
        in_center_x = (x_pos >= CENTER_MIN) & (x_pos < CENTER_MAX)
        in_center_y = (y_pos >= CENTER_MIN) & (y_pos < CENTER_MAX)
        
        # (N,) 形状的布尔数组
        in_center = in_center_x & in_center_y
        
        # np.sum(True) == 1, np.sum(False) == 0
        total_dwell_steps += np.sum(in_center)
        
        # 2b. 随机移动所有 N 个粒子 (向量化)
        # 1. 生成 N 个随机移动方向 (0-3)
        random_moves = np.random.randint(0, 4, size=N)
        
        # 2. 使用 "fancy indexing" 从查找表中获取 N 个 [dx, dy] 增量
        deltas = MOVE_DELTAS[random_moves]
        
        # 3. 将增量应用到所有粒子
        # (particles + deltas + L) 确保负数索引正确回绕
        particles = (particles + deltas + L) % L
            
    return total_dwell_steps, particles

def run_numpy_core(particles, L, N, T):
    """纯计算核心（不含初始化，用于公平对比）"""
    total_dwell_steps = 0

    for _ in range(T):
        # 检查中心区域 (向量化)
        x_pos = particles[:, 0]
        y_pos = particles[:, 1]
        
        in_center_x = (x_pos >= CENTER_MIN) & (x_pos < CENTER_MAX)
        in_center_y = (y_pos >= CENTER_MIN) & (y_pos < CENTER_MAX)
        in_center = in_center_x & in_center_y
        
        total_dwell_steps += np.sum(in_center)
        
        # 随机移动所有粒子 (向量化)
        random_moves = np.random.randint(0, 4, size=N)
        deltas = MOVE_DELTAS[random_moves]
        particles = (particles + deltas + L) % L
            
    return total_dwell_steps, particles

def main():
    print(f"Starting NumPy simulation: L={L}, N={N}, T={T}")
    
    # 数据准备（不计时）
    particles = np.random.randint(0, L, size=(N, 2), dtype=np.int32)
    
    # 纯计算部分（计时）
    start_time = time.time()
    total_dwell_steps, final_particles = run_numpy_core(particles, L, N, T)
    end_time = time.time()
    
    simulation_time = end_time - start_time
    dwell_ratio = total_dwell_steps / (N * T)
    
    print(f"Average dwell ratio: {dwell_ratio:.4f}")
    print(f"Simulation time (core only): {simulation_time:.2f}s")
    
    # 可视化（不计时）
    visualize_heatmap(final_particles, L, N, T, 
                     filename='particle_heatmap_numpy.png',
                     title_prefix='NumPy')

if __name__ == "__main__":
    main()