import time
import random
import sys
import os

# 导入配置文件
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import L, N, T, CENTER_MIN, CENTER_MAX

def run_baseline_simulation(L, N, T):
    """纯 Python 串行模拟（返回particles用于计时分离）"""
    
    # 1. 初始化粒子位置：使用列表的列表
    # [[x1, y1], [x2, y2], ...]
    particles = [[random.randint(0, L - 1) for _ in range(2)] for _ in range(N)]
    
    total_dwell_steps = 0

    # 2. 开始模拟
    for _ in range(T):  # 遍历时间步
        for i in range(N):  # 遍历所有粒子
            
            # 2a. 检查粒子是否在中心区域
            x, y = particles[i]
            if (CENTER_MIN <= x < CENTER_MAX) and (CENTER_MIN <= y < CENTER_MAX):
                total_dwell_steps += 1
            
            # 2b. 随机移动粒子
            direction = random.randint(0, 3) # 0:上, 1:下, 2:左, 3:右
            
            if direction == 0:
                y = (y + 1) % L  # 向上 (+y)
            elif direction == 1:
                y = (y - 1 + L) % L  # 向下 (-y)，注意 +L 确保周期性边界正确
            elif direction == 2:
                x = (x - 1 + L) % L  # 向左 (-x)
            else: # direction == 3
                x = (x + 1) % L  # 向右 (+x)
            
            # 更新粒子位置
            particles[i] = [x, y]
            
    return total_dwell_steps

def run_baseline_core(particles, L, N, T):
    """纯计算核心（不含初始化，用于公平对比）"""
    total_dwell_steps = 0

    for _ in range(T):
        for i in range(N):
            x, y = particles[i]
            if (CENTER_MIN <= x < CENTER_MAX) and (CENTER_MIN <= y < CENTER_MAX):
                total_dwell_steps += 1
            
            direction = random.randint(0, 3)
            
            if direction == 0:
                y = (y + 1) % L
            elif direction == 1:
                y = (y - 1 + L) % L
            elif direction == 2:
                x = (x - 1 + L) % L
            else:
                x = (x + 1) % L
            
            particles[i] = [x, y]
            
    return total_dwell_steps

def main():
    print(f"Starting baseline simulation: L={L}, N={N}, T={T}")
    
    # 数据准备（不计时）
    particles = [[random.randint(0, L - 1) for _ in range(2)] for _ in range(N)]
    
    # 纯计算部分（计时）
    start_time = time.time()
    total_dwell_steps = run_baseline_core(particles, L, N, T)
    end_time = time.time()
    
    # --- 3. 计算并输出结果 ---
    simulation_time = end_time - start_time
    dwell_ratio = total_dwell_steps / (N * T)
    
    print(f"Average dwell ratio: {dwell_ratio:.4f}")
    print(f"Simulation time (core only): {simulation_time:.2f}s")

if __name__ == "__main__":
    main()