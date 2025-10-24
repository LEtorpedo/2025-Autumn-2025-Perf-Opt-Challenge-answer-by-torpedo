"""
可视化模块
提供粒子分布热力图绘制功能
"""
import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(particles, L, N, T, filename='particle_heatmap.png', title_prefix=''):
    """
    绘制粒子分布热力图
    
    Args:
        particles: (N, 2) 粒子位置数组
        L: 网格大小
        N: 粒子数量
        T: 时间步数
        filename: 输出文件名
        title_prefix: 标题前缀（如 'NumPy', 'Numba'）
    """
    print(f"Generating heatmap...")
    
    x = particles[:, 0]
    y = particles[:, 1]
    
    # 使用 2D 直方图统计每个格子的粒子数
    counts, _, _ = np.histogram2d(x, y, bins=L, range=[[0, L], [0, L]])
    
    plt.figure(figsize=(10, 8))
    # 使用 imshow 绘制热力图, .T 是因为 hist2d 和 imshow 的 (x,y) 轴向不同
    # origin='lower' 确保 (0,0) 在左下角
    plt.imshow(counts.T, origin='lower', cmap='inferno')
    plt.colorbar(label='Particle Count')
    
    title = f'Final Particle Distribution (L={L}, N={N}, T={T})'
    if title_prefix:
        title = f'{title_prefix} - {title}'
    plt.title(title)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Heatmap saved to {filename}")
    plt.close()

