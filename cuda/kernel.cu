/**
 * Random Walk Simulation - CUDA Kernel (Basic Version)
 * 
 * 基础GPU实现:
 * - 每个线程处理一个粒子
 * - 使用原子操作累加dwell steps
 * - 基本的全局内存访问
 * 
 * 目标: 1000-3000×加速
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * 设备端Xorshift随机数生成器
 * 比cuRAND更快，但质量稍低（对于随机游走足够）
 */
__device__ inline unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * CUDA核函数 - 基础版本
 * 
 * @param particles_x X坐标数组
 * @param particles_y Y坐标数组
 * @param dwell_steps 总停留步数（原子累加）
 * @param L 网格大小
 * @param N 粒子数量
 * @param T 时间步数
 * @param center_min 中心区域最小坐标
 * @param center_max 中心区域最大坐标
 */
__global__ void random_walk_kernel_basic(
    int* particles_x,
    int* particles_y,
    unsigned long long* dwell_steps,
    int L,
    int N,
    int T,
    int center_min,
    int center_max
) {
    // 全局线程ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 超出范围的线程直接返回
    if (tid >= N) return;
    
    // 初始化随机数生成器（每个线程独立）
    unsigned int rng_state = tid + 123456789;
    
    // 加载粒子初始位置
    int x = particles_x[tid];
    int y = particles_y[tid];
    
    // 时间步循环
    for (int t = 0; t < T; ++t) {
        // 检查是否在中心区域
        if (x >= center_min && x < center_max && 
            y >= center_min && y < center_max) {
            // 原子操作累加（瓶颈！）
            atomicAdd(dwell_steps, 1ULL);
        }
        
        // 生成随机移动方向
        unsigned int rand_val = xorshift32(&rng_state);
        int direction = rand_val & 3;  // 位运算代替 % 4
        
        // 移动粒子（周期性边界条件）
        switch (direction) {
            case 0:  // 上
                y = (y + 1) % L;
                break;
            case 1:  // 下
                y = (y - 1 + L) % L;
                break;
            case 2:  // 左
                x = (x - 1 + L) % L;
                break;
            case 3:  // 右
                x = (x + 1) % L;
                break;
        }
    }
    
    // 写回最终位置
    particles_x[tid] = x;
    particles_y[tid] = y;
}

/**
 * 启动核函数的C接口（供main.cpp调用）
 */
extern "C" void launch_random_walk_kernel(
    int* d_particles_x,
    int* d_particles_y,
    unsigned long long* d_dwell_steps,
    int L,
    int N,
    int T,
    int center_min,
    int center_max,
    int grid_size,
    int block_size
) {
    random_walk_kernel_basic<<<grid_size, block_size>>>(
        d_particles_x, d_particles_y, d_dwell_steps,
        L, N, T, center_min, center_max
    );
    
    // 检查核函数启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

