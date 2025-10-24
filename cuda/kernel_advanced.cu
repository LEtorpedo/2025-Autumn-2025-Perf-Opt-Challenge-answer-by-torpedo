/**
 * Random Walk Simulation - CUDA Kernel (Advanced Optimizations)
 * 
 * 高级优化版本，解决两大瓶颈:
 * 1. 原子锁争用 -> 并行归约
 * 2. 内存带宽 -> 共享内存缓存
 * 
 * 目标: 5000-10000×加速
 */

#include <cuda_runtime.h>

// 设备端Xorshift RNG
__device__ inline unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * Warp级归约 - 使用shuffle指令
 * 在warp内高效求和，无需共享内存
 */
__device__ inline unsigned int warp_reduce_sum(unsigned int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * 块级归约 - 树状求和
 * 使用共享内存在块内求和
 */
__device__ void block_reduce_sum(
    unsigned int* shared_data,
    unsigned int tid,
    unsigned int block_size
) {
    __syncthreads();
    
    // 树状归约
    for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个warp使用shuffle
    if (tid < 32) {
        unsigned int val = shared_data[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared_data[0] = val;
        }
    }
}

/**
 * CUDA核函数 - 高级优化版本
 * 
 * 核心优化:
 * 1. 并行归约: 使用共享内存块内累加，减少原子操作
 * 2. 共享内存缓存: 一次模拟K步，减少全局内存访问
 * 3. Warp shuffle: 加速归约
 */
__global__ void random_walk_kernel_advanced(
    int* particles_x,
    int* particles_y,
    unsigned long long* dwell_steps,
    int L,
    int N,
    int T,
    int center_min,
    int center_max,
    int K  // 每次在共享内存中模拟K步
) {
    // 共享内存声明
    extern __shared__ unsigned int shared_mem[];
    
    // 共享内存布局:
    // [0 ~ blockDim.x-1]: 块内dwell count
    // [blockDim.x ~ blockDim.x + blockDim.x-1]: particles_x缓存
    // [blockDim.x*2 ~ blockDim.x*3-1]: particles_y缓存
    unsigned int* block_dwell_counts = shared_mem;
    int* cache_x = (int*)&shared_mem[blockDim.x];
    int* cache_y = (int*)&shared_mem[blockDim.x * 2];
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // 初始化块内计数
    block_dwell_counts[tid] = 0;
    
    if (global_tid >= N) return;
    
    // 初始化RNG
    unsigned int rng_state = global_tid + 123456789;
    
    // 加载粒子位置到共享内存
    cache_x[tid] = particles_x[global_tid];
    cache_y[tid] = particles_y[global_tid];
    __syncthreads();
    
    int x = cache_x[tid];
    int y = cache_y[tid];
    
    // 主循环: 分批处理K步
    for (int t_start = 0; t_start < T; t_start += K) {
        int t_end = min(t_start + K, T);
        unsigned int local_dwell = 0;
        
        // 在寄存器中模拟K步（或剩余步数）
        for (int t = t_start; t < t_end; ++t) {
            // 检查中心区域（无分支版本）
            int in_x = (x >= center_min) && (x < center_max);
            int in_y = (y >= center_min) && (y < center_max);
            local_dwell += (in_x && in_y);
            
            // 移动（使用位运算，假设L=512）
            unsigned int rand_val = xorshift32(&rng_state);
            int direction = rand_val & 3;
            
            // 移动增量
            const int dx[4] = {0, 0, -1, 1};
            const int dy[4] = {1, -1, 0, 0};
            
            x = (x + dx[direction] + L) & (L - 1);  // 假设L是2的幂
            y = (y + dy[direction] + L) & (L - 1);
        }
        
        // 累加到共享内存
        block_dwell_counts[tid] += local_dwell;
    }
    
    // 块级归约
    block_reduce_sum(block_dwell_counts, tid, blockDim.x);
    
    // 块内第一个线程执行原子操作（大幅减少原子操作次数）
    if (tid == 0 && block_dwell_counts[0] > 0) {
        atomicAdd(dwell_steps, (unsigned long long)block_dwell_counts[0]);
    }
    
    // 写回最终位置到全局内存
    particles_x[global_tid] = x;
    particles_y[global_tid] = y;
}

/**
 * 启动高级优化核函数的C接口
 */
extern "C" void launch_random_walk_kernel_advanced(
    int* d_particles_x,
    int* d_particles_y,
    unsigned long long* d_dwell_steps,
    int L,
    int N,
    int T,
    int center_min,
    int center_max,
    int grid_size,
    int block_size,
    int K  // 批处理大小
) {
    // 计算共享内存大小
    // blockDim.x * sizeof(unsigned int) + 2 * blockDim.x * sizeof(int)
    size_t shared_mem_size = block_size * (sizeof(unsigned int) + 2 * sizeof(int));
    
    random_walk_kernel_advanced<<<grid_size, block_size, shared_mem_size>>>(
        d_particles_x, d_particles_y, d_dwell_steps,
        L, N, T, center_min, center_max, K
    );
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

