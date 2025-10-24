/**
 * Random Walk Simulation - CUDA Advanced (High Performance)
 * 
 * 高级优化版本：块级归约 + 共享内存 + Warp Shuffle
 * 目标加速比：5000-10000×
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>
#include <chrono>

namespace py = pybind11;

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + \
                                     cudaGetErrorString(error)); \
        } \
    } while(0)

/**
 * Xorshift随机数生成器（设备端）
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
 * Warp级归约 - 使用shuffle指令
 */
__device__ inline unsigned int warp_reduce_sum(unsigned int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * 块级归约 - 树状求和
 */
__device__ void block_reduce_sum(
    unsigned int* shared_data,
    unsigned int tid,
    unsigned int block_size
) {
    __syncthreads();
    
    // 树状归约
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
}

/**
 * CUDA核函数 - 高级优化版本
 * 
 * 优化策略：
 * 1. 块级归约：减少原子操作到 grid_size 次
 * 2. 本地累加：每个线程先累加到寄存器
 * 3. 无分支移动：避免 switch 分支
 */
__global__ void random_walk_kernel_advanced(
    int* particles_x,
    int* particles_y,
    unsigned long long* dwell_steps,
    int L,
    int N,
    int T,
    int center_min,
    int center_max
) {
    // 共享内存：块内dwell counts
    extern __shared__ unsigned int block_dwell_counts[];
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // 初始化共享内存
    block_dwell_counts[tid] = 0;
    
    if (global_tid >= N) {
        __syncthreads();  // 即使超出范围也要参与同步
        return;
    }
    
    // 初始化RNG
    unsigned int rng_state = global_tid + 123456789;
    
    // 加载粒子位置
    int x = particles_x[global_tid];
    int y = particles_y[global_tid];
    
    // 本地dwell计数
    unsigned int local_dwell = 0;
    
    // 移动增量表（避免 switch）
    const int dx[4] = {0, 0, -1, 1};
    const int dy[4] = {1, -1, 0, 0};
    
    // 时间步循环
    for (int t = 0; t < T; ++t) {
        // 检查是否在中心区域（无分支版本）
        int in_x = (x >= center_min) && (x < center_max);
        int in_y = (y >= center_min) && (y < center_max);
        local_dwell += (in_x && in_y);
        
        // 生成随机移动方向
        unsigned int rand_val = xorshift32(&rng_state);
        int direction = rand_val & 3;
        
        // 移动粒子（周期性边界）
        x += dx[direction];
        y += dy[direction];
        
        // 处理边界（周期性）
        x = (x + L) % L;
        y = (y + L) % L;
    }
    
    // 存储到共享内存
    block_dwell_counts[tid] = local_dwell;
    
    // 块级归约
    block_reduce_sum(block_dwell_counts, tid, blockDim.x);
    
    // 块内第一个线程执行原子操作
    if (tid == 0 && block_dwell_counts[0] > 0) {
        atomicAdd(dwell_steps, (unsigned long long)block_dwell_counts[0]);
    }
    
    // 写回最终位置
    particles_x[global_tid] = x;
    particles_y[global_tid] = y;
}

/**
 * Python调用接口（只计算核函数执行时间）
 */
py::tuple run_cuda_simulation_advanced(int L, int N, int T, int center_min, int center_max) {
    // 1. 主机端内存分配和初始化
    std::vector<int> h_particles_x(N);
    std::vector<int> h_particles_y(N);
    
    unsigned int seed = 42;
    for (int i = 0; i < N; ++i) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        h_particles_x[i] = seed % L;
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        h_particles_y[i] = seed % L;
    }
    
    // 2. 设备端内存分配
    int *d_particles_x, *d_particles_y;
    unsigned long long *d_dwell_steps;
    
    CUDA_CHECK(cudaMalloc(&d_particles_x, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particles_y, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dwell_steps, sizeof(unsigned long long)));
    
    unsigned long long zero = 0;
    CUDA_CHECK(cudaMemcpy(d_dwell_steps, &zero, sizeof(unsigned long long), 
                          cudaMemcpyHostToDevice));
    
    // 3. 数据传输到设备（不计时）
    CUDA_CHECK(cudaMemcpy(d_particles_x, h_particles_x.data(), N * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particles_y, h_particles_y.data(), N * sizeof(int), 
                          cudaMemcpyHostToDevice));
    
    // 4. 配置核函数参数
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    const size_t shared_mem_size = block_size * sizeof(unsigned int);
    
    // 5. 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 6. 开始计时并启动核函数
    CUDA_CHECK(cudaEventRecord(start));
    
    random_walk_kernel_advanced<<<grid_size, block_size, shared_mem_size>>>(
        d_particles_x, d_particles_y, d_dwell_steps,
        L, N, T, center_min, center_max
    );
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // 7. 计算执行时间
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double seconds = milliseconds / 1000.0;
    
    // 8. 将结果传回主机
    unsigned long long h_dwell_steps;
    CUDA_CHECK(cudaMemcpy(&h_dwell_steps, d_dwell_steps, 
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(h_particles_x.data(), d_particles_x, N * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_particles_y.data(), d_particles_y, N * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    
    // 9. 清理设备内存
    CUDA_CHECK(cudaFree(d_particles_x));
    CUDA_CHECK(cudaFree(d_particles_y));
    CUDA_CHECK(cudaFree(d_dwell_steps));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // 10. 创建NumPy数组返回给Python
    py::array_t<int> particles({N, 2});
    auto particles_ptr = particles.mutable_unchecked<2>();
    for (int i = 0; i < N; ++i) {
        particles_ptr(i, 0) = h_particles_x[i];
        particles_ptr(i, 1) = h_particles_y[i];
    }
    
    // 11. 计算dwell ratio
    double dwell_ratio = static_cast<double>(h_dwell_steps) / 
                         (static_cast<long long>(N) * T);
    
    return py::make_tuple(particles, dwell_ratio, seconds);
}

/**
 * Python调用接口（包含数据传输时间）
 */
py::tuple run_cuda_simulation_advanced_with_transfer(int L, int N, int T, int center_min, int center_max) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // 1. 主机端内存分配和初始化
    std::vector<int> h_particles_x(N);
    std::vector<int> h_particles_y(N);
    
    unsigned int seed = 42;
    for (int i = 0; i < N; ++i) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        h_particles_x[i] = seed % L;
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        h_particles_y[i] = seed % L;
    }
    
    // 2. 设备端内存分配
    int *d_particles_x, *d_particles_y;
    unsigned long long *d_dwell_steps;
    
    CUDA_CHECK(cudaMalloc(&d_particles_x, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particles_y, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dwell_steps, sizeof(unsigned long long)));
    
    unsigned long long zero = 0;
    CUDA_CHECK(cudaMemcpy(d_dwell_steps, &zero, sizeof(unsigned long long), 
                          cudaMemcpyHostToDevice));
    
    // 3. 数据传输到设备
    CUDA_CHECK(cudaMemcpy(d_particles_x, h_particles_x.data(), N * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particles_y, h_particles_y.data(), N * sizeof(int), 
                          cudaMemcpyHostToDevice));
    
    // 4. 配置核函数参数
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    const size_t shared_mem_size = block_size * sizeof(unsigned int);
    
    // 5. 创建CUDA事件用于精确计时
    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    
    // 6. 开始计时并启动核函数
    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    random_walk_kernel_advanced<<<grid_size, block_size, shared_mem_size>>>(
        d_particles_x, d_particles_y, d_dwell_steps,
        L, N, T, center_min, center_max
    );
    
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));
    
    // 7. 计算核函数执行时间
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    double seconds_kernel = milliseconds_kernel / 1000.0;
    
    // 8. 将结果传回主机
    unsigned long long h_dwell_steps;
    CUDA_CHECK(cudaMemcpy(&h_dwell_steps, d_dwell_steps, 
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(h_particles_x.data(), d_particles_x, N * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_particles_y.data(), d_particles_y, N * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    
    // 9. 计算总时间
    auto end_total = std::chrono::high_resolution_clock::now();
    double seconds_total = std::chrono::duration<double>(end_total - start_total).count();
    
    // 10. 清理设备内存
    CUDA_CHECK(cudaFree(d_particles_x));
    CUDA_CHECK(cudaFree(d_particles_y));
    CUDA_CHECK(cudaFree(d_dwell_steps));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    
    // 11. 创建NumPy数组返回给Python
    py::array_t<int> particles({N, 2});
    auto particles_ptr = particles.mutable_unchecked<2>();
    for (int i = 0; i < N; ++i) {
        particles_ptr(i, 0) = h_particles_x[i];
        particles_ptr(i, 1) = h_particles_y[i];
    }
    
    // 12. 计算dwell ratio
    double dwell_ratio = static_cast<double>(h_dwell_steps) / 
                         (static_cast<long long>(N) * T);
    
    return py::make_tuple(particles, dwell_ratio, seconds_kernel, seconds_total);
}

/**
 * Pybind11模块定义
 */
PYBIND11_MODULE(random_walk_cuda_advanced, m) {
    m.doc() = "CUDA-accelerated Random Walk Simulation (Advanced Optimizations)";
    
    m.def("run_cuda_simulation_advanced", &run_cuda_simulation_advanced,
          py::arg("L"), py::arg("N"), py::arg("T"),
          py::arg("center_min"), py::arg("center_max"),
          R"pbdoc(
              Run CUDA-accelerated random walk (Advanced - Block Reduction).
              
              Returns: (particles, dwell_ratio, kernel_time)
          )pbdoc");
    
    m.def("run_cuda_simulation_advanced_with_transfer", &run_cuda_simulation_advanced_with_transfer,
          py::arg("L"), py::arg("N"), py::arg("T"),
          py::arg("center_min"), py::arg("center_max"),
          R"pbdoc(
              Run CUDA-accelerated random walk (Advanced - With Transfer).
              
              Returns: (particles, dwell_ratio, kernel_time, total_time)
          )pbdoc");
}

