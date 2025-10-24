/**
 * Random Walk Simulation - CUDA with Python Binding
 * 
 * 将CUDA实现封装为Python可调用的接口
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
 * CUDA核函数 - 基础版本
 */
__global__ void random_walk_kernel(
    int* particles_x,
    int* particles_y,
    unsigned long long* dwell_steps,
    int L,
    int N,
    int T,
    int center_min,
    int center_max
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= N) return;
    
    // 初始化随机数生成器
    unsigned int rng_state = tid + 123456789;
    
    // 加载粒子位置
    int x = particles_x[tid];
    int y = particles_y[tid];
    
    // 本地dwell计数（减少原子操作冲突）
    int local_dwell = 0;
    
    // 时间步循环
    for (int t = 0; t < T; ++t) {
        // 检查是否在中心区域
        if (x >= center_min && x < center_max && 
            y >= center_min && y < center_max) {
            local_dwell++;
        }
        
        // 生成随机移动方向
        unsigned int rand_val = xorshift32(&rng_state);
        int direction = rand_val & 3;
        
        // 移动粒子
        switch (direction) {
            case 0: y = (y + 1) % L; break;
            case 1: y = (y - 1 + L) % L; break;
            case 2: x = (x - 1 + L) % L; break;
            case 3: x = (x + 1) % L; break;
        }
    }
    
    // 一次性原子累加（大幅减少原子操作次数）
    if (local_dwell > 0) {
        atomicAdd(dwell_steps, (unsigned long long)local_dwell);
    }
    
    // 写回最终位置
    particles_x[tid] = x;
    particles_y[tid] = y;
}

/**
 * Python调用接口（只计算核函数执行时间）
 * 返回: (particles, dwell_ratio, kernel_time)
 */
py::tuple run_cuda_simulation(int L, int N, int T, int center_min, int center_max) {
    // 1. 主机端内存分配和初始化
    std::vector<int> h_particles_x(N);
    std::vector<int> h_particles_y(N);
    
    // 使用简单的随机初始化
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
    
    // 初始化
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
    
    // 5. 创建CUDA事件用于精确计时（只计时核函数执行）
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 6. 开始计时并启动核函数
    CUDA_CHECK(cudaEventRecord(start));
    
    random_walk_kernel<<<grid_size, block_size>>>(
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
 * 返回: (particles, dwell_ratio, kernel_time, total_time)
 */
py::tuple run_cuda_simulation_with_transfer(int L, int N, int T, int center_min, int center_max) {
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
    
    // 5. 创建CUDA事件用于精确计时（只计时核函数执行）
    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    
    // 6. 开始计时并启动核函数
    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    random_walk_kernel<<<grid_size, block_size>>>(
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
PYBIND11_MODULE(random_walk_cuda, m) {
    m.doc() = "CUDA-accelerated Random Walk Simulation";
    
    m.def("run_cuda_simulation", &run_cuda_simulation,
          py::arg("L"), py::arg("N"), py::arg("T"),
          py::arg("center_min"), py::arg("center_max"),
          R"pbdoc(
              Run CUDA-accelerated random walk simulation (kernel time only).
              
              Args:
                  L: Grid size
                  N: Number of particles
                  T: Number of time steps
                  center_min: Minimum coordinate of center region
                  center_max: Maximum coordinate of center region
              
              Returns:
                  Tuple of (particles, dwell_ratio, kernel_time)
                  - particles: NumPy array of shape (N, 2)
                  - dwell_ratio: Float
                  - kernel_time: Float (seconds, kernel execution only)
          )pbdoc");
    
    m.def("run_cuda_simulation_with_transfer", &run_cuda_simulation_with_transfer,
          py::arg("L"), py::arg("N"), py::arg("T"),
          py::arg("center_min"), py::arg("center_max"),
          R"pbdoc(
              Run CUDA-accelerated random walk simulation (with data transfer time).
              
              Args:
                  L: Grid size
                  N: Number of particles
                  T: Number of time steps
                  center_min: Minimum coordinate of center region
                  center_max: Maximum coordinate of center region
              
              Returns:
                  Tuple of (particles, dwell_ratio, kernel_time, total_time)
                  - particles: NumPy array of shape (N, 2)
                  - dwell_ratio: Float
                  - kernel_time: Float (seconds, kernel execution only)
                  - total_time: Float (seconds, including data transfer)
          )pbdoc");
}

