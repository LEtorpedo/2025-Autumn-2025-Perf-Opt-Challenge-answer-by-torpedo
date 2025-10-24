/**
 * Random Walk Simulation - CUDA Main Host Code
 * 
 * 主机端C++代码，负责:
 * 1. 内存分配和数据传输
 * 2. 核函数调用
 * 3. 计时和结果收集
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// 声明核函数（定义在kernel.cu中）
extern "C" {
    void launch_random_walk_kernel(
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
    );
}

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main(int argc, char** argv) {
    // 默认参数
    int L = 512;
    int N = 100000;
    int T = 1000;
    
    // 从命令行读取参数
    if (argc == 4) {
        L = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        T = std::atoi(argv[3]);
    }
    
    const int center_min = L / 4;
    const int center_max = 3 * L / 4;
    
    std::cout << "CUDA Random Walk Simulation" << std::endl;
    std::cout << "Parameters: L=" << L << ", N=" << N << ", T=" << T << std::endl;
    
    // 查询GPU设备信息
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    // 1. 主机端内存分配和初始化
    std::vector<int> h_particles_x(N);
    std::vector<int> h_particles_y(N);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, L - 1);
    
    for (int i = 0; i < N; ++i) {
        h_particles_x[i] = dis(gen);
        h_particles_y[i] = dis(gen);
    }
    
    // 2. 设备端内存分配
    int *d_particles_x, *d_particles_y;
    unsigned long long *d_dwell_steps;
    
    CUDA_CHECK(cudaMalloc(&d_particles_x, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particles_y, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dwell_steps, sizeof(unsigned long long)));
    
    // 初始化dwell_steps为0
    unsigned long long zero = 0;
    CUDA_CHECK(cudaMemcpy(d_dwell_steps, &zero, sizeof(unsigned long long), 
                          cudaMemcpyHostToDevice));
    
    // 3. 数据传输到设备
    CUDA_CHECK(cudaMemcpy(d_particles_x, h_particles_x.data(), N * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particles_y, h_particles_y.data(), N * sizeof(int), 
                          cudaMemcpyHostToDevice));
    
    // 4. 配置核函数参数
    const int block_size = 256;  // 每个块的线程数
    const int grid_size = (N + block_size - 1) / block_size;  // 块数
    
    std::cout << "Launch configuration: " << grid_size << " blocks × " 
              << block_size << " threads" << std::endl;
    
    // 5. 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 6. 开始计时并启动核函数
    CUDA_CHECK(cudaEventRecord(start));
    
    launch_random_walk_kernel(
        d_particles_x, d_particles_y, d_dwell_steps,
        L, N, T, center_min, center_max,
        grid_size, block_size
    );
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // 7. 计算执行时间
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // 8. 将结果传回主机
    unsigned long long h_dwell_steps;
    CUDA_CHECK(cudaMemcpy(&h_dwell_steps, d_dwell_steps, 
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(h_particles_x.data(), d_particles_x, N * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_particles_y.data(), d_particles_y, N * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    
    // 9. 输出结果
    double dwell_ratio = static_cast<double>(h_dwell_steps) / (static_cast<long long>(N) * T);
    double seconds = milliseconds / 1000.0;
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Average dwell ratio: " << dwell_ratio << std::endl;
    std::cout << "  Simulation time: " << seconds << " seconds" << std::endl;
    std::cout << "  (" << milliseconds << " ms)" << std::endl;
    
    // 10. 清理
    CUDA_CHECK(cudaFree(d_particles_x));
    CUDA_CHECK(cudaFree(d_particles_y));
    CUDA_CHECK(cudaFree(d_dwell_steps));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}

