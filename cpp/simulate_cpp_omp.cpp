/**
 * Random Walk Simulation - C++ OpenMP Implementation
 * 
 * 使用C++和OpenMP实现多线程并行的随机游走模拟
 * 目标: 超越Numba性能，达到300-500×加速
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <omp.h>
#include <cmath>

namespace py = pybind11;

/**
 * C++ OpenMP实现的核心模拟函数
 * 
 * @param L 网格大小
 * @param N 粒子数量
 * @param T 时间步数
 * @param center_min 中心区域最小坐标
 * @param center_max 中心区域最大坐标
 * @return tuple (total_dwell_steps, final_particles)
 */
py::tuple run_cpp_omp_simulation(int L, int N, int T, int center_min, int center_max) {
    // 初始化粒子位置
    std::vector<int> particles_x(N);
    std::vector<int> particles_y(N);
    
    // 使用随机设备初始化
    std::random_device rd;
    
    // 并行初始化粒子位置
    #pragma omp parallel
    {
        // 每个线程有自己的随机数生成器
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_int_distribution<int> dis(0, L - 1);
        
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            particles_x[i] = dis(gen);
            particles_y[i] = dis(gen);
        }
    }
    
    // 总停留步数（使用long long避免溢出）
    long long total_dwell_steps = 0;
    
    // 时间步循环
    for (int t = 0; t < T; ++t) {
        long long dwell_this_step = 0;
        
        // 并行处理所有粒子
        #pragma omp parallel reduction(+:dwell_this_step)
        {
            // 每个线程的随机数生成器
            std::mt19937 gen(rd() + omp_get_thread_num() + t * 1000);
            std::uniform_int_distribution<int> move_dis(0, 3);
            
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                int x = particles_x[i];
                int y = particles_y[i];
                
                // 检查是否在中心区域
                if (x >= center_min && x < center_max && 
                    y >= center_min && y < center_max) {
                    dwell_this_step++;
                }
                
                // 生成随机移动方向
                int direction = move_dis(gen);
                
                // 移动粒子（周期性边界条件）
                switch (direction) {
                    case 0: // 上
                        y = (y + 1) % L;
                        break;
                    case 1: // 下
                        y = (y - 1 + L) % L;
                        break;
                    case 2: // 左
                        x = (x - 1 + L) % L;
                        break;
                    case 3: // 右
                        x = (x + 1) % L;
                        break;
                }
                
                particles_x[i] = x;
                particles_y[i] = y;
            }
        }
        
        total_dwell_steps += dwell_this_step;
    }
    
    // 将结果转换为numpy数组
    py::array_t<int> result_particles({N, 2});
    auto r = result_particles.mutable_unchecked<2>();
    
    for (int i = 0; i < N; ++i) {
        r(i, 0) = particles_x[i];
        r(i, 1) = particles_y[i];
    }
    
    return py::make_tuple(total_dwell_steps, result_particles);
}

// Python模块定义
PYBIND11_MODULE(random_walk_cpp_omp, m) {
    m.doc() = "Random Walk Simulation using C++ and OpenMP";
    
    m.def("run_simulation", &run_cpp_omp_simulation,
          "Run random walk simulation with C++ and OpenMP",
          py::arg("L"), py::arg("N"), py::arg("T"),
          py::arg("center_min"), py::arg("center_max"));
}

