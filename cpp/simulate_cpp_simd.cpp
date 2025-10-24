/**
 * Random Walk Simulation - C++ SIMD Implementation
 * 
 * 使用AVX2 SIMD指令集和极致优化技巧
 * 目标: 达到800-1500×加速
 * 
 * 核心优化:
 * 1. Xorshift RNG - 比mt19937快10倍
 * 2. 位运算消除分支 - L必须是2的幂
 * 3. SIMD处理 - 一次处理8个粒子
 * 4. 非对齐加载 - 避免线程崩溃
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <immintrin.h>  // AVX2指令集
#include <omp.h>
#include <vector>
#include <cstdint>
#include <chrono>

namespace py = pybind11;

// 快速Xorshift随机数生成器
class XorshiftRNG {
private:
    uint64_t state;
    
public:
    explicit XorshiftRNG(uint64_t seed = 123456789) : state(seed) {}
    
    // 生成下一个随机数
    inline uint32_t next() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return static_cast<uint32_t>(x);
    }
    
    // 生成8个随机数（用于SIMD）
    inline void next8(uint32_t* output) {
        for (int i = 0; i < 8; ++i) {
            output[i] = next();
        }
    }
};

/**
 * C++ SIMD实现的核心模拟函数
 * 
 * 注意: L必须是2的幂 (512 = 2^9)，以便使用位运算
 */
py::tuple run_cpp_simd_simulation(int L, int N, int T, int center_min, int center_max) {
    // 验证L是2的幂
    if ((L & (L - 1)) != 0) {
        throw std::invalid_argument("L must be a power of 2 for SIMD optimization");
    }
    
    const int L_mask = L - 1;  // 用于位运算取模: x & L_mask == x % L
    
    // 初始化粒子位置（对齐到32字节以优化SIMD加载）
    std::vector<int> particles_x(N);
    std::vector<int> particles_y(N);
    
    // 并行初始化
    #pragma omp parallel
    {
        XorshiftRNG rng(123456789 + omp_get_thread_num());
        
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            particles_x[i] = rng.next() & L_mask;
            particles_y[i] = rng.next() & L_mask;
        }
    }
    
    long long total_dwell_steps = 0;
    
    // 开始核心计算计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 预计算SIMD常量
    const __m256i simd_L_mask = _mm256_set1_epi32(L_mask);
    const __m256i simd_center_min = _mm256_set1_epi32(center_min);
    const __m256i simd_center_max = _mm256_set1_epi32(center_max);
    const __m256i simd_one = _mm256_set1_epi32(1);
    const __m256i simd_three = _mm256_set1_epi32(3);
    
    // 移动增量表: [上, 下, 左, 右] -> [(0,1), (0,-1), (-1,0), (1,0)]
    const int dx[4] = {0, 0, -1, 1};
    const int dy[4] = {1, -1, 0, 0};
    
    // 时间步循环
    for (int t = 0; t < T; ++t) {
        long long dwell_this_step = 0;
        
        #pragma omp parallel reduction(+:dwell_this_step)
        {
            XorshiftRNG rng(123456789 + omp_get_thread_num() + t * 10000);
            
            #pragma omp for
            for (int i = 0; i < N; i += 8) {
                // 处理剩余不足8个粒子的情况
                int batch_size = std::min(8, N - i);
                
                if (batch_size == 8) {
                    // SIMD路径: 一次处理8个粒子
                    
                    // 使用非对齐加载（更安全）
                    __m256i x_vec = _mm256_loadu_si256((__m256i*)&particles_x[i]);
                    __m256i y_vec = _mm256_loadu_si256((__m256i*)&particles_y[i]);
                    
                    // 检查是否在中心区域（无分支）
                    __m256i in_x_min = _mm256_cmpgt_epi32(x_vec, _mm256_sub_epi32(simd_center_min, simd_one));
                    __m256i in_x_max = _mm256_cmpgt_epi32(simd_center_max, x_vec);
                    __m256i in_y_min = _mm256_cmpgt_epi32(y_vec, _mm256_sub_epi32(simd_center_min, simd_one));
                    __m256i in_y_max = _mm256_cmpgt_epi32(simd_center_max, y_vec);
                    
                    __m256i in_center = _mm256_and_si256(
                        _mm256_and_si256(in_x_min, in_x_max),
                        _mm256_and_si256(in_y_min, in_y_max)
                    );
                    
                    // 计数（每个-1表示true）
                    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(in_center));
                    dwell_this_step += __builtin_popcount(mask);
                    
                    // 生成随机移动方向（8个）
                    uint32_t random_moves[8];
                    rng.next8(random_moves);
                    
                    // 移动粒子（标量路径，因为方向不同）
                    int x_arr[8], y_arr[8];
                    _mm256_storeu_si256((__m256i*)x_arr, x_vec);
                    _mm256_storeu_si256((__m256i*)y_arr, y_vec);
                    
                    for (int j = 0; j < 8; ++j) {
                        int dir = random_moves[j] & 3;  // 位运算代替 % 4
                        x_arr[j] = (x_arr[j] + dx[dir] + L) & L_mask;
                        y_arr[j] = (y_arr[j] + dy[dir] + L) & L_mask;
                    }
                    
                    // 存回
                    _mm256_storeu_si256((__m256i*)&particles_x[i], _mm256_loadu_si256((__m256i*)x_arr));
                    _mm256_storeu_si256((__m256i*)&particles_y[i], _mm256_loadu_si256((__m256i*)y_arr));
                    
                } else {
                    // 标量路径: 处理剩余粒子
                    for (int j = 0; j < batch_size; ++j) {
                        int idx = i + j;
                        int x = particles_x[idx];
                        int y = particles_y[idx];
                        
                        if (x >= center_min && x < center_max && 
                            y >= center_min && y < center_max) {
                            dwell_this_step++;
                        }
                        
                        int dir = rng.next() & 3;
                        x = (x + dx[dir] + L) & L_mask;
                        y = (y + dy[dir] + L) & L_mask;
                        
                        particles_x[idx] = x;
                        particles_y[idx] = y;
                    }
                }
            }
        }
        
        total_dwell_steps += dwell_this_step;
    }
    
    // 计时结束
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    // 计算 dwell ratio
    double dwell_ratio = static_cast<double>(total_dwell_steps) / (static_cast<long long>(N) * T);
    
    // 转换结果
    py::array_t<int> result_particles({N, 2});
    auto r = result_particles.mutable_unchecked<2>();
    
    for (int i = 0; i < N; ++i) {
        r(i, 0) = particles_x[i];
        r(i, 1) = particles_y[i];
    }
    
    // 返回 (particles, dwell_ratio, execution_time)
    return py::make_tuple(result_particles, dwell_ratio, elapsed_seconds);
}

// Python模块定义
PYBIND11_MODULE(random_walk_cpp_simd, m) {
    m.doc() = "Random Walk Simulation using C++ SIMD (AVX2)";
    
    m.def("run_simulation", &run_cpp_simd_simulation,
          "Run random walk simulation with C++ SIMD optimization",
          py::arg("L"), py::arg("N"), py::arg("T"),
          py::arg("center_min"), py::arg("center_max"));
}

