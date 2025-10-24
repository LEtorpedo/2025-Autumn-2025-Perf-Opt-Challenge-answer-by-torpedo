# Makefile for CUDA Random Walk Simulation
#
# 使用方法:
#   make            - 编译基础版本
#   make advanced   - 编译高级优化版本
#   make all        - 编译所有版本
#   make clean      - 清理编译文件

# 编译器
NVCC = nvcc
CXX = g++

# CUDA架构 (根据你的GPU调整)
# 常见值:
#   sm_52  - Maxwell (GTX 900 系列)
#   sm_60  - Pascal (GTX 1000 系列)
#   sm_70  - Volta (V100)
#   sm_75  - Turing (RTX 2000 系列)
#   sm_80  - Ampere (A100, RTX 3000 系列)
#   sm_86  - Ampere (RTX 3090)
#   sm_89  - Ada Lovelace (RTX 4000 系列)
# 使用 native 让nvcc自动检测
CUDA_ARCH = -arch=native

# 编译选项
NVCCFLAGS = -O3 $(CUDA_ARCH) -use_fast_math --maxrregcount=32
CXXFLAGS = -O3 -march=native -std=c++17

# 链接选项
LDFLAGS = -lcudart

# 目标文件
TARGET_BASIC = simulate_cuda_basic
TARGET_ADVANCED = simulate_cuda_advanced

# 源文件
SRC_MAIN = main.cpp
SRC_KERNEL_BASIC = kernel.cu
SRC_KERNEL_ADVANCED = kernel_advanced.cu

# 对象文件
OBJ_MAIN = main.o
OBJ_KERNEL_BASIC = kernel.o
OBJ_KERNEL_ADVANCED = kernel_advanced.o

.PHONY: all basic advanced clean test help

# 默认目标
all: basic advanced

# 基础版本
basic: $(TARGET_BASIC)

$(TARGET_BASIC): $(OBJ_MAIN) $(OBJ_KERNEL_BASIC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# 高级优化版本
advanced: $(TARGET_ADVANCED)

$(TARGET_ADVANCED): main_advanced.o $(OBJ_KERNEL_ADVANCED)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# 编译主机代码
$(OBJ_MAIN): $(SRC_MAIN)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 高级版本的main（使用不同的核函数）
main_advanced.o: main_advanced.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 如果没有main_advanced.cpp，从main.cpp创建
main_advanced.cpp: $(SRC_MAIN)
	@echo "// Auto-generated from main.cpp for advanced kernel" > $@
	@echo "#define USE_ADVANCED_KERNEL" >> $@
	@cat $< >> $@

# 编译CUDA核函数
$(OBJ_KERNEL_BASIC): $(SRC_KERNEL_BASIC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_KERNEL_ADVANCED): $(SRC_KERNEL_ADVANCED)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 清理
clean:
	rm -f *.o $(TARGET_BASIC) $(TARGET_ADVANCED) main_advanced.cpp

# 测试（需要先编译）
test: basic
	@echo "Running basic CUDA simulation..."
	./$(TARGET_BASIC)

test_advanced: advanced
	@echo "Running advanced CUDA simulation..."
	./$(TARGET_ADVANCED)

# 显示GPU信息
gpu_info:
	@echo "=== GPU Information ==="
	nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
	@echo ""
	@echo "=== CUDA Version ==="
	nvcc --version

# 显示帮助
help:
	@echo "CUDA Random Walk Simulation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make            - Compile basic version"
	@echo "  make advanced   - Compile advanced version"
	@echo "  make all        - Compile all versions"
	@echo "  make clean      - Remove compiled files"
	@echo "  make test       - Run basic version"
	@echo "  make test_advanced - Run advanced version"
	@echo "  make gpu_info   - Show GPU information"
	@echo "  make help       - Show this help"
	@echo ""
	@echo "Usage examples:"
	@echo "  ./simulate_cuda_basic 512 100000 1000"
	@echo "  ./simulate_cuda_advanced 512 100000 1000"

