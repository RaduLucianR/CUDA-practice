// cuda_matrix_mult.cu
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication (row-major)
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k){
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// CPU matrix multiplication for verification and timing
void cpuMatrixMultiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
}

// Compare two matrices (elementwise) with a tolerance
bool compareMatrices(const std::vector<float>& mat1, const std::vector<float>& mat2, int N, float tolerance = 1e-2f) {
    for (int i = 0; i < N * N; ++i) {
        if (fabs(mat1[i] - mat2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // File name can be provided as a command line argument.
    std::string filename = "matrix_test_cases.txt";
    if (argc > 1)
        filename = argv[1];
    
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return 1;
    }
    
    // Read header: number of test cases and matrix dimension.
    int T, N;
    ifs >> T >> N;
    ifs.ignore(); // Skip rest of header line
    
    std::cout << "Number of test cases: " << T << ", Matrix dimension: " << N << "\n";
    
    float totalGpuTime = 0.0f;
    float totalCpuTime = 0.0f;
    
    // Process each test case
    for (int t = 0; t < T; t++) {
        std::vector<float> A(N * N);
        std::vector<float> B(N * N);
        std::vector<float> C_expected(N * N);
        std::vector<float> C_gpu(N * N, 0.0f);
        std::string line;
        
        // ---- Read matrix A (N lines) ----
        for (int i = 0; i < N; ++i) {
            std::getline(ifs, line);
            if (line.empty()) { 
                i--; 
                continue; 
            }
            std::istringstream iss(line);
            for (int j = 0; j < N; ++j) {
                iss >> A[i * N + j];
            }
        }
        std::getline(ifs, line); // read delimiter line (should be "-")
        
        // ---- Read matrix B (N lines) ----
        for (int i = 0; i < N; ++i) {
            std::getline(ifs, line);
            if (line.empty()) { 
                i--; 
                continue; 
            }
            std::istringstream iss(line);
            // Corrected: increment j, not i, in the inner loop
            for (int j = 0; j < N; ++j) {
                iss >> B[i * N + j];
            }
        }
        std::getline(ifs, line); // delimiter
        
        // ---- Read expected matrix C (N lines) ----
        for (int i = 0; i < N; i++) {
            std::getline(ifs, line);
            if (line.empty()) { 
                i--; 
                continue; 
            }
            std::istringstream iss(line);
            for (int j = 0; j < N; j++) {
                iss >> C_expected[i * N + j];
            }
        }
        std::getline(ifs, line); // delimiter
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        size_t bytes = N * N * sizeof(float);
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        
        // Copy A and B to device
        cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
        
        // Define kernel launch dimensions
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        // 10/16 + 15/16
        // Create CUDA events for timing the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Launch kernel and measure its execution time
        cudaEventRecord(start);
        matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalGpuTime += ms;
        
        // Copy result from device to host
        cudaMemcpy(C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);
        
        // Cleanup device memory and events
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // Also perform CPU multiplication (and measure its time)
        std::vector<float> C_cpu(N * N, 0.0f);
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpuMatrixMultiply(A, B, C_cpu, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuTime = cpu_end - cpu_start;
        totalCpuTime += cpuTime.count();
        
        // Compare the GPU result with the expected result from file.
        bool correct = compareMatrices(C_gpu, C_expected, N);
        if (!correct) {
            std::cerr << "Test case " << t+1 
                      << " FAILED: GPU result does not match expected result.\n";
        } else {
            std::cout << "Test case " << t+1 << " passed. ";
            std::cout << "GPU kernel time: " << ms << " ms, ";
            std::cout << "CPU multiplication time: " << cpuTime.count() << " ms.\n";
        }
    }
    
    return 0;
}
