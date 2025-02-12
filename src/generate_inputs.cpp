// generate_inputs.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <iomanip>

// Simple CPU matrix multiplication (row-major ordering)
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

int main(int argc, char* argv[]) {
    // Default: 5 test cases of 4x4 matrices.
    int T = 5; // number of test cases
    int N = 4; // matrix dimension

    if (argc >= 3) {
        T = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    } else {
        std::cout << "Usage: " << argv[0] << " <num_test_cases> <matrix_dimension>\n";
        std::cout << "Using default values: " << T << " test cases, " << N << "x" << N << " matrices.\n";
    }
    
    std::ofstream ofs("matrix_test_cases.txt");
    if (!ofs) {
        std::cerr << "Error opening file for output.\n";
        return 1;
    }
    
    // Write a header: first line is "T N"
    ofs << T << " " << N << "\n";
    
    double totalCpuTime = 0.0;
    
    for (int t = 0; t < T; ++t) {
        std::vector<float> A(N * N);
        std::vector<float> B(N * N);
        std::vector<float> C(N * N, 0.0f);
        
        // Fill A and B with random floats in [0,1]
        for (int i = 0; i < N * N; ++i) {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
            B[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        // Measure CPU multiplication time for this test case
        auto start = std::chrono::high_resolution_clock::now();
        cpuMatrixMultiply(A, B, C, N);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuTime = end - start;
        totalCpuTime += cpuTime.count();
        
        std::cout << "Test case " << t+1 << ": CPU multiplication time = " 
                  << cpuTime.count() << " ms\n";
        
        // Write matrix A (each row on its own line)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                ofs << A[i * N + j] << " ";
            ofs << "\n";
        }
        ofs << "-\n";
        
        // Write matrix B
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                ofs << B[i * N + j] << " ";
            ofs << "\n";
        }
        ofs << "-\n";
        
        // Write computed matrix C (expected result)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                ofs << C[i * N + j] << " ";
            ofs << "\n";
        }
        ofs << "-\n";
    }
    
    ofs.close();
    
    return 0;
}
