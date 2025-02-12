#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixMulSimple(float * A, float * B, float * C, int N) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;

    if (line < N && column < N) {
        float s = 0.0f;

        for (int i = 0; i < N; i ++) {
            s += A[line * N + i] * B[i * N + column];
        }

        C[line * N + column] = s;
    }
}

__global__ void matrixAdd(float * A, float * B, float * C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}


__global__ void applySobelKernel(float * A, float * B, float * C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int directions[3] = {-1, 0, 1};


    if (i < N - 1 && j < N - 1 && i > 0 && j > 0) {
        int center = i * N + j;
        int s = 0;
        
        for (int x : directions) {
            for (int y : directions) {
                s += A[(i + x) * N + (j + y)] * B[(x + 1) * 3 + (y + 1)];
            }
        }

        C[center] = s;
    }
}

void prettyPrintMatrix(vector<float>& matrix, int N) {
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N; j ++) {
            cout << matrix[i * N + j] << " ";
        }

        cout << "\n";
    }
}

void checkMatrix(vector<float>& result, vector<float>& target, int N) {
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N; j ++) {
            if (result[i * N + j] != target[i * N + j]) {
                cout << "Incorrect!\n";
                return;
            }
        }
    }

    cout << "Correct!\n";
}

vector<float> readMatrix(ifstream& ifs, int N) {
    vector<float> matrix(N * N);

    int i = 0;
    while (i < N * N && ifs) {
        float value;
        if (ifs >> value) {
            matrix[i] = value;
            i ++;
        } else {
            // Extraction failed, skip element
            ifs.clear();
            string skip;
            ifs >> skip; 
        }
    }
    
    return matrix;
}

vector<float> applyConvolution(const vector<float>& input, int N) {
    // Create an output vector initialized to 0.
    vector<float> output(N * N, 0.0f);

    // Define the Sobel kernel for horizontal edge detection.
    int kernel[3][3] = {
        { 1,  0, -1},
        { 2,  0, -2},
        { 1,  0, -1}
    };

    // Convolve the kernel over the image, skipping the border pixels.
    for (int row = 1; row < N - 1; ++row) {
        for (int col = 1; col < N - 1; ++col) {
            float sum = 0.0f;
            // Apply the kernel.
            for (int kRow = -1; kRow <= 1; ++kRow) {
                for (int kCol = -1; kCol <= 1; ++kCol) {
                    int currentRow = row + kRow;
                    int currentCol = col + kCol;
                    sum += input[currentRow * N + currentCol] * kernel[kRow + 1][kCol + 1];
                }
            }
            output[row * N + col] = sum;
        }
    }

    return output;
}

int main() {
    string file = "matrix_test_cases.txt";
    ifstream ifs(file);
    string dummy = "";
    int nrOfInputMuls = 0;
    int N = 0; // Size of one square matrix
    ifs >> nrOfInputMuls;
    ifs >> N;
    cout << "Number of inputs: " << nrOfInputMuls << " and matrices of size: " << N << "x" << N << "\n";

    // Store the matrices as 1-dimensional arrays
    // Initialize the matrices with 0s via the vector constructor
    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N);

    // Read inputs and expected result
    A = readMatrix(ifs, N);
    getline(ifs, dummy);
    B = {1, 0, -1, 2, 0, -2, 1, 0, -1}; // Sobel kernel
    // B = readMatrix(ifs, N);
    // getline(ifs, dummy);
    // C = readMatrix(ifs, N);
    
    // Check they are read correctly
    prettyPrintMatrix(A, N);
    // prettyPrintMatrix(B, N);
    prettyPrintMatrix(B, 3);
    // prettyPrintMatrix(C, N);

    // Allocate memory on the device (i.e. on the GPU - is it actually *the* GPU?)
    // We are now on the host (i.e. the CPU)
    // We have matrices that are stored as arrays, thus we need a pointer to an array for a matrix
    float * d_A; // d_A i.e. A on the device
    float * d_B;
    float * d_C;

    // This also modifies d_A to an address in the GPU memory space (actually is this true?)
    cudaMalloc(&d_A, N * N * sizeof(float)); // cudaMalloc(pointer, size in *bytes* not int/long)
    // cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_B, 3 * 3 * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float)); // We also need to allocate memory for the result

    // Now we allocated the necessary memory on the GPU
    // but we have nothing in that memory, so we need to fill it with our data
    // so we want to transfer the data from the host to the device

    // cudaMemcpy takes the destination memory address, the 
    cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    /**
     * Steps:
     * - Create data on the host
     * - Allocate memory on the device
     * - Transfer the data to the device from the host
     * - Define the kernel dimensions i.e. number of blocks and number of threads per block
     * - Call the kernel
     * - Synchronize
     * - Copy result to host
     * - Free memory on GPU
     */

    // Define kernel dimensions
    dim3 threadsPerBlock(16, 16);
    // We want to cover a 2D matrix with 16x16 squares
    // So we can calculate how many squares we need to cover one dimension
    // and do this for the other dimension as well
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Kernel syntax: kernel_name<<<gridDim, BlockDim>>>(arguments);
    // matrixMulSimple<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    applySobelKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);


    // Wait until the device finishes the operation
    cudaDeviceSynchronize();

    // Retrive the result from the device to the host
    cudaMemcpy(C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";
    }

    // Free the used memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    prettyPrintMatrix(C, N);
    vector<float> conv_res = applyConvolution(A, N);
    checkMatrix(C, conv_res, N);
}