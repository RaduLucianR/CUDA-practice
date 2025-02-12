// hello_world.cu
#include <cstdio>

__global__ void helloWorldKernel() {
    // This will print from the GPU. Ensure your device supports device-side printf.
    printf("Hello, World from CUDA kernel!\n");
}

int main() {
    // Launch the kernel with a single block and a single thread.
    helloWorldKernel<<<1, 1>>>();

    // Wait for the GPU to finish executing the kernel.
    cudaDeviceSynchronize();

    return 0;
}
