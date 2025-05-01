#include<iostream>
// In this example we will implement a simple Matrix x Vector

__global__ void vectorMatrixMultiplication(const float *A, const float *B, float* C,int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        float sum = 0.0f;
        for(int j=0;j<size;j++){ sum += A[index * size +j] * B[j]; }
        C[index] = sum;
    }
}
int main(){
    const int size = 5;
    int matrixSize = size * size * sizeof(float);
    int vectorSize = size * sizeof(float);
    float *A, *B, *C;

    // Initialize the input metrics
    A = (float *)malloc(matrixSize);
    B = (float *)malloc(vectorSize);
    C = (float *)malloc(vectorSize);

    for(int i=0;i<size; ++i){
        for(int j=0;j<size; ++j){
            A[i * size + j] = (i+j) % 100;
        }
        B[i] = i % 50;
        C[i] = 0.0;
    }

    float *d_a, *d_b, *d_c;
    // allocate memory for matrix, vector and output vector
    cudaMalloc(&d_a, matrixSize);
    cudaMalloc(&d_b, vectorSize);
    cudaMalloc(&d_c, vectorSize);

    // copy the Matrix A and matrix B to GPU
    cudaMemcpy(d_a, A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, vectorSize, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = ( size + blockSize - 1) / blockSize;
    vectorMatrixMultiplication<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Likely PTX for my GPU(check)
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    // synchronize the device and host
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_c, vectorSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) { printf("%.2f ", A[i * size + j]); }
        printf("\n");
    }
    for (int i = 0; i < size; i++) { printf("%f ",C[i]); }
    printf("\n");

    // free the device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}