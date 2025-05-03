#include <stdio.h>
#include <assert.h>
#include <chrono> 
#include <iostream>
#include "driver_types.h"
#include <cuda_runtime.h>
// #include "helper_cuda.h"

/*

// kernel definition
__global void matAdd(float a[n][n], float b[n][n], float c[n][n]){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.yl
    if(i<n && j<n){
        c[i][j]=a[i][j]+b[i][j];
    }
}
int main(){
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(n / threadsPerBlock.x, n/threadPerBlock.y);
    matAdd<<<numBlocks, threadPerBlock>>> (a,b,c);
}

// the thread block size of 16x16 is common choice.
*/
#define BLOCK_SIZE 16
#define TILE_SIZE 16

struct Matrix{
    int height;
    int width;
    float *elements;
};
#define CUDA_CHECK_ERROR(call)
{
    cudaError_t err = call;                                          
    if (err != cudaSuccess) {                                        
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; 
        std::cerr << " code=" << static_cast<int>(err)               
                    << " " << cudaGetErrorString(err) << " " << std::endl; 
        exit(1);                                                    
    }                                                               
}
void matrixMultiplicationCPU(const Matrix &A, const Matrix &B, Matrix &C){
    for(int i=0; i<A.height; ++i){
        for(int j=0; j<B.width; ++j){
            float cValue = 0;
            for(int k=0; k<A.width; ++k){
                cValue += A.elements[i*A.width + k] * B.elements[k*B.width + j];
            }
            C.elements[i*C.width + j] = cValue;
        }
    }
}


void initializeMatrix(Matrix &X){
    for(int i=0;i< X.height * X.width;i++){
        X.elements[i] = rand() % 10+0.0f;
    }
}
void printMatrix(Matrix X){
    for(int i=0;i<X.height * X.width;++i){
        printf("%f ",X.elements[i]);
    }
    printf("n");
}
// __global__ keyword is defined as a CUDA kernel
// threads and blocks are indexed  using the built-in 3D variable : threadIdx, blockIdx
// blockDim: gives the dimention of thread block
// combination of these can be used to find the element in the thread

__global__ void matMulNaiveMultiplication(Matrix A, Matrix B, Matrix C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.y + threadIdx.x;

    // Each thread accumulates one element of C by accumulating results into cValue
    float cValue = 0;
    // Now we can iterate over common dimentions of A and B (k = A.width = B.height)
    if(row < A.height && col< B.height){
        for(int k=0;k<A.width;++k){
            cValue += A.elements[row * A.width + k] * B.elements[k * B.width +col];
        }
        C.elements[row * C.width + col] = cValue;
    }
}

// Kernel for matrix multiplication using tiling and shared memory
__global__ void matMulSharedMemoryKernel(Matrix A, Matrix B, Matrix C)
{
    // Shared memory for tiles of A and B
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // Calculate the global row and column index of the element
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over the tiles of the input matrices
    // A.width/TILE_SIZE and B.height/TILE_SIZE; take care of the last tile
    for (int m = 0; m < (A.width + TILE_SIZE - 1) / TILE_SIZE; ++m)
    {
        // Load elements of A into shared memory
        // if shared memory defined using 1d array, we'd have used shared_A[row * TILE_SIZE + col]
        if (row < A.height && (m * TILE_SIZE + col) < A.width) 
        {
            shared_A[row][col] = A.elements[globalRow * A.width + m * TILE_SIZE + col];
        } else 
        {
            // When matrix dimensions are not exact multiples of the tile size,
            // some threads in the last blocks might access elements outside
            // the matrix boundaries. By setting out-of-bounds elements to zero,
            // we ensure that these threads do not contribute invalid values to final result.
            // e.g. Matrix A = [100x100] and TILE_SIZE = 16
            shared_A[row][col] = 0.0f;
        }
        // Load elements of B into shared memory
        if (col < B.width && (m * TILE_SIZE + row) < B.height) 
        {
            shared_B[row][col] = B.elements[(m * TILE_SIZE + row) * B.width + globalCol];
        } else 
        {
            shared_B[row][col] = 0.0f;
        }
        // Synchronize to ensure all threads have loaded their elements
        __syncthreads();

        // Compute the partial result
        for (int k = 0; k < TILE_SIZE; ++k)
            Cvalue += shared_A[row][k] * shared_B[k][col];

        // Synchronize to ensure all threads have completed the computation
        __syncthreads();
    }

    // Write the result to global memory
    if (globalRow < C.height && globalCol < C.width)
        C.elements[globalRow * C.width + globalCol] = Cvalue;
}
void runKernel(void(*kernel)(Matrix, Matrix, Matrix),
               const Matrix &A, const Matrix &B, Matrix &C,
               dim3 gridDim, dim3 blockDim)
{
    // Load matrices to device memory
    Matrix d_A, d_B, d_C;
    size_t size_A = A.width * A.height * sizeof(float);
    size_t size_B = B.width * B.height * sizeof(float);
    size_t size_C = C.width * C.height * sizeof(float);
    d_A.width = A.width; d_A.height = A.height;
    d_B.width = B.width; d_B.height = B.height;
    d_C.width = C.width; d_C.height = C.height;

    // Allocate device memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_A.elements, size_A));

    CUDA_CHECK_ERROR(cudaMalloc(&d_B.elements, size_B));

    CUDA_CHECK_ERROR(cudaMalloc(&d_C.elements, size_C));

    // Copy A, B to device memory
    // Profile data copy
    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_ERROR(cudaMemcpy(d_A.elements, A.elements, size_A, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    printf("-Copying A took: %f ms\n",duration.count() * 1000.0f);

    start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_ERROR(cudaMemcpy(d_B.elements, B.elements, size_B, cudaMemcpyHostToDevice));
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    printf("-Copying B took: %f ms\n",duration.count() * 1000.0f);


    // Launch kernel
    start = std::chrono::high_resolution_clock::now();
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    // Synchronize device memory
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "-Kernel execution time + synctime: " << duration.count() * 1000.0f << " ms" << std::endl;

    // Copy C from device memory to host memory
    start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_ERROR(cudaMemcpy(C.elements, d_C.elements, size_C, cudaMemcpyDeviceToHost));
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    printf("-Copying C took: %f ms\n",duration.count() * 1000.0f);

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(d_A.elements));
    CUDA_CHECK_ERROR(cudaFree(d_B.elements));
    CUDA_CHECK_ERROR(cudaFree(d_C.elements));
}
int main(){
    const int M = 1024;
    const int K = 768;
    const int N = 1024;

    // Allocate matrics A, B ,C
    Matrix A = {M, K, new float[M * K]}; // 1024x768
    Matrix B = {K, N, new float[K * N]}; // 768x1024

    Matrix C = {M, N, new float[M * N]};

    // initialize metrics A and B with random values
    initializeMatrix(A);
    initializeMatrix(B);

    // printMatrix(A);

    // Measure the time taken for matric multiplication on the CPU
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplicationCPU(A,B,C);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = stop - start;

    std::cout<< "CPU Matrix Multiplication time: "<<duration.count() * 1000.0f <<" ms"<<std::endl;

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((B.width + threadsPerBlock.x-1) / threadsPerBlock.x, (A.height + threadsPerBlock.y -1)/ threadsPerBlock.y);
    printf("Naive Matrix Multiplication Using GPU---\n");
    runKernel(matMulNaiveMultiplication,A, B, C, blocksPerGrid, threadsPerBlock);

    printf("Shared Memory Model:---\n");
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((C.width + TILE_SIZE-1)/TILE_SIZE, (C.height + TILE_SIZE-1)/TILE_SIZE);
    runKernel(matMulSharedMemoryKernel, A,B,C, gridDim, blockDim);
    delete[] A.elements;
    delete[] B.elements;
    delete[] C.elements;
    return 0;
}