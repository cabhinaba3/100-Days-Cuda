#include<stdio.h>
#include<cuda_runtime.h>

/*
// kernel defintiions
__global__ void vecAdd(float *A, float *B, float *C){
    int index = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(){
    ...
    // kernel invocation with N threads
    vecAdd<<<1, N>>> (A,B,C);
}
Each of N threads that execute vecAdd() performs one pair-wise addition.
*/

/*
// Kernel definition
__global__ void matAdd(float a[n][n], float b[n][n], float c[n][n]){
    int i = threadIdx.x;
    int j = threadIdx.y;
    c[i][j] = a[i][j] + b[i][j];
}
int main(){
    // kernel invocation with one block of n * n * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(n,n);
    matAdd<<<numBlocks, threadsPerBlock>>>(a,b,c)
    // the number of thread blocks are limited to 1024(constant) as of now in 12.2
}
*/
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < numElements){
        C[index] = A[index] + B[index];
    }
}
int main(void){
    // Length of the vector
    const int numElements = 50000;
    // Declare the allocated size for float
    size_t size = numElements * sizeof(float);
    // Allocate host memory for vector A
    float *A = (float *)malloc(size);
    // Allocate the host memory for vector B
    float *B = (float *)malloc(size);
    for(int i=0;i<numElements;i++){
        A[i]=rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
        printf("%f %f\n",A[i],B[i]);
    }
    // Allocate the outout vector C
    float *C = (float *)malloc(size);

    // Allocate device input vector
    float *A_d = NULL, *B_d = NULL, *C_d = NULL;
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    printf("Copy input data from host memory to CUDA device\n");
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>> (A_d, B_d, C_d, numElements);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize(); // Some PTX Error for 4090
    // verify the result
    for(int i=0;i<numElements;++i){
        if(fabs(A[i]+B[i]-C[i])> 1e-6){
            printf("Result verification failed\n");
            // exit(0);
            // printf("%f ", C[i]);
        }else{
            printf("%f ",C[i]);
        }
    }
    printf("\n");
    // Free all the memory in GPU
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;

}