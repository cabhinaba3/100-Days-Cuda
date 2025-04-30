// This example shows how to use the clock function to measure the performance of block of threads of kernels accurately

#include<assert.h>
#include<stdint.h>
#include<stdio.h>

// CUDA Runtime
#include<cuda_runtime.h>
// Helper functions and utilities to work with CUDA
// #include<helper_cuda.h>
// #include<helper_functions.h>

// This kernel computes a standard parallel reduction and evaluates the time it takes to do that for each block.
// The timing results are stored in device memory


__global__ static void timeReduction(const float *input, float *output, clock_t *timer){
    // shared __float shared[2* blockDim.x]
    extern __shared__ float shared[];
    const int tid = threadIdx.x, bid=blockIdx.x;
    if(tid == 0){ timer[bid] = clock();}

    // copy input
    shared[tid] = input[tid];
    shared[tid + blockIdx.x] = input[tid + blockDim.x];

    // perform reduction to find minimum
    for(int d=blockDim.x; d>0; d/=2){
        __syncthreads;
        if(tid < 0){
            float f0 = shared[tid], f1 = shared[tid + d];
            if(f1<f0){ shared[tid] = f1; }
        }
    }
    // Write result
    if(tid == 0) { output[bid] = shared[0]; }
    __syncthreads();
    if(tid == 0) { timer[bid + gridDim.x] = clock(); }
}

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(int argc, char *argv[]){
    printf("CUDA Clock sample\n");

    float *dinput = NULL, *doutput = NULL; 
    clock_t *dtimer = NULL;
    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    for(int i=0;i< NUM_THREADS * 2; ++i){
        input[i] = (float)i;
    }

    cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2);
    cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS);
    cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2);
    cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2,cudaMemcpyHostToDevice);

    timeReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) *2* NUM_THREADS >>>(dinput, doutput, dtimer);

    cudaMemcpy(timer, dtimer, sizeof(clock_t)*NUM_BLOCKS*2, cudaMemcpyDeviceToHost);
    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dtimer);

    long double avgClocks = 0;
    for(int i=0;i<NUM_BLOCKS; ++i){
        avgClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }
    avgClocks /= NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n",avgClocks);
    return EXIT_SUCCESS;

}
