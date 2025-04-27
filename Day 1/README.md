# 100 Days of CUDA - Day 1 - Vector Additon
Today I learned about Vector Addition
# Cuda Program Structure

```cudaMalloc``` allocates object in the device object memory. We need two parameters, Address of a pointer and the size of the allocated object in terms of bytes.

```cudaFree``` frees object from device global memory which requires pointer to the freed object.

## Grids
Threads in the Grid are organized into two levels of hierarchy.  At the top level, each grid consists of one or more thread blocks. All blocks in a grid have the same number of threads. Each block has a unique two dimentional coordinate given by keywords, ```blockIdx.x``` and ```blockIdx.y```. The threads in the same block can cooperate with each other by synchronizing their excution (for hazard free shared memory access). They share data through a low latecy shared memory. Two threads from two different blocks cannot cooperate. 

Each threads in a block is indentified by three indices, ```threadIdx.x```, ```threadIdx.y``` and ```threadIdx.z```

A thread in the matrix multiplication can be identified by,

```threadId = blockIdx.x * blockDim.x + threadIdx```

```blockDim``` and ```gridDim``` provide the dimention of the grid and the dimension of each block respectively.