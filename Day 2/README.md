# 100 Days of CUDA - Day 2 - Matrix Multiplication
## Thread Hierarchy (continued...)
![Grid of threads](../images/grid-of-thread-blocks.png)

The number of threads per block and the number of blocks per grid specicified in the ```<<<...>>>``` syntax can be of type ```int``` or ```dim3```.

Each block in the grid can be identified by a 1/2/3-dimentional unique index accesible within the kernel through the built-in ```blockIdx``` variable.

The dimention of thread block is accessible within the kernel through the builtin ```blockDim``` variable. (see code comment 1).

the grid is  created with enough blocks to have one thread per matrux element as before.

thread blocks are required to execute indenpendetly. It must be possible to executed then in any order, in parallel or series.

Threads within block can cooperate by sharing data through some shared memory and by synchronizing their execution to coordinate memory accesses. 

One can specifiy sync-points by using ```__syncthreads()```(this is force the threads to wait for their execution to finish before proceesding further.)

