# 100 Days of CUDA - Day 1 - Vector Additon
CUDA C++ extends C++ by allowing the programmer to define C++ functions, called kernels. C++ Fucntions <==> CUDA C++ Kernels.

A kernel is defined using ```__global__``` and the number of CUDA threads that execute the kernel for a given kernel call is specified using a new ```<<<...>>>```.

Each thead has its own threadID that is accesible within the kernel through build-in variables(see code comment 1).

## Thread Hierarchy
threadIdx is a 3-component vector, so that threads can be identified using one-dimentional, two-dimentional or three dimentional thread index, forming one dimentional, two/ three dimentional blocks of threads, called *thread block*.

The relation between index of a thread and threadID are as follows(see code comment 2),
<ol>
<li>For one dimentional block it is all the same</li>
<li>For two dimentional block of size, (Dx,Dy), the thread index (x,y) is (x+yDx)</li>
<li>For three dimentional block of size (Dx,DY,Dz), the thread IS of a thread index (x,y,z) is (x + yDx + z Dx Dy)</li>
</ol>

There is a limit to the number of threads in a block, since all threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited memory resources of that core. On current GPUs, a thread block may contain upto ```1024``` threads.

A kernel can be executed by multuple equally-shaped thread blocks.
```
Total threads = number of threads per block * number of blocks
```

Blocks are organised into 1/2/3-dimentional grid of thread-blocks.
![Thread Blocks](../images/grid-of-thread-blocks.png)