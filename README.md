# 100-Days-Cuda

## CPU vs GPU
The GPU is specialized for highly arallel computations and therefore designed such that more transistors are devoted to data processing rathet than data caching and flow control.

![CPU vs GPU](images/1.%20cpu_vs_gpu.png)

## Scalable Programming Model
At its core there are three key abstractions -  a hierarchy of thread groups, shared mempries, and barries synchronization - that are simply exposed to the programmer as a minimal set of language extensions.

![CUDA Hierarchy](images/2.%20automatic-scalability.png)

## 100 Days of CUDA Challange

Day 1: Overall GPU Architecture, Vector Addition

Day 2: Matrix Multiplication (Naive + Shared Memory)
    Todo: Thread block clusters

Day 3: Measureing Performance of a block

Day 4: Matrix x Vector = Vector

Day 5: Memory Hierarchy

Note: All images are taken from CUDA porgramming Guide. This repo is used for Note taking and practising purposes only.