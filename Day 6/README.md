# 100 Days of CUDA - Day 6 - Hardware Implementation

The NVIDIA GPU architecture is built around a scalable array of multi-threaded *Streaming Multiprocessors (SMs)*. When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to mutliprocessors with available execution capacity. The threads of a thread block execite concurrently on one multiprocessor, and multiple thread blocks are can execute concurrently on one multi-processor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors.

A multiprocessors is designed to execute hundred of threads concurrently. To manage such a large number of threads,it employs a unique architecture called SIMT (Singlr-Instruction, Multiple-Thread) that is described in SIMT architecture. The instructions are pipelined, leveraging instruction-level parallelism within a single thread, as well as extensive thread-level parallelism through simultaneous hardware multithreading as detailed in Hardware Multi-threading. Unline CPU cores, ther are issued in order and there is no branch prediction. They use *little-endian* representation.

## SIMT Architecture

The multiprocessor creates, manages, schedules, and execute threads in groups of 32 parallel threads called warps.
Individual threads composing a warp start togther at the same program address, but they have their own instruction address counter and regsiter state and therefore free to branch and execute independetly. 