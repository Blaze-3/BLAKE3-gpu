# BLAKE3-gpu
Parallelizing the BLAKE3 crypto hash function via its merkle tree structure.  
Check [Presentation](./final-presentation.pdf) for a complete explanation.

### Current best speedup :zap: -> 4.5x at 1.07 GiB/s on an octacore

## What
BLAKE3 is a gg crypto hash function. It has good scope for parallelism.  
We try to extract as much of that parallelism as possible by using GPUs.  
We also try to speed it up on the CPU with Open-MP and AVX2.  
All of this is possible due to our new algorithm - Blaze3.

## How 
- [x] Rewrite the basic, reference implemenation in C++
- [x] Rewrite it again, in CUDA C++
- [x] Make sure all the tests pass (Continuous process)
- [x] Optimize it, fix memory bandwidth issues if they exist (Continuous process)

## Development
- The `basic` directory has the reference implementations.
- A full copy of the original reference implementation is in `testing`.
- The blake3 paper is also here for reference.  
- Openmp work in `openmp`. This version is maxed out for efficency.
- Cuda work in `cuda`. This version uses dynamic parallelism.
- Dark cuda work happens in `dark`.
