# BLAKE3-gpu
Parallelizing the BLAKE3 crypto hash function via the merkle tree structure

### Current best speedup :zap: -> 4.5x at 1.07 GiB/s on an octacore

## What
BLAKE3 is a gg crypto hash function. It has good scope for parallelism.  
We try to extract as much of that parallelism as possible by using GPUs.

## How 
- [x] Rewrite the basic, reference implemenation in C++
- [x] Rewrite it again, in CUDA C++
- [x] Make sure all the tests pass (Continuous process)
- [ ] Optimize it, fix memory bandwidth issues if they exist

## Development
- The `basic` directory has the reference implementations.
- The blake3 paper is also here for reference.  
- Openmp work in `openmp`. This version is maxed out for efficency.
- Cuda work in `cuda`
- Dark cuda work happens in `dark`