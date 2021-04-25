# BLAKE3-gpu
Parallelizing the BLAKE3 crypto hash function via the merkle tree structure

### Current best speedup :zap: -> 3x (with 8 physical cores)
### Compared to original blake3 -> 4x (slowdown)

## What
BLAKE3 is a gg crypto hash function. It has good scope for parallelism.  
We try to extract as much of that parallelism as possible by using GPUs.

## How 
- [x] Rewrite the basic, reference implemenation in C++
- [ ] Rewrite it again, in CUDA C++
- [ ] Make sure all the tests pass
- [ ] Optimize it, fix memory bandwidth issues if they exist

## Development
- The `basic` directory has the reference implementations.
- The blake3 paper is also here for reference.  
- Openmp work in `openmp`. It works fine, maybe try to move it to the GPU?
- Cuda work in `cuda`