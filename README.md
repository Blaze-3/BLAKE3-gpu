# BLAKE3-gpu
Parallelizing the BLAKE3 crypto hash function via the merkle tree structure


## What
BLAKE3 is a gg crypto hash function. It has good scope for parallelism.  
We try to extract as much of that parallelism as possible by using GPUs.

## How 
- [ ] Rewrite the basic, reference implemenation in C++
- [ ] Rewrite it again, in CUDA C++
- [ ] Make sure all the tests pass
- [ ] Optimize it, fix memory bandwidth issues if they exist
