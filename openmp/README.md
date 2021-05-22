# Openmp version of BLAKE3

## Usage
How to use:  
* Run `g++ -Wall main.cpp` or `clang++ -Wall main.cpp`
* To use CPU parallelism use `compiler main.cpp -fopenmp`
* Execute the file as `./a.{exe, out} input_file`  
* Remember to use the flags - `-march=native` to try out SIMD
**Enhancement! -> Optimize the binary with -O3 while building the executable**  

## Testing
### Test for correctness
Build the executable as mentioned above  
Run `pytest -q -r N --tb=no`  

### Benchmark tests 
* To build the sequential version, compile without `-fopenmp` and place the binary
in the bench folder.
* To run the benchmarks do - `python3 bench.py FILE_SIZE(in bytes) [seq|og|all]`  
Option `seq` will run the sequential version too.  
Option `og` will run the original blake3.  
Option `all` will run all of them.