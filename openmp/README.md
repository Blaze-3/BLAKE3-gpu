# Openmp version of BLAKE3

## Usage
How to use:  
* Run `g++ main.cpp` or `clang++ main.cpp`
* To use CPU parallelism use `compiler main.cpp -fopenmp`
* Execute the file as `./a.{exe, out} input_file`  
**Enhancement! -> Optimize the binary with -O3 while building the executable**  

## Testing
### Test for correctness
Build the executable as mentioned above  
Run `pytest -q -r N --tb=no`  

### Benchmark tests
Build the executalbe as mentioned above.  
If you want to compare with sequential execution build it as:  
`g++ bench/main.cpp -o bench/a.out`  
To run the benchmarks do - `python3 bench.py FILE_SIZE(in bytes) [seq|og|all]`  
Option `seq` will run the sequential version too.  
Option `og` will run the original blake3 and `all` will run all of them.