# CUDA version of BLAKE3

How to use:  
* Run `g++ main.cpp` or `clang++ main.cpp`
* To use CPU parallelism use `compiler main.cpp -fopenmp`
* Execute the file as `./a.{exe, out} input_file`

## Testing
### Test for correctness
Build the executable as mentioned above  
Run `python3 test.py`  

### Benchmark tests
Build the executalbe as mentioned above.  
If you want to compare with sequential execution build it as:  
`g++ bench/main.cpp -o bench/a.out`  
To run the benchmarks do - `python3 bench.py FILE_SIZE(in bytes) [seq|og|all]`  
Option `seq` will run the sequential version too.  
Option `og` will run the original blake3 and `all` will run all of them.