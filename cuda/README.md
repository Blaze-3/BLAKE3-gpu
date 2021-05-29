# CUDA version of BLAKE3

## Usage
How to use:  
* Run `nvcc -std=c++17 -rdc=true main.cu`
* Execute the file as `./a.{exe, out} input_file`   

## Testing
### Test for correctness
Build the executable as mentioned above  
Run `pytest -q -r N --tb=no`  

### Benchmark tests
* The sequential binary can be built from the openmp folder and copied to `bench`
* To run the benchmarks do - `python3 bench.py FILE_SIZE(in bytes) [seq|og|all]`  
Option `seq` will run the sequential version too.  
Option `og` will run the original blake3.  
Option `all` will run all of them.