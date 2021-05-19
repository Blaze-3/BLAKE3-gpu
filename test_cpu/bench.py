import subprocess
import os
import platform
import time
from pathlib import Path
import sys

REBUILD = False
if platform.system() == 'Windows':
    EXECUTABLE = './a.exe'
    SEQUENTIAL = 'bench/a.exe'
    ORIGINAL = "bench/b3sum.exe"
    COMPILER = "clang++ -O3"
else:
    EXECUTABLE = './a.out'
    SEQUENTIAL = 'bench/a.out'
    ORIGINAL = "bench/b3sum"
    COMPILER = "g++ -O3"

FILE_TO_HASH = 'bench/bencher.bin'

raw_file_size = sys.argv[1]
# Shortcut to generate large files
if raw_file_size[-1].isalpha():
    file_size = int(raw_file_size[:-1])
    modifier = raw_file_size[-1]
    file_size *= {
        "K": 1e+3,
        "M": 1e+6,
        "G": 1e+9
    }[modifier.upper()]
    file_size = int(file_size)
else:
    file_size = int(raw_file_size)
if not Path(FILE_TO_HASH).exists() or Path(FILE_TO_HASH).stat().st_size != file_size:
    with open(FILE_TO_HASH, 'wb') as wire:
        wire.write(os.urandom(file_size))
    print("Made a new file")

print("File size in bytes: ", Path(FILE_TO_HASH).stat().st_size)

# Before any execution generate the file
if REBUILD:
    os.system(COMPILER + ' main.cpp -fopenmp -o ' + EXECUTABLE)

start = time.time()
process = subprocess.run([EXECUTABLE, FILE_TO_HASH], stdout=subprocess.PIPE)
result = process.stdout.decode().strip()
print(result)
end = time.time()
exec_time = end-start

print("\x1b[35m", f"Execution time: {exec_time:.2f}s", "\033[0m")

if len(sys.argv) < 3:
    exit(0)

if sys.argv[2] in {"seq", "all"}:
    # Before any execution generate the file
    if REBUILD:
        os.system(COMPILER + ' main.cpp -o ' + SEQUENTIAL)
    start = time.time()
    process = subprocess.run([SEQUENTIAL, FILE_TO_HASH], stdout=subprocess.PIPE)
    result = process.stdout.decode().strip()
    print(result)
    end = time.time()
    exec_time = end-start
    print("\x1b[35m", f"Sequential execution time: {exec_time:.2f}s", "\033[0m")

if sys.argv[2] in {"og", "all"}:
    start = time.time()
    process = subprocess.run([
        ORIGINAL, FILE_TO_HASH, '--no-names'
    ], stdout=subprocess.PIPE)
    result = process.stdout.decode().strip()
    print(result)
    end = time.time()
    exec_time = end-start
    print("\x1b[35m", f"Original blake3 execution time: {exec_time:.2f}s", "\033[0m")