import subprocess
import os
import platform
import time
from pathlib import Path
import sys


if platform.system() == 'Windows':
    EXECUTABLE = './a.exe'
    SEQUENTIAL = 'bench/a.exe'
    ORIGINAL = "bench/b3sum.exe"
else:
    EXECUTABLE = './a.out'
    SEQUENTIAL = 'bench/a.out'
    ORIGINAL = "bench/b3sum"

FILE_TO_HASH = 'bench/bencher.bin'

file_size = int(sys.argv[1])
if not Path(FILE_TO_HASH).exists() or Path(FILE_TO_HASH).stat().st_size != file_size:
    with open(FILE_TO_HASH, 'wb') as wire:
        wire.write(os.urandom(file_size))
    print("Made a new file")

print("File size in bytes: ", Path(FILE_TO_HASH).stat().st_size)

start = time.time()
process = subprocess.run([EXECUTABLE, FILE_TO_HASH], stdout=subprocess.PIPE)
result = process.stdout.decode().strip()
print(result)
end = time.time()
exec_time = end-start

print("\x1b[32m", f"Execution time: {exec_time:.2f}s", "\033[0m")

if len(sys.argv) < 3:
    exit(0)

if sys.argv[2] in {"s", "all"}:
    start = time.time()
    process = subprocess.run([SEQUENTIAL, FILE_TO_HASH], stdout=subprocess.PIPE)
    result = process.stdout.decode().strip()
    print(result)
    end = time.time()
    exec_time = end-start
    print("\x1b[32m", f"Sequential execution time: {exec_time:.2f}s", "\033[0m")

if sys.argv[2] in {"og", "all"}:
    start = time.time()
    process = subprocess.run([
        ORIGINAL, FILE_TO_HASH, '--no-names'
    ], stdout=subprocess.PIPE)
    result = process.stdout.decode().strip()
    print(result)
    end = time.time()
    exec_time = end-start
    print("\x1b[32m", f"Original blake3 execution time: {exec_time:.2f}s", "\033[0m")