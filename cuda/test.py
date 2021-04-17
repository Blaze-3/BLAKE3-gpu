import json
import subprocess
import os
import platform


if platform.system() == 'Windows':
    EXECUTABLE = './a.exe'
else:
    EXECUTABLE = './a.out'

with open('test_vectors.json') as red:
    vectors = json.load(red)

key = vectors["key"]
context = vectors["context_string"]

filler = [x for x in range(0, 251)]
total = 0
passed = 0

print("Running test cases")
for case in vectors["cases"]:
    total += 1
    lenx = case["input_len"]
    hashx = case["hash"]
    hashx_key = case["keyed_hash"]
    hashx_der = case["derive_key"]
    filler_size = 1 + lenx // 251
    data = filler * filler_size
    data = b''.join(bytes([x]) for x in data[:lenx])
    with open('test.bin', 'wb') as wire:
        wire.write(data)
    process = subprocess.run([EXECUTABLE, "test.bin"], stdout=subprocess.PIPE)
    result = process.stdout.decode().strip()
    passed += result == hashx[:64]
    os.remove('test.bin')

print(f"{passed} test cases passed out of {total}")