import json
import subprocess
import os
import platform
import pytest


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

test_cases = []
for case in vectors["cases"]:
    lenx = case["input_len"]
    hashx = case["hash"]
    hashx_key = case["keyed_hash"]
    hashx_der = case["derive_key"]
    test_cases.append((lenx, hashx))

@pytest.mark.parametrize("lenx, hashx", test_cases)
def test_vector(lenx, hashx):
    filler_size = 1 + lenx // 251
    data = filler * filler_size
    data = b''.join(bytes([x]) for x in data[:lenx])
    with open('test.bin', 'wb') as wire:
        wire.write(data)
    process = subprocess.run([EXECUTABLE, "test.bin"], stdout=subprocess.PIPE)
    result = process.stdout.decode().strip()
    os.remove('test.bin')
    assert result == hashx[:64]
