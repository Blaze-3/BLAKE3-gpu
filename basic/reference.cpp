#include <bits/stdc++.h>
using namespace std;

 
unsigned int OUT_LEN = 32;
unsigned int KEY_LEN = 32;
unsigned int BLOCK_LEN = 64;
unsigned int CHUNK_LEN = 1024;

unsigned int CHUNK_START = 1 << 0;
unsigned int CHUNK_END = 1 << 1;
unsigned int PARENT = 1 << 2;
unsigned int ROOT = 1 << 3;
unsigned int KEYED_HASH = 1 << 4;
unsigned int DERIVE_KEY_CONTEXT = 1 << 5;
unsigned int DERIVE_KEY_MATERIAL = 1 << 6;

unsigned int IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

unsigned int MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

int main() {
    cout << "cats\n";
}
