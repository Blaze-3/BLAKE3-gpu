#include <iostream>
using namespace std;

using u32 = unsigned int;
using u64 = unsigned long;
 
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

void round(u32 state[16], u32 m[16]) {

}

void permute(u32 m[16]) {
    u32 permuted[16];
    for(int i=0; i<16; i++)
        permuted[i] = m[MSG_PERMUTATION[i]];
    for(int i=0; i<16; i++)
        m[i] = permuted[i];
}

u32* compress(
    u32 chaining_value[8],
    u32 block_words[16],
    u64 counter,
    u32 block_len,
    u32 flags
) {
    u32 *state = new u32[16]{
        chaining_value[0],
        chaining_value[1],
        chaining_value[2],
        chaining_value[3],
        chaining_value[4],
        chaining_value[5],
        chaining_value[6],
        chaining_value[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        u32(counter),
        u32(counter >> 32),
        block_len,
        flags,
    };
    u32 block[16];
    copy(block_words, block_words+16, block);

    round(state, block); // round 1
    permute(block);
    round(state, block); // round 2
    permute(block);
    round(state, block); // round 3
    permute(block);
    round(state, block); // round 4
    permute(block);
    round(state, block); // round 5
    permute(block);
    round(state, block); // round 6
    permute(block);
    round(state, block); // round 7

    for(int i=0; i<8; i++){
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }

    return state;
}

int main() {
    cout << "cats\n";
}
