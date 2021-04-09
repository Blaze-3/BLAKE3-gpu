#include <iostream>
using namespace std;

#define INT_BITS sizeof(int)*8

using u32 = unsigned int;
using u64 = unsigned long;
using u8  = unsigned char;
 
const u32 OUT_LEN = 32;
const u32 KEY_LEN = 32;
const u32 BLOCK_LEN = 64;
const u32 CHUNK_LEN = 1024;

const u32 CHUNK_START = 1 << 0;
const u32 CHUNK_END = 1 << 1;
const u32 PARENT = 1 << 2;
const u32 ROOT = 1 << 3;
const u32 KEYED_HASH = 1 << 4;
const u32 DERIVE_KEY_CONTEXT = 1 << 5;
const u32 DERIVE_KEY_MATERIAL = 1 << 6;

unsigned int IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

unsigned int MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

int leftRotate(int n, unsigned int d)
{
      
    /* In n<<d, last d bits are 0. To
     put first 3 bits of n at 
    last, do bitwise or of n<<d 
    with n >>(INT_BITS - d) */
    return (n << d)|(n >> (INT_BITS - d));
}

int rightRotate(int n, unsigned int d)
{
    /* In n>>d, first d bits are 0. 
    To put last 3 bits of at 
    first, do bitwise or of n>>d
    with n <<(INT_BITS - d) */
    return (n >> d)|(n << (INT_BITS - d));
}

void g (u32 state[16], u32 a, u32 b, u32 c, u32 d, u32 mx, u32 my)
{
    state[a]=state[a]+state[b]+state[mx];
    state[d]=rightRotate((state[d] ^ state[a]),16);
    state[c]=state[c]+state[d];

    state[b]=rightRotate((state[b] ^ state[c]),12);
    state[a]=state[a]+state[b]+my;
    state[d]=rightRotate((state[d] ^ state[a]),8);

    state[c]=state[c]+state[d];
    state[b]=rightRotate((state[b] ^ state[c]),7);
}

void round(u32 state[16], u32 m[16])
{
    // Mix the columns.
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
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

u32* first_8_words(u32 compression_output[16]) {
    u32 *cmprs = new u32[8];
    copy(compression_output, compression_output+8, cmprs);
    return cmprs;
}

void words_from_little_endian_bytes(u8* bytes, u32* words) {

}

struct Output {
    u32 input_chaining_value[8];
    u32 block_words[16];
    u64 counter;
    u32 block_len;
    u32 flags;

    // methods
    u32* chaining_value() {
        return first_8_words(compress(
            input_chaining_value,
            block_words,
            counter,
            block_len,
            flags
        ));
    }
    void root_output_bytes(u8* out_slice);
};

struct ChunkState {
    u32 chaining_value[8];
    u64 chunk_counter;
    u8 block[BLOCK_LEN];
    u8 block_len, blocks_compressed, flags;

    // methods
    ChunkState(u32 key[8], u64 chunk_counter, u32 flags);
    size_t len();
    u32 start_flag();
    void update(u8* input);
    Output output();
};

Output parent_output(u32 left_child_cv[8], u32 right_child_cv[8],
u32 key[8], u32 flagse) {

}

u32* parent_cv(u32 left_child_cv[8], u32 right_child_cv[8],
u32 key[8], u32 flags) {
    return parent_output(left_child_cv, right_child_cv, key, flags).chaining_value();
}

struct Hasher {
    ChunkState chunk_state;
    u32 key[8], cv_stack[54][8], flags;
    u8 cv_stack_len;

    // methods
    Hasher new_internal(u32 key[8], u32 flags);
    // Hasher new();
    Hasher new_keyed(u8 key[KEY_LEN]);
    Hasher new_derive_key(string context);
    void push_stack(u32 cv[8]);
    u32 *pop_stack();
    void add_chunk_chaining_value(u32 new_cv[8], u64 total_chunks);
    void update(u8* input);
    void finalize(u8* out_slice);
};

int main() {
    cout << "cats\n";
}
