// Basic building blocks of blaze3
#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
#include <future>
#include <mutex>

#include <immintrin.h>
using namespace std;

using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;
 
const u32 OUT_LEN = 32;
const u32 KEY_LEN = 32;
const u32 BLOCK_LEN = 64;
const u32 CHUNK_LEN = 1024;
// Multiple chunks make a snicker bar :)
const u32 SNICKER = 1U << 7;
// Factory height and snicker size have an inversly propotional relationship
// FACTORY_HT * (log2 SNICKER) + 10 >= 64 
const u32 FACTORY_HT = 8;

const u32 CHUNK_START = 1 << 0;
const u32 CHUNK_END = 1 << 1;
const u32 PARENT = 1 << 2;
const u32 ROOT = 1 << 3;
const u32 KEYED_HASH = 1 << 4;

const int usize = sizeof(u32) * 8;
mutex factory_lock;
const int IS_ASYNC = 0;

u32 IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

const int MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

u32 rotr(u32 value, int shift) {
    return (value >> shift)|(value << (usize - shift));
}

__m128i _avx2_rotr(__m128i a, int shift) {
    int lshift = usize-shift;
    int rshift = shift;
    __m128i l = _mm_slli_epi32(a, lshift);
    __m128i r = _mm_srli_epi32(a, rshift);
    return _mm_or_si128(l, r);
}

void g(u32 state[16], u32 a, u32 b, u32 c, u32 d, u32 mx, u32 my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotr((state[d] ^ state[a]), 16);
    state[c] = state[c] + state[d];

    state[b] = rotr((state[b] ^ state[c]), 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rotr((state[d] ^ state[a]), 8);

    state[c] = state[c] + state[d];
    state[b] = rotr((state[b] ^ state[c]), 7);
}

void cols(u32 state[16], const u32 m[16]) {
    __m128i mxx = _mm_set_epi32(m[0], m[2], m[4], m[6]);
    __m128i myx = _mm_set_epi32(m[1], m[3], m[5], m[7]);

    __m128i statex[4] = {
        _mm_set_epi32(state[0],  state[1],  state[2],  state[3]),
        _mm_set_epi32(state[4],  state[5],  state[6],  state[7]),
        _mm_set_epi32(state[8],  state[9],  state[10], state[11]),
        _mm_set_epi32(state[12], state[13], state[14], state[15])
    };

    statex[0] = _mm_add_epi32(statex[0], statex[1]);
    statex[0] = _mm_add_epi32(statex[0], mxx);
    statex[3] = _mm_xor_si128(statex[3], statex[0]);
    statex[3] = _avx2_rotr(statex[3], 16);
    statex[2] = _mm_add_epi32(statex[2], statex[3]);
    statex[1] = _mm_xor_si128(statex[1], statex[2]);
    statex[1] = _avx2_rotr(statex[1], 12);
    statex[0] = _mm_add_epi32(statex[0], statex[1]);
    statex[0] = _mm_add_epi32(statex[0], myx);
    statex[3] = _mm_xor_si128(statex[3], statex[0]);
    statex[3] = _avx2_rotr(statex[3], 8);
    statex[2] = _mm_add_epi32(statex[2], statex[3]);
    statex[1] = _mm_xor_si128(statex[1], statex[2]);
    statex[1] = _avx2_rotr(statex[1], 7);

    for(int i=3, k=0; i>=0; i++) {
        state[k++] = _mm_extract_epi32(statex[i], 0);
        state[k++] = _mm_extract_epi32(statex[i], 1);
        state[k++] = _mm_extract_epi32(statex[i], 2);
        state[k++] = _mm_extract_epi32(statex[i], 3);
    }
}

void round(u32 state[16], const u32 m[16]) {
    // Mix the columns.
    // cols(state, m);
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    // cols(state, m);
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