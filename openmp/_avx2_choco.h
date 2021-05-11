#ifndef _AVX2_CHOCO_H
#define _AVX2_CHOCO_H

#include <iostream>
using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;

#include <immintrin.h>

inline __m128i _avx2_rotr(__m128i a, int shift) {
    // lshift = usize-shift
    // but usize is always 32
    int lshift = 32-shift;
    int rshift = shift;
    return _mm_or_si128(
        _mm_slli_epi32(a, lshift),
        _mm_srli_epi32(a, rshift)
    );
}

void _avx2_g0(u32 state[16], const u32 m[16]) {
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

    state[15] = _mm_extract_epi32(statex[3], 0);
    state[14] = _mm_extract_epi32(statex[3], 1);
    state[13] = _mm_extract_epi32(statex[3], 2);
    state[12] = _mm_extract_epi32(statex[3], 3);

    state[11] = _mm_extract_epi32(statex[2], 0);
    state[10] = _mm_extract_epi32(statex[2], 1);
    state[9]  = _mm_extract_epi32(statex[2], 2);
    state[8]  = _mm_extract_epi32(statex[2], 3);

    state[7] = _mm_extract_epi32(statex[1], 0);
    state[6] = _mm_extract_epi32(statex[1], 1);
    state[5] = _mm_extract_epi32(statex[1], 2);
    state[4] = _mm_extract_epi32(statex[1], 3);

    state[3] = _mm_extract_epi32(statex[0], 0);
    state[2] = _mm_extract_epi32(statex[0], 1);
    state[1] = _mm_extract_epi32(statex[0], 2);
    state[0] = _mm_extract_epi32(statex[0], 3);
}

void _avx2_g1(u32 state[16], const u32 m[16]) {
    __m128i mxx = _mm_set_epi32(m[8], m[10], m[12], m[14]);
    __m128i myx = _mm_set_epi32(m[9], m[11], m[13], m[15]);

    __m128i statex[4] = {
        _mm_set_epi32(state[0],  state[1],  state[2],  state[3]),
        _mm_set_epi32(state[5],  state[6],  state[7],  state[4]),
        _mm_set_epi32(state[10], state[11], state[8],  state[9]),
        _mm_set_epi32(state[15], state[12], state[13], state[14])
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

    state[3] = _mm_extract_epi32(statex[0], 0);
    state[2] = _mm_extract_epi32(statex[0], 1);
    state[1] = _mm_extract_epi32(statex[0], 2);
    state[0] = _mm_extract_epi32(statex[0], 3);

    state[4] = _mm_extract_epi32(statex[1], 0);
    state[7] = _mm_extract_epi32(statex[1], 1);
    state[6] = _mm_extract_epi32(statex[1], 2);
    state[5] = _mm_extract_epi32(statex[1], 3);

    state[9]  = _mm_extract_epi32(statex[2], 0);
    state[8]  = _mm_extract_epi32(statex[2], 1);
    state[11] = _mm_extract_epi32(statex[2], 2);
    state[10] = _mm_extract_epi32(statex[2], 3);

    state[14] = _mm_extract_epi32(statex[3], 0);
    state[13] = _mm_extract_epi32(statex[3], 1);
    state[12] = _mm_extract_epi32(statex[3], 2);
    state[15] = _mm_extract_epi32(statex[3], 3);
}

void round(u32 state[16], const u32 m[16]) {
    // Mix the columns.
    _avx2_g0(state, m);
    // Mix the diagonals.
    _avx2_g1(state, m);
}
#endif