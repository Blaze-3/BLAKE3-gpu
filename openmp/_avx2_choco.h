#ifndef _AVX2_CHOCO_H
#define _AVX2_CHOCO_H

#include <immintrin.h>
#include <iostream>
using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;

inline __m128i _avx2_rotr(__m128i a, int shift) {
    return _mm_or_si128(
        _mm_slli_epi32(a, 32-shift),
        _mm_srli_epi32(a, shift)
    );
}

void _avx2_g0(__m128i statex[4], const u32 m[16]) {
    __m128i mxx = _mm_set_epi32(m[0], m[2], m[4], m[6]);
    __m128i myx = _mm_set_epi32(m[1], m[3], m[5], m[7]);

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
}

void _avx2_g1(__m128i statex[4], const u32 m[16]) {
    __m128i mxx = _mm_set_epi32(m[8], m[10], m[12], m[14]);
    __m128i myx = _mm_set_epi32(m[9], m[11], m[13], m[15]);

    __m256i perm = _mm256_set_m128i(statex[0], statex[1]);
    __m256i order = _mm256_set_epi32(7, 6, 5, 4, 2, 1, 0, 3);
    perm = _mm256_permutevar8x32_epi32(perm, order);
    statex[0] = _mm256_extracti128_si256(perm, 0b1);
    statex[1] = _mm256_extracti128_si256(perm, 0b0);

    perm = _mm256_set_m128i(statex[2], statex[3]);
    order = _mm256_set_epi32(5, 4, 7, 6, 0, 3, 2, 1);
    perm = _mm256_permutevar8x32_epi32(perm, order);
    statex[2] = _mm256_extracti128_si256(perm, 0b1);
    statex[3] = _mm256_extracti128_si256(perm, 0b0);

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

    perm = _mm256_set_m128i(statex[0], statex[1]);
    order = _mm256_set_epi32(7, 6, 5, 4, 0, 3, 2, 1);
    perm = _mm256_permutevar8x32_epi32(perm, order);
    statex[0] = _mm256_extracti128_si256(perm, 0b1);
    statex[1] = _mm256_extracti128_si256(perm, 0b0);
    
    perm = _mm256_set_m128i(statex[2], statex[3]);
    order = _mm256_set_epi32(5, 4, 7, 6, 2, 1, 0, 3);
    perm = _mm256_permutevar8x32_epi32(perm, order);
    statex[2] = _mm256_extracti128_si256(perm, 0b1);
    statex[3] = _mm256_extracti128_si256(perm, 0b0);
}

inline void _avx2_round(__m128i statex[4], const u32 m[16]) {
    // Mix the columns.
    _avx2_g0(statex, m);
    // Mix the diagonals.
    _avx2_g1(statex, m);
}

extern const u32 IV[8];
void permute(u32 m[16]);

void compress(
    u32 cv[8],
    u32 block_words[16],
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 state[16]
) {
    __m128i statex[4] = {
        _mm_set_epi32(cv[0], cv[1], cv[2], cv[3]),
        _mm_set_epi32(cv[4], cv[5], cv[6], cv[7]),
        _mm_set_epi32(IV[0], IV[1], IV[2], IV[3]),
        _mm_set_epi32((u32)counter, (u32)(counter >> 32), block_len, flags)
    };

    u32 block[16];
    memcpy(block, block_words, 16*sizeof(*block));

    _avx2_round(statex, block); // round 1
    permute(block);
    _avx2_round(statex, block); // round 2
    permute(block);
    _avx2_round(statex, block); // round 3
    permute(block);
    _avx2_round(statex, block); // round 4
    permute(block);
    _avx2_round(statex, block); // round 5
    permute(block);
    _avx2_round(statex, block); // round 6
    permute(block);
    _avx2_round(statex, block); // round 7

    __m128i cvx[2] = {
        _mm_set_epi32(cv[0], cv[1], cv[2], cv[3]),
        _mm_set_epi32(cv[4], cv[5], cv[6], cv[7])
    };
    statex[0] = _mm_xor_si128(statex[0], statex[2]);
    statex[1] = _mm_xor_si128(statex[1], statex[3]);
    statex[2] = _mm_xor_si128(statex[2], cvx[0]);
    statex[3] = _mm_xor_si128(statex[3], cvx[1]);

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
#endif