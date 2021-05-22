#include "blazer3_cpu.cuh"

// Number of threads per thread block
__constant__ const int NUM_THREADS = 16;

// redefine functions, but for the GPU
// all of them are the same but with g_ prefixed
__constant__ const u32 g_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

__constant__ const int g_MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

__device__ u32 g_rotr(u32 value, int shift) {
    return (value >> shift)|(value << (usize - shift));
}

__device__ void g_g(u32 state[16], u32 a, u32 b, u32 c, u32 d, u32 mx, u32 my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = g_rotr((state[d] ^ state[a]), 16);
    state[c] = state[c] + state[d];

    state[b] = g_rotr((state[b] ^ state[c]), 12);
    state[a] = state[a] + state[b] + my;
    state[d] = g_rotr((state[d] ^ state[a]), 8);

    state[c] = state[c] + state[d];
    state[b] = g_rotr((state[b] ^ state[c]), 7);
}

__device__ void g_round(u32 state[16], u32 m[16]) {
    // Mix the columns.
    g_g(state, 0, 4, 8, 12, m[0], m[1]);
    g_g(state, 1, 5, 9, 13, m[2], m[3]);
    g_g(state, 2, 6, 10, 14, m[4], m[5]);
    g_g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    g_g(state, 0, 5, 10, 15, m[8], m[9]);
    g_g(state, 1, 6, 11, 12, m[10], m[11]);
    g_g(state, 2, 7, 8, 13, m[12], m[13]);
    g_g(state, 3, 4, 9, 14, m[14], m[15]);
}

__device__ void g_permute(u32 m[16]) {
    u32 permuted[16];
    for(int i=0; i<16; i++)
        permuted[i] = m[g_MSG_PERMUTATION[i]];
    for(int i=0; i<16; i++)
        m[i] = permuted[i];
}

// custom memcpy, apparently cuda's memcpy is slow 
// when called within a kernel
__device__ void g_memcpy(u32 *lhs, const u32 *rhs, int size) {
    // assuming u32 is 4 bytes
    int len = size / 4;
    for(int i=0; i<len; i++)
        lhs[i] = rhs[i];
}

// custom memset
template<typename T, typename ptr_t>
__device__ void g_memset(ptr_t dest, T val, int count) {
    for(int i=0; i<count; i++)
        dest[i] = val;
}

__device__ void g_compress(
    u32 *chaining_value,
    u32 *block_words,
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 *state
) {
    // Search for better alternative
    g_memcpy(state, chaining_value, 32);
    g_memcpy(state+8, g_IV, 16);
    state[12] = (u32)counter;
    state[13] = (u32)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    u32 block[16];
    g_memcpy(block, block_words, 64);
    
    g_round(state, block); // round 1
    g_permute(block);
    g_round(state, block); // round 2
    g_permute(block);
    g_round(state, block); // round 3
    g_permute(block);
    g_round(state, block); // round 4
    g_permute(block);
    g_round(state, block); // round 5
    g_permute(block);
    g_round(state, block); // round 6
    g_permute(block);
    g_round(state, block); // round 7

    for(int i=0; i<8; i++){
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
}

__device__ void g_words_from_little_endian_bytes(
    u8 *bytes, u32 *words, u32 bytes_len
) {
    u32 tmp;
    for(u32 i=0; i<bytes_len; i+=4) {
        tmp = (bytes[i+3]<<24) | (bytes[i+2]<<16) | (bytes[i+1]<<8) | bytes[i];
        words[i/4] = tmp;
    }
}

__device__ void Chunk::g_compress_chunk(u32 out_flags) {
    if(flags&PARENT) {
        g_compress(
            key,
            data,
            0,  // counter is always zero for parent nodes
            BLOCK_LEN,
            flags | out_flags,
            raw_hash
        );
        return;
    }

    u32 chaining_value[8];
    u32 block_len = BLOCK_LEN, flagger;
    g_memcpy(chaining_value, key, 32);

    bool empty_input = (leaf_len==0);
    if(empty_input) {
        for(u32 i=0; i<BLOCK_LEN; i++)
            leaf_data[i] = 0U;
        leaf_len = BLOCK_LEN;
    }

    // move all mem allocs outside loop
    u32 block_words[16];
    u8 block_cast[BLOCK_LEN];

    for(u32 i=0; i<leaf_len; i+=BLOCK_LEN) {
        flagger = flags;
        // for the last message block
        if(i+BLOCK_LEN > leaf_len)
            block_len = leaf_len%BLOCK_LEN;
        else
            block_len = BLOCK_LEN;

        // special case
        if(empty_input)
            block_len = 0;
        
        // clear up block_words
        g_memset(block_words, 0, 16);

        u32 new_block_len(block_len);
        if(block_len%4)
            new_block_len += 4 - (block_len%4);
        
        // BLOCK_LEN is the max possible length of block_cast
        g_memset(block_cast, 0, new_block_len);
        
        // This memcpy is fine since data is a byte array
        memcpy(block_cast, leaf_data+i, block_len*sizeof(*block_cast));
        
        g_words_from_little_endian_bytes(block_cast, block_words, new_block_len);

        if(i==0)
            flagger |= CHUNK_START;
        if(i+BLOCK_LEN >= leaf_len)
            flagger |= CHUNK_END | out_flags;

        // raw hash for root node
        g_compress(
            chaining_value,
            block_words,
            counter,
            block_len,
            flagger,
            raw_hash
        );

        g_memcpy(chaining_value, raw_hash, 32);
    }
}

__global__ void h_compute(Chunk *gdata, int N, int jump) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    tid *= jump;
    if(tid >= N)
        return;

    // shared memory for compute
    __shared__ Chunk sdata[NUM_THREADS];
    // block local thread ID
    int bl_tid = threadIdx.x;

    sdata[bl_tid] = gdata[tid];
    sdata[bl_tid].g_compress_chunk();
    g_memcpy(gdata[tid].raw_hash, sdata[bl_tid].raw_hash, 64);
}

__global__ void h_move(Chunk *gdata, int N, int jump) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int lhs = (2*tid)*jump, rhs = (2*tid+1)*jump;
    if(rhs >= N)
        return;

    // Parent is LHS itself
    int out = lhs;

    // Key and other flags are already present
    gdata[out].flags |= PARENT;
    g_memcpy(gdata[out].data, gdata[lhs].raw_hash, 32);
    g_memcpy(gdata[out].data+8, gdata[rhs].raw_hash, 32);
}

void dark_hash(Chunk *data, int N, Chunk *result) {    
    const int data_size = N*sizeof(Chunk);
    
    Chunk *gdata_ptr;
    cudaMalloc(&gdata_ptr, data_size);
    cudaMemcpy(gdata_ptr, data, data_size, cudaMemcpyHostToDevice);

    int num_thr, num_blk;
    for(int i=1; i<N; i<<=1) {
        num_thr = min(NUM_THREADS, N/i);
        num_blk = (N/i) / num_thr;
        // compute all hashes
        h_compute<<<num_blk, num_thr>>>(gdata_ptr, N, i);
        
        num_thr /= 2;
        // move hashes to parent nodes
        h_move<<<num_blk, num_thr>>>(gdata_ptr, N, i);
    }

    cudaMemcpy(result, gdata_ptr, sizeof(Chunk), cudaMemcpyDeviceToHost);
    cudaFree(gdata_ptr);
}
