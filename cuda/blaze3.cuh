#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
using namespace std;

using u32 = unsigned int;
using u64 = unsigned long long;
using u8  = unsigned char;
 
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
const u32 DERIVE_KEY_CONTEXT = 1 << 5;
const u32 DERIVE_KEY_MATERIAL = 1 << 6;

const int usize = sizeof(u32) * 8;

u32 IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

int MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

__constant__ const u32 g_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

__constant__ const int g_MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

u32 rotr(u32 value, int shift) {
    return (value >> shift)|(value << (usize - shift));
}

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

void round(u32 state[16], u32 m[16]) {
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

__device__ void g_permute(u32 m[16]) {
    u32 permuted[16];
    for(int i=0; i<16; i++)
        permuted[i] = m[g_MSG_PERMUTATION[i]];
    for(int i=0; i<16; i++)
        m[i] = permuted[i];
}

__device__ void g_memcpy(u32 *lhs, const u32 *rhs, int size) {
    // assuming u32 is 4 bytes
    int len = size / 4;
    for(int i=0; i<len; i++)
        lhs[i] = rhs[i];
}

template<typename T, typename ptr_t>
__device__ void g_memset(ptr_t dest, T val, int count) {
    for(int i=0; i<count; i++)
        dest[i] = val;
}

void permute(u32 m[16]) {
    u32 permuted[16];
    for(int i=0; i<16; i++)
        permuted[i] = m[MSG_PERMUTATION[i]];
    for(int i=0; i<16; i++)
        m[i] = permuted[i];
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

void compress(
    u32 *chaining_value,
    u32 *block_words,
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 *state
) {
    
    copy(chaining_value, chaining_value+8, state);
    copy(IV, IV+4, state+8);
    state[12] = (u32)counter;
    state[13] = (u32)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    u32 block[16];
    copy(block_words, block_words+16, block);
    //printf("%s","cuda can fuckoff");
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
}

void words_from_little_endian_bytes(u8 *bytes, u32 *words, u32 bytes_len) {
    u32 tmp;
    for(u32 i=0; i<bytes_len; i+=4) {
        tmp = (bytes[i+3]<<24) | (bytes[i+2]<<16) | (bytes[i+1]<<8) | bytes[i];
        words[i/4] = tmp;
    }
}

struct Chunk {
    // use only when it is a leaf node
    // leaf data may have less than 1024 bytes

    u8 leaf_data[1024];
    u32 leaf_len;
    // use in all other cases
    // data will always have 64 bytes
    u32 data[16];
    u32 flags;
    u32 raw_hash[16];
    u32 key[8];
    // only useful for leaf nodes
    u64 counter;
    // Constructor for leaf nodes
    Chunk(vector<u8> &input, u32 _flags, u32 *_key, u64 ctr){
        counter = ctr;
        flags = _flags;
        copy(_key, _key+8, key);
        copy(begin(input), end(input), begin(leaf_data));
        leaf_len = input.size();
    }
    Chunk(u32 _flags, u32 *_key) {
        counter = 0;
        flags = _flags;
        copy(_key, _key+8, key);
        leaf_len = 0;
    }
    Chunk() : leaf_len(0) {}
    // process data in sizes of message blocks and store cv in hash
    void compress_chunk(u32=0);
    __device__ void g_compress_chunk(u32=0);
};

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
        
        // This memcpy is fine since data is a byte array
        memcpy(block_cast, leaf_data+i, new_block_len*sizeof(*block_cast));
        
        g_words_from_little_endian_bytes(leaf_data+i, block_words, new_block_len);

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

void Chunk::compress_chunk(u32 out_flags) {
    // cout << "Compress called\n";
    if(flags&PARENT) {
        compress(
            key,
            data,
            0,  // counter is always zero for parent nodes
            BLOCK_LEN,
            flags | out_flags,
            raw_hash
        );
    }

    // for leaf nodes
    else {
        // cout << "Compressing leaf of size: " << leaf_data.size() << endl;
        u32 chaining_value[8], block_len = BLOCK_LEN, flagger;
        copy(key, key+8, chaining_value);

        bool empty_input = (leaf_len==0);
        if(empty_input) {
            // cout << "empty yo\n";
            for(u32 i=0; i<BLOCK_LEN; i++)
                leaf_data[i] = 0U;
            leaf_len = BLOCK_LEN;
        }

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
            
            // TODO: Convert to CudaMalloc
            u32 *block_words = new u32[16];
            // TODO: Convert to cudaMemset
            memset(block_words, 0, 16*sizeof(*block_words));
            u32 new_block_len(block_len);
            if(block_len%4)
                new_block_len += 4 - (block_len%4);
            
            // TODO: Convert to CudaMalloc
            u8 *block_cast = new u8[new_block_len];
            // TODO: Convert to cudaMemset
            memset(block_cast, 0, new_block_len*sizeof(*block_cast));
            // TODO: convert to cudaMemcpy
            memcpy(block_cast, leaf_data+i, block_len*sizeof(*block_cast));
            
            words_from_little_endian_bytes(block_cast, block_words, new_block_len);

            // TODO: Convert to CudaFree
            delete[] block_cast;
            if(i==0)
                flagger |= CHUNK_START;
            if(i+BLOCK_LEN >= leaf_len)
                flagger |= CHUNK_END | out_flags;

            // raw hash for root node
            compress(
                chaining_value,
                block_words,
                counter,
                block_len,
                flagger,
                raw_hash
            );

            // TODO: Convert to CudaFree
            delete[] block_words;
            copy(raw_hash, raw_hash+8, chaining_value);
        }
    }
    // cout << "Compress worked\n";
}

void hash_many(Chunk *data, int first, int last, Chunk *parent);
Chunk merge(Chunk &left, Chunk &right);
void hash_root(Chunk &node, vector<u8> &out_slice);

struct Hasher {
    u32 key[8];
    u32 flags;
    u64 ctr;
    // Factory is an array of FACTORY_HT possible SNICKER bars
    vector<Chunk> factory[FACTORY_HT];
    vector<Chunk> bar;

    // methods
    static Hasher new_internal(u32 key[8], u32 flags);
    static Hasher _new();

    void update(vector<u8> &input);
    void finalize(vector<u8> &out_slice);
    void propagate();
};

Hasher Hasher::new_internal(u32 key[8], u32 flags) {
    return Hasher{
        {
            key[0], key[1], key[2], key[3],
            key[4], key[5], key[6], key[7]
        },
        flags,
        0   // counter
    };
}

Hasher Hasher::_new() {
    return new_internal(IV, 0);
}

void Hasher::propagate() {
    int level=0;
    // nodes move to upper levels if lower one is one SNICKER long
    while(factory[level].size() == SNICKER) {
        // TODO: Convert to CudaMalloc
        Chunk *subtree = new Chunk;

        // Move factory[level].data() to Cuda memory
        hash_many(factory[level].data(), 0, factory[level].size(), subtree);
        cudaDeviceSynchronize();
        factory[level].clear();
        ++level;
        factory[level].push_back(*subtree);

        // TODO: Convert to CudaFree
        delete subtree;
    }
} 

void Hasher::update(vector<u8> &input) {
    bar.push_back(Chunk(input, flags, key, ctr));
    ++ctr;
    if(bar.size() == SNICKER) {
        copy(begin(bar), end(bar), back_inserter(factory[0]));
        bar.clear();
        propagate();
    }
}

void Hasher::finalize(vector<u8> &out_slice) {
    // cout << "Finalizing\n";
    // New style
    // At every level, compress biggest to smallest, then merge them all in the reverse order
    // Pass on the new node to the upper level
    // Continue till topmost level reached. Guaranteed only one node there
    // Root hash the final node
    copy(begin(bar), end(bar), back_inserter(factory[0]));

    Chunk root(flags, key);
    for(int i=0; i<FACTORY_HT; i++) {
        vector<Chunk> subtrees;
        u32 n = factory[i].size(), divider=SNICKER;
        if(!n)
            continue;
        int start = 0;
        while(divider) {
            if(n&divider) {
                // TODO: Convert to CudaMalloc
                Chunk *subtree = new Chunk;
                // Move factory[i].data() to Cuda memory
                //cout<<"iter"<<i<<endl;
                hash_many(factory[i].data(), start, start+divider, subtree);
                cudaDeviceSynchronize();
                subtrees.push_back(*subtree);
                // TODO: Convert to CudaFree
                delete subtree;
                start += divider;
            }
            divider >>= 1;
        }
        while(subtrees.size()>1) {
            Chunk tmp1 = subtrees.back();
            subtrees.pop_back();
            Chunk tmp2 = subtrees.back();
            subtrees.pop_back();
            // tmp2 is the left child
            // tmp1 is the right child
            // that's the order they appear within the array
            Chunk tmp = merge(tmp2, tmp1);
            subtrees.push_back(tmp);
        }
        if(i<FACTORY_HT-1)
            factory[i+1].push_back(subtrees[0]);
        else
            root = subtrees[0];
    }
    hash_root(root, out_slice);
}

__global__ void d_hashmany(Chunk *d_data, int first, int last, Chunk* d_parent){
    //  int idx = blockDim.x*blockIdx.x + threadIdx.x;
    // if(idx > 0)
    //      return;

    int n = last-first;
    if(n==1){
        d_data[first].g_compress_chunk();
        memcpy(d_parent, d_data+first, sizeof(*d_parent));
        return;
    }
    // printf("Calling for first : %d and last : %d\n", first, last);
    
    Chunk *left;
    cudaMalloc(&left,sizeof(Chunk));
    Chunk *right;
    cudaMalloc(&right,sizeof(Chunk));

    cudaStream_t s1,s2;

    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    d_hashmany<<<32,128,0,s1>>>(d_data, first, first+n/2, left);
    cudaDeviceSynchronize();

    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    d_hashmany<<<32,128,0,s2>>>(d_data, first+n/2, last, right);
    cudaDeviceSynchronize();

    d_parent->flags = left->flags | PARENT;
    memcpy(d_parent->key, left->key, 32);
    memcpy(d_parent->data, left->raw_hash, 32);
    memcpy(d_parent->data+8, right->raw_hash, 32);
    d_parent->g_compress_chunk();

    cudaFree(left);
    cudaFree(right);
}


// A divide and conquer approach
void hash_many(Chunk *data, int first, int last, Chunk *parent) {
    Chunk *d_data;
    Chunk *d_parent;
    
    //allocate on device
    cudaMalloc(&d_data,sizeof(Chunk)*(last-first));
    cudaMalloc(&d_parent,sizeof(Chunk));

    //populate on device
    cudaMemcpy(d_data,data,sizeof(Chunk)*(last-first),cudaMemcpyHostToDevice);

    d_hashmany<<<1,1>>>(d_data,first,last,d_parent);
    cudaMemcpy(parent,d_parent,sizeof(Chunk),cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_parent);
}

Chunk merge(Chunk &left, Chunk &right) {
    // cout << "Called merge once\n";
    left.compress_chunk();
    right.compress_chunk();

    Chunk parent(left.flags, left.key);
    // 32 bytes need to be copied for all of these
    memcpy(parent.data, left.raw_hash, 32);
    memcpy(parent.data+8, right.raw_hash, 32);
    return parent;
}

void hash_root(Chunk &node, vector<u8> &out_slice) {
    // the last message block must not be hashed like the others
    // it needs to be hashed with the root flag
    u64 output_block_counter = 0;
    u64 i=0, k=2*OUT_LEN;
    
    u32 words[16] = {};
    for(; int(out_slice.size()-i)>0; i+=k) {
        node.counter = output_block_counter;
        node.compress_chunk(ROOT);
        
        // words is u32[16]
        copy(node.raw_hash, node.raw_hash+16, words);
        
        vector<u8> out_block(min(k, (u64)out_slice.size()-i));
        for(u32 l=0; l<out_block.size(); l+=4) {
            for(u32 j=0; j<min(4U, (u32)out_block.size()-l); j++)
                out_block[l+j] = (words[l/4]>>(8*j)) & 0x000000FF;
        }
        
        for(u32 j=0; j<out_block.size(); j++)
            out_slice[i+j] = out_block[j];
        
        ++output_block_counter;
    }
}