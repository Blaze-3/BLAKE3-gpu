#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
#include <future>
#include <mutex>
using namespace std;

#if defined(_OPENMP)
// 4 lane (i.e. kit-kat) style speed up only when OMP used
// for no reason other than removing all parallelism from serial version
#include "_avx2_choco.h"
#endif

#ifndef _AVX2_CHOCO_H
// these types are already defined there
using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;
#endif
 
const u32 OUT_LEN = 32;
const u32 KEY_LEN = 32;
const u32 BLOCK_LEN = 64;
const u32 CHUNK_LEN = 1024;
// Multiple chunks make a snicker bar :)
const u32 SNICKER = 1U << 9;
// Factory height and snicker size have an inversly propotional relationship
// FACTORY_HT * (log2 SNICKER) + 10 >= 64 
const u32 FACTORY_HT = 6;

const u32 CHUNK_START = 1 << 0;
const u32 CHUNK_END = 1 << 1;
const u32 PARENT = 1 << 2;
const u32 ROOT = 1 << 3;
const u32 KEYED_HASH = 1 << 4;

const int usize = sizeof(u32) * 8;
mutex factory_lock;
const int IS_ASYNC = 0;

const u32 IV[8] = {
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

void permute(u32 m[16]) {
    u32 permuted[16];
    for(int i=0; i<16; i++)
        permuted[i] = m[MSG_PERMUTATION[i]];
    for(int i=0; i<16; i++)
        m[i] = permuted[i];
}

#ifndef _AVX2_CHOCO_H
// AVX2 has a specialized version for this
void compress(
    u32 chaining_value[8],
    u32 block_words[16],
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 state[16]
) {
    memcpy(state, chaining_value, 8*sizeof(*state));
    memcpy(state+8, IV, 4*sizeof(*state));
    state[12] = (u32)counter;
    state[13] = (u32)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    u32 block[16];
    memcpy(block, block_words, 16*sizeof(*block));
    
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
#endif

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
    Chunk(char *input, int size, u32 _flags, u32 *_key, u64 ctr){
        counter = ctr;
        flags = _flags;
        memcpy(key, _key, 8*sizeof(*key));
        memset(leaf_data, 0, 1024);
        memcpy(leaf_data, input, size);
        leaf_len = size;
    }
    Chunk(u32 _flags, u32 *_key) {
        counter = 0;
        flags = _flags;
        memcpy(key, _key, 8*sizeof(*key));
        leaf_len = 0;
    }
    Chunk() : leaf_len(0) {}
    // process data in sizes of message blocks and store cv in hash
    void compress_chunk(u32=0);
};

void Chunk::compress_chunk(u32 out_flags) {
    if(flags&PARENT) {
        compress(
            key,
            data,
            0,  // counter is always zero for parent nodes
            BLOCK_LEN,
            flags | out_flags,
            raw_hash
        );
        return;
    }

    u32 chaining_value[8], block_len = BLOCK_LEN, flagger;
    memcpy(chaining_value, key, 8*sizeof(*chaining_value));

    bool empty_input = (leaf_len==0);
    if(empty_input) {
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
        
        u32 block_words[16];
        memset(block_words, 0, 16*sizeof(*block_words));
        u32 new_block_len(block_len);
        if(block_len%4)
            new_block_len += 4 - (block_len%4);

        words_from_little_endian_bytes(leaf_data+i, block_words, new_block_len);

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

        memcpy(chaining_value, raw_hash, 8*sizeof(*chaining_value));
    }
}

Chunk hash_many(vector<Chunk> &data, int first, int last);
Chunk merge(Chunk &left, Chunk &right);
void hash_root(Chunk &node, vector<u8> &out_slice);

struct Hasher {
    u32 key[8];
    u32 flags;
    u64 ctr;
    // Factory of FACTORY_HT possible SNICKER bars
    vector<Chunk> factory[FACTORY_HT];

    // methods
    static Hasher new_internal(const u32 key[8], u32 flags);
    static Hasher _new();

    void update(char *input, int size);
    void finalize(vector<u8> &out_slice);
};

Hasher Hasher::new_internal(const u32 key[8], u32 flags) {
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

void propagate(Hasher *h) {
    int level=0;
    while(h->factory[level].size() == SNICKER) {
        // nodes move to upper levels if lower one is one SNICKER long
        Chunk subtree = hash_many(
            h->factory[level], 0, h->factory[level].size()
        );
        h->factory[level].clear();
        ++level;
        h->factory[level].emplace_back(subtree);
    }
    #if IS_ASYNC
    factory_lock.unlock();
    #endif
} 

void Hasher::update(char *input, int size) {
    factory[0].emplace_back(input, size, flags, key, ctr);
    ++ctr;
    if(factory[0].size() == SNICKER) {
        // Let this run in the background
        // Async version slows down execution by 2x
        #if IS_ASYNC
        factory_lock.lock();
        static_cast<void>(async(propagate, this));
        #else
        propagate(this);
        #endif
    }
}

void Hasher::finalize(vector<u8> &out_slice) {
    // cout << "Finalizing\n";
    // New style
    // At every level, compress biggest to smallest, then merge them all in the reverse order
    // Pass on the new node to the upper level
    // Continue till topmost level reached. Guaranteed only one node there
    // Root hash the final node
    #if IS_ASYNC
    factory_lock.lock();
    #endif

    vector<Chunk> subtrees;
    Chunk root(flags, key);
    for(u32 i=0; i<FACTORY_HT; i++) {
        u32 n = factory[i].size(), divider=SNICKER;
        if(!n)
            continue;
        int start = 0;
        while(divider) {
            if(n&divider) {
                subtrees.emplace_back(
                    hash_many(factory[i], start, start+divider)
                );
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
        subtrees.clear();
    }
    hash_root(root, out_slice);
}

// A divide and conquer approach
Chunk hash_many(vector<Chunk> &data, int first, int last) {
    // n will always be a power of 2
    int n = last-first;
    if(n == 1) {
        data[first].compress_chunk();
        return data[first];
    }
    // cout << "Called hash many for size: " << n << endl;

    Chunk left, right;
    // parallelism here
    // left and right computation can be done simultaneously
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            left = hash_many(data, first, first+n/2);
            #pragma omp task
            right = hash_many(data, first+n/2, last);
        }
    }
    // parallelism ends

    Chunk parent(left.flags, left.key);
    parent.flags |= PARENT;
    memcpy(parent.data, left.raw_hash, 32);
    memcpy(parent.data+8, right.raw_hash, 32);

    parent.compress_chunk();
    return parent;
}

Chunk merge(Chunk &left, Chunk &right) {
    // cout << "Called merge once\n";
    left.compress_chunk();
    right.compress_chunk();

    Chunk parent(left.flags, left.key);
    parent.flags |= PARENT;
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
        memcpy(words, node.raw_hash, 16*sizeof(*words));
        
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