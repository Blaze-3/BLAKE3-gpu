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
const u32 SNICKER = 1U << 6;
// Factory height and snicker size have an inversly propotional relationship
// FACTORY_HT * (log2 SNICKER) + 10 >= 64 
const u32 FACTORY_HT = 9;

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

const int MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

u32 rotr(u32 value, int shift) {
    return (value >> shift)|(value << (usize - shift));
}

void g (u32 state[16], u32 a, u32 b, u32 c, u32 d, u32 mx, u32 my) {
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
    for(int i=0; i<4; i++)
        g(state, i, i+4, i+8, i+12, m[2*i], m[2*i+1]);
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
    u32 *state = new u32[16] {
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
        (u32)counter,
        (u32)(counter >> 32),
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

void words_from_little_endian_bytes(vector<u8> &bytes, vector<u32> &words) {
    u32 tmp;
    for(u32 i=0; i<bytes.size(); i+=4) {
        tmp = (bytes[i+3]<<24) | (bytes[i+2]<<16) | (bytes[i+1]<<8) | bytes[i];
        words[i/4] = tmp;
    }
}

struct Chunk {
    // use only when it is a leaf node
    vector<u8> leaf_data;
    // use in all other cases
    vector<u32> data;
    u32 flags;
    u32 hash[8], raw_hash[16];
    u32 key[8];
    // only useful for leaf nodes
    u64 counter;
    // Constructor for leaf nodes
    Chunk(vector<u8> &input, u32 _flags, u32 *_key, u64 ctr) : leaf_data(input) {
        counter = ctr;
        flags = _flags;
        copy(_key, _key+8, key);
    }
    Chunk(u32 _flags, u32 *_key) {
        counter = 0;
        flags = _flags;
        copy(_key, _key+8, key);
    }
    Chunk() {}
    // process data in sizes of message blocks and store cv in hash
    void compress_chunk(u32=0);
};

void Chunk::compress_chunk(u32 out_flags) {
    // cout << "Compress called\n";
    if(flags&PARENT) {
        // cout << "Compressing parent\n";
        // only 1 message block
        // raw hash for root node
        u32 *transfer = compress(
            key,
            data.data(),
            0,  // counter is always zero for parent nodes
            BLOCK_LEN,
            flags | out_flags
        );
        copy(transfer, transfer+8, hash);
        copy(transfer, transfer+16, raw_hash);
        delete[] transfer;
    }
    else {
        // cout << "Compressing leaf of size: " << leaf_data.size() << endl;
        u32 chaining_value[8], block_len = BLOCK_LEN, flagger;
        copy(key, key+8, chaining_value);

        bool empty_input = leaf_data.empty();
        if(empty_input) {
            // cout << "empty yo\n";
            for(u32 i=0; i<BLOCK_LEN; i++)
                leaf_data.push_back(0U);
        }

        for(u32 i=0; i<leaf_data.size(); i+=BLOCK_LEN) {
            flagger = flags;
            // for the last message block
            if(i+BLOCK_LEN > leaf_data.size())
                block_len = leaf_data.size()%BLOCK_LEN;
            else
                block_len = BLOCK_LEN;

            // special case
            if(empty_input)
                block_len = 0;
            
            vector<u32> block_words(16, 0);
            vector<u8> block_cast(leaf_data.begin()+i, leaf_data.begin()+i+block_len);
            
            // to pad the message block
            if(block_cast.size()%4) {
                for(int j=4 - (block_cast.size()%4); j>0; j--)
                    block_cast.push_back(0U);
            }
            
            words_from_little_endian_bytes(block_cast, block_words);
            
            if(i==0)
                flagger |= CHUNK_START;
            if(i+BLOCK_LEN >= leaf_data.size())
                flagger |= CHUNK_END | out_flags;

            // raw hash for root node
            u32 *full_transfer = compress(
                chaining_value,
                block_words.data(),
                counter,
                block_len,
                flagger
            );
            copy(full_transfer, full_transfer+16, raw_hash);
            copy(full_transfer, full_transfer+8, chaining_value);
            delete[] full_transfer;
        }
        copy(chaining_value, chaining_value+8, hash);
    }
    // cout << "Compress worked\n";
}

Chunk hash_many(vector<Chunk> &data, int first, int last);
Chunk merge(Chunk &left, Chunk &right);
void hash_root(Chunk &node, vector<u8> &out_slice);

struct Hasher {
    u32 key[8];
    u32 flags;
    u64 ctr;
    // Factory is an array of FACTORY_HT possible SNICKER bars
    vector<Chunk> factory[FACTORY_HT];

    // methods
    static Hasher new_internal(u32 key[8], u32 flags);
    static Hasher _new();

    void update(vector<u8> &input);
    void finalize(vector<u8> &out_slice);
};

Hasher Hasher::new_internal(u32 key[8], u32 flags) {
    return Hasher{
        {key[0], key[1], key[2], key[3],
         key[4], key[5], key[6], key[7]
        },
        flags,
        0   // counter
    };
}

Hasher Hasher::_new() {
    return new_internal(IV, 0);
}

void Hasher::update(vector<u8> &input) {
    // New style
    int level=0;
    factory[level].push_back(Chunk(input, flags, key, ctr));
    ++ctr;
    while(factory[level].size() == SNICKER) {
        // nodes move to upper levels if lower one is one SNICKER long
        Chunk subtree = hash_many(factory[level], 0, factory[level].size());
        factory[level].clear();
        ++level;
        factory[level].push_back(subtree);
    }
    // cout << "Chunks in level: " << factory[level].size() << endl;
}

void Hasher::finalize(vector<u8> &out_slice) {
    // cout << "Finalizing\n";
    // New style
    // At every level, compress biggest to smallest, then merge them all in the reverse order
    // Pass on the new node to the upper level
    // Continue till topmost level reached. Guaranteed only one node there
    // Root hash the final node
    Chunk root(flags, key);
    for(int i=0; i<FACTORY_HT; i++) {
        vector<Chunk> subtrees;
        u32 n = factory[i].size(), divider=SNICKER;
        if(!n)
            continue;
        int start = 0;
        while(divider) {
            if(n&divider) {
                // cout << "hashing " << divider << " at level " << i << endl;
                subtrees.push_back(hash_many(factory[i], start, start+divider));
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
    parent.data.insert(end(parent.data), begin(left.hash), end(left.hash));
    parent.data.insert(end(parent.data), begin(right.hash), end(right.hash));

    parent.compress_chunk();
    return parent;
}

Chunk merge(Chunk &left, Chunk &right) {
    // cout << "Called merge once\n";
    left.compress_chunk();
    right.compress_chunk();

    Chunk parent(left.flags, left.key);
    parent.flags |= PARENT;
    parent.data.insert(end(parent.data), begin(left.hash), end(left.hash));
    parent.data.insert(end(parent.data), begin(right.hash), end(right.hash));
    return parent;
}

void hash_root(Chunk &node, vector<u8> &out_slice) {
    // the last message block must not be hashed like the others
    // it needs to be hashed with the root flag
    u64 output_block_counter = 0;
    u64 i=0, k=2*OUT_LEN;
    auto osb = begin(out_slice);
    for(; int(out_slice.size()-i)>0; i+=k) {
        node.counter = output_block_counter;
        node.compress_chunk(ROOT);
        
        // words is u32[16]
        u32* words = new u32[16];
        copy(node.raw_hash, node.raw_hash+16, words);
        
        vector<u8> out_block(min(k, (u64)out_slice.size()-i));
        for(u32 l=0; l<out_block.size(); l+=4) {
            for(u32 j=0; j<min(4U, (u32)out_block.size()-l); j++)
                out_block[l+j] = (words[l/4]>>(8*j)) & 0x000000FF;
        }
        
        for(u32 j=0; j<out_block.size(); j++)
            out_slice[i+j] = out_block[j];
        
        ++output_block_counter;
        delete []words;
    }
}