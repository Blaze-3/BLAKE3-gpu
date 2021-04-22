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
const u32 SNICKER = 1U << 5;

const u32 CHUNK_START = 1 << 0;
const u32 CHUNK_END = 1 << 1;
const u32 PARENT = 1 << 2;
const u32 ROOT = 1 << 3;
const u32 KEYED_HASH = 1 << 4;
const u32 DERIVE_KEY_CONTEXT = 1 << 5;
const u32 DERIVE_KEY_MATERIAL = 1 << 6;

u32 IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

int MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

u32 rotr(u32 value, int shift) {
    int usize = sizeof(u32) * 8;
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

u32* first_8_words(u32 *compression_output) {
    u32 *cmprs = new u32[8];
    copy(compression_output, compression_output+8, cmprs);
    delete []compression_output;
    return cmprs;
}

void words_from_little_endian_bytes(vector<u8> &bytes, vector<u32> &words) {
    u32 tmp;
    for(u32 i=0; i<bytes.size(); i+=4) {
        tmp = (bytes[i+3]<<24) | (bytes[i+2]<<16) | (bytes[i+1]<<8) | bytes[i];
        words[i/4] = tmp;
    }
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
    void root_output_bytes(vector<u8> &out_slice) {
        u64 output_block_counter = 0;
        u64 i=0, k=2*OUT_LEN;
        auto osb = begin(out_slice);
        for(; int(out_slice.size()-i)>0; i+=k) {
            // words is u32[16]
            u32* words = compress(
                input_chaining_value,
                block_words,
                output_block_counter,
                block_len,
                flags | ROOT
            );
            vector<u8> out_block(osb+i, osb+i+min(k, (u64)out_slice.size()-i));
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
    void update(vector<u8> &input);
    Output output();
};

ChunkState::ChunkState(u32 key[8], u64 chunk_counter, u32 flags) {
    copy(key, key+8, chaining_value);
    this->chunk_counter = chunk_counter;
    for(int i=0; i<BLOCK_LEN; i++)
        block[i] = 0;
    block_len = 0;
    blocks_compressed = 0;
    this->flags = flags;
}

size_t ChunkState::len() {
    return BLOCK_LEN*blocks_compressed + block_len;
}

u32 ChunkState::start_flag() {
    if(blocks_compressed==0)
        return CHUNK_START;
    return 0;
}

void ChunkState::update(vector<u8> &input) {
    while (!input.empty()) {
        if (u32(block_len) == BLOCK_LEN) {
            vector<u32> block_words(16, 0);
            vector<u8> block_cast(begin(block), end(block));
            words_from_little_endian_bytes(block_cast, block_words);
            
            //chaining_value
            u32* transfer = compress(
                chaining_value,
                block_words.data(),
                chunk_counter,
                u32(BLOCK_LEN),
                flags | start_flag()
            );
            copy(transfer, transfer+8, chaining_value);
            delete []transfer;
            blocks_compressed += 1;
            for (u32 i=0; i < BLOCK_LEN; i++)
                block[i]=0;
            block_len=0;
        }

        u32 want = BLOCK_LEN - u32(block_len);
        u32 take =  min(want, (u32)input.size());
        for(u32 i=block_len; i<block_len+take; i++)
            block[i] = input[i-block_len];
        block_len += take;
        for(u32 i=0; i<input.size()-take; i++)
            input[i] = input[i+take];
        for(u32 i=0; i<take; i++)
            input.pop_back();
    }
}

Output ChunkState::output() {
    vector<u32> block_words(16, 0);
    vector<u8> block_cast(begin(block), end(block));
    words_from_little_endian_bytes(block_cast, block_words); 
    
    Output out;

    for(u32 j=0; j<8; j++)
        out.input_chaining_value[j]=chaining_value[j];
    copy(begin(block_words), end(block_words), out.block_words);
    out.block_len = block_len;
    out.counter = chunk_counter;
    out.flags = flags | start_flag() | CHUNK_END;
    return out;
}

Output parent_output(u32 left_child_cv[8], u32 right_child_cv[8],
u32 key[8], u32 flags) {
    Output out_one;
    for(u32 j=0;j<8;j++){
        out_one.block_words[j] = left_child_cv[j];
        out_one.block_words[j+8] = right_child_cv[j];
        out_one.input_chaining_value[j] = key[j];
    }
    out_one.counter = 0;
    out_one.block_len = BLOCK_LEN;
    out_one.flags = PARENT | flags;
    return out_one;
}

u32* parent_cv(u32 left_child_cv[8], u32 right_child_cv[8],
u32 key[8], u32 flags) {
    return parent_output(left_child_cv, right_child_cv, key, flags).chaining_value();
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
        u32 *transfer = first_8_words(compress(
            key,
            data.data(),
            0,  // counter is always zero for parent nodes
            BLOCK_LEN,
            flags | CHUNK_START | CHUNK_END | out_flags
        ));
        copy(transfer, transfer+8, hash);
        delete[] transfer;

        // raw hash for root node
        transfer = compress(
            key,
            data.data(),
            0,  // counter is always zero for parent nodes
            BLOCK_LEN,
            flags | CHUNK_START | CHUNK_END | out_flags
        );
        copy(transfer, transfer+16, raw_hash);
        delete[] transfer;
    }
    else {
        // cout << "Compressing leaf of size: " << leaf_data.size() << endl;
        u32 chaining_value[8], block_len = BLOCK_LEN, flagger;
        copy(key, key+8, chaining_value);

        for(u32 i=0; i<leaf_data.size(); i+=BLOCK_LEN) {
            flagger = flags;
            // for the last message block
            if(i+BLOCK_LEN > leaf_data.size())
                block_len = leaf_data.size()%BLOCK_LEN;
            else
                block_len = BLOCK_LEN;
            
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

Chunk hash_many(vector<Chunk>::iterator, vector<Chunk>::iterator);
Chunk merge(Chunk left, Chunk right);
void hash_root(Chunk node, vector<u8> &out_slice);

struct Hasher {
    ChunkState chunk_state;
    u32 key[8];
    u32 cv_stack[54][8];
    u32 flags;
    u8 cv_stack_len;
    u64 ctr;
    // Factory is an array of 64 possible rows of chunks
    // Correct size of array is log(2^64) base SNICKER
    vector<Chunk> factory[64];

    // methods
    static Hasher new_internal(u32 key[8], u32 flags);
    static Hasher _new();
    Hasher new_keyed(u8 key[KEY_LEN]);
    Hasher new_derive_key(string context);
    void push_stack(u32 cv[8]);
    u32 *pop_stack();
    void add_chunk_chaining_value(u32 *new_cv, u64 total_chunks);
    void update(vector<u8> &input);
    void finalize(vector<u8> &out_slice);
};

Hasher Hasher::new_internal(u32 key[8], u32 flags) {
    Hasher tmp = Hasher{
        ChunkState(key, 0, flags),
        {}, // key
        {{}},  // cv_stack
        flags,
        0,
        0
    };
    // Not sure about keys and cv_stack, will set them by hand
    copy(key, key+8, tmp.key);
    for(int i=0; i<54; i++)
        for(int j=0; j<8; j++)
            tmp.cv_stack[i][j] = 0;
    return tmp;
}

Hasher Hasher::_new() {
    return new_internal(IV, 0);
}

Hasher Hasher::new_keyed(u8 key[KEY_LEN]) {
    vector<u32> key_words(8, 0);
    vector<u8> key_cast(key, key+KEY_LEN);
    words_from_little_endian_bytes(key_cast, key_words);
    return new_internal(key_words.data(), KEYED_HASH);
}

Hasher Hasher::new_derive_key(string context) {
    Hasher context_hasher = new_internal(IV, DERIVE_KEY_CONTEXT);
    vector<u8> context_bytes(context.length());
    for(u32 i=0; i<context.length(); i++)
        context_bytes[i] = context[i];
    context_hasher.update(context_bytes);
    
    vector<u8> context_key(KEY_LEN);
    context_hasher.finalize(context_key);
    vector<u32> context_key_words(KEY_LEN);
    words_from_little_endian_bytes(context_key, context_key_words);
    return new_internal(context_key_words.data(), DERIVE_KEY_MATERIAL);
}

void Hasher::push_stack(u32 cv[8]) {
    copy(cv, cv+8, cv_stack[cv_stack_len]);
    ++cv_stack_len;
}

u32* Hasher::pop_stack() {
    --cv_stack_len;
    u32 *tmp = new u32[8];
    copy(cv_stack[cv_stack_len], cv_stack[cv_stack_len]+8, tmp);
    return tmp;
}

void Hasher::add_chunk_chaining_value(u32 *new_cv, u64 total_chunks) {
    while ((total_chunks & 1) == 0) {
        u32 *cv = pop_stack();
        u32 *old_cv = new_cv;
        new_cv = parent_cv(cv, old_cv, key, flags);
        delete []cv;
        delete []old_cv;
        total_chunks >>= 1;
    }
    push_stack(new_cv);
    delete []new_cv;
}

void Hasher::update(vector<u8> &input) {
    // while(!input.empty()) {
    //     if(chunk_state.len() == CHUNK_LEN) {
    //         u32* chunk_cv = chunk_state.output().chaining_value();
    //         u64 total_chunks = chunk_state.chunk_counter + 1;
    //         add_chunk_chaining_value(chunk_cv, total_chunks);
    //         chunk_state = ChunkState(key, total_chunks, flags);
    //     }

    //     u32 want = CHUNK_LEN - chunk_state.len();
    //     u32 take = min(want, (u32)input.size());
    //     vector<u8> tmp(begin(input), begin(input)+take);
    //     chunk_state.update(tmp);
    //     for(u32 i=0; i<input.size()-take; i++)
    //         input[i] = input[i+take];
    //     for(u32 i=0; i<take; i++)
    //         input.pop_back();
    // }

    // New style
    int level=0;
    factory[level].push_back(Chunk(input, flags, key, ctr));
    ++ctr;
    while(factory[level].size() == SNICKER) {
        // nodes move to upper levels if lower one is one SNICKER long
        Chunk subtree = hash_many(begin(factory[level]), end(factory[level]));
        factory[level].clear();
        ++level;
        factory[level].push_back(subtree);
    }
    // cout << "Chunks in level: " << factory[level].size() << endl;
}

void Hasher::finalize(vector<u8> &out_slice) {
    // Output output = chunk_state.output();
    // long parent_nodes_remaining = cv_stack_len;
    // while(parent_nodes_remaining > 0) {
    //     --parent_nodes_remaining;
    //     u32 *right_child_cv = output.chaining_value();
    //     output = parent_output(
    //         cv_stack[parent_nodes_remaining],
    //         right_child_cv,
    //         key,
    //         flags
    //     );
    //     delete []right_child_cv;
    // }
    // output.root_output_bytes(out_slice);

    cout << "Finalizing\n";
    // New style
    // At every level, compress biggest to smallest, then merge them all in the reverse order
    // Pass on the new node to the upper level
    // Continue till topmost level reached. Guaranteed only one node there
    // Root hash the final node
    Chunk root;
    for(int i=0; i<64; i++) {
        vector<Chunk> subtrees;
        u32 n = factory[i].size(), divider=SNICKER;
        if(!n)
            continue;
        vector<Chunk>::iterator start = begin(factory[i]);
        while(divider) {
            if(n&divider) {
                // cout << "hashing " << divider << " at level " << i << endl;
                subtrees.push_back(hash_many(start, start+divider));
                start += divider;
            }
            divider >>= 1;
        }
        while(subtrees.size()>1) {
            Chunk tmp1 = subtrees.back();
            subtrees.pop_back();
            Chunk tmp2 = subtrees.back();
            subtrees.pop_back();
            Chunk tmp = merge(tmp1, tmp2);
            subtrees.push_back(tmp);
        }
        if(i<63)
            factory[i+1].push_back(subtrees[0]);
        else
            root = subtrees[0];
    }
    hash_root(root, out_slice);
}

// A divide and conquer approach
Chunk hash_many(vector<Chunk>::iterator first, vector<Chunk>::iterator last) {
    // n will always be a power of 2
    int n = last-first;
    if(n == 1)
        return *first;

    cout << "Called hash many for size: " << n << endl;

    // parallelism here
    // left and right computation can be done simultaneously
    Chunk left = hash_many(first, first+n/2);
    left.compress_chunk();
    Chunk right = hash_many(first+n/2, last);
    right.compress_chunk();
    // parallelism ends

    Chunk parent(left.flags, left.key);
    parent.flags |= PARENT;
    parent.data.insert(end(parent.data), begin(left.hash), end(left.hash));
    parent.data.insert(end(parent.data), begin(right.hash), end(right.hash));
    return parent;
}

Chunk merge(Chunk left, Chunk right) {
    cout << "Called merge once\n";
    left.compress_chunk();
    right.compress_chunk();

    Chunk parent(left.flags, left.key);
    parent.flags |= PARENT;
    parent.data.insert(end(parent.data), begin(left.hash), end(left.hash));
    parent.data.insert(end(parent.data), begin(right.hash), end(right.hash));
    return parent;
}

void hash_root(Chunk node, vector<u8> &out_slice) {
    // the last message block must not be hashed
    u64 output_block_counter = 0;
    u64 i=0, k=2*OUT_LEN;
    auto osb = begin(out_slice);
    for(; int(out_slice.size()-i)>0; i+=k) {
        node.counter = output_block_counter;
        node.compress_chunk(ROOT);
        // words is u32[16]
        u32* words = new u32[16];
        copy(node.raw_hash, node.raw_hash+16, words);
        vector<u8> out_block(osb+i, osb+i+min(k, (u64)out_slice.size()-i));
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