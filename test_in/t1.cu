#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
using namespace std;
#include "blaze3.cuh"
#include <fstream>
#include <iomanip>

Chunk hash_many_cpu(Chunk *data, int first, int last) {
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
    
    left = hash_many_cpu(data, first, first+n/2);
    right = hash_many_cpu(data, first+n/2, last);
     

    Chunk parent(left.flags, left.key);
    parent.flags |= PARENT;
    //parent.data.insert(end(parent.data), left.raw_hash, left.raw_hash+8);
    //parent.data.insert(end(parent.data), right.raw_hash, right.raw_hash+8);
    memcpy(parent.data,left.raw_hash,32);
    memcpy(parent.data+8,right.raw_hash,32);
    parent.compress_chunk();
    return parent;
}

__managed__ u32 temp;

int main(){
    //checking rotr
    u32 val =22;
    u32 shift=12;
    d_rotr<<<1,1>>>(val,shift,&temp);
    cudaDeviceSynchronize();
    
    if(rotr(val,shift)==temp){
        cout<<"d_rotr passed!"<<endl;
    }

    //checking g
    u32* state_gpu;
    u32* state_cpu=(u32*) malloc(sizeof(32)*16);
    u32 state_check[16];
    for(int i=0;i<16;i++){
        state_cpu[i]=rand() % 100 + 1; 
        state_check[i]=state_cpu[i];  
    }

    cudaMalloc(&state_gpu,sizeof(u32)*16);
    cudaMemcpy(state_gpu,state_cpu,sizeof(32)*16,cudaMemcpyHostToDevice);
    //u32* state_cpu_check=(u32*) malloc(sizeof(32)*16);
    d_g<<<1,1>>>(state_gpu,3,5,9,4,73,23);

    cudaMemcpy(state_cpu,state_gpu,sizeof(32)*16,cudaMemcpyDeviceToHost);
    
    int check=1;
    g(state_check,3,5,9,4,73,23);
    for(int i=0;i<16;i++){
        if(state_check[i]!=state_cpu[i]){
            check=0;
        }
    }
    if(check){
        cout<<"g passed!"<<endl;
    }
    cudaFree(state_gpu);

    //testing compress
    Chunk* chunk_cpu;
    Chunk* chunk_gpu;
    chunk_cpu=(Chunk*)malloc(sizeof(Chunk));
    
    cudaMalloc(&chunk_gpu,sizeof(Chunk));
    char c;
    int r;
    srand (time(NULL));   
    for (int i=0; i<1024; i++)
    {    r = rand() % 10;   
         c =  '0'+r; 
         chunk_cpu->leaf_data[i]=c;
         
    }
    chunk_cpu->leaf_len=1024;
    for(int i=0; i<16; i++){
        chunk_cpu->data[i]=rand() % 10;
        chunk_cpu->raw_hash[i]=rand() % 10;
        chunk_cpu->key[i%8]=rand() % 10;
        
    }
    chunk_cpu->flags=1;
    Chunk chunk_check=*chunk_cpu;
    
    cudaMemcpy(chunk_gpu,chunk_cpu,sizeof(Chunk),cudaMemcpyHostToDevice);
    d_compress<<<1,1>>>(0,chunk_gpu);
    cudaDeviceSynchronize();
    
    cudaMemcpy(chunk_cpu,chunk_gpu,sizeof(Chunk),cudaMemcpyDeviceToHost);
    int cop=1;
    chunk_check.compress_chunk();
    for(int i=0; i<16;i++){
        if(chunk_cpu->raw_hash[i]!=chunk_check.raw_hash[i]){
            cop=0;
        }
    }
    if(cop){
        cout<<"compress passed!"<<endl;
    }
    
    //testing hashmany
    int size_array=1;
    Chunk* chunk_cpu_arr;
    Chunk* chunk_gpu_arr;
    chunk_cpu_arr=(Chunk*)malloc(sizeof(Chunk)*size_array);
    cudaMalloc(&chunk_gpu_arr,sizeof(Chunk)*size_array);
    Chunk checker_many;
    Chunk* res_gpu;
    cudaMalloc(&res_gpu,sizeof(Chunk));
    Chunk* res_cpu=(Chunk*)malloc(sizeof(Chunk));
    Chunk* chunk_gpu_arr2;
    cudaMalloc(&chunk_gpu_arr2,sizeof(Chunk)*size_array);
    Chunk* res_cpu2=(Chunk*)malloc(sizeof(Chunk));
      
    srand(42);
    for(int k=0;k<size_array;k++){
        for (int i=0; i<1024; i++)
        {    r = rand() % 10;   
            c =  '0'+r; 
            chunk_cpu_arr[k].leaf_data[i]=c;    
        }
        chunk_cpu_arr[k].leaf_len=1024;
        for(int i=0; i<16; i++){
            chunk_cpu_arr[k].data[i]=rand() % 10;
            chunk_cpu_arr[k].raw_hash[i]=rand() % 10;
            chunk_cpu_arr[k].key[i%8]=rand() % 10;   
        }
        chunk_cpu_arr[k].flags=1;
    }

    cudaMemcpy(chunk_gpu_arr,chunk_cpu_arr,size_array*sizeof(Chunk),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // printf("ptr: 0x%p\n", chunk_gpu_arr);
    hash_many(chunk_gpu_arr,0,size_array,res_gpu);
    cudaDeviceSynchronize();
    // cudaMemcpy(chunk_gpu_arr2,chunk_gpu_arr,sizeof(Chunk),cudaMemcpyDeviceToDevice);
    // d_compress<<<1,1>>>(0,chunk_gpu_arr2);
    cudaDeviceSynchronize();
    cudaMemcpy(res_cpu,res_gpu,sizeof(Chunk),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(res_cpu2,chunk_gpu_arr2,sizeof(Chunk),cudaMemcpyDeviceToHost);

    checker_many=hash_many_cpu(chunk_cpu_arr,0,size_array);
    
    int bob=1;
    for(int i=0;i<16;i++){
        if(checker_many.raw_hash[i]!=res_cpu->raw_hash[i]){
            bob=0;
            // cout << "Does not match at : " << i << endl;
        }
        // cout<< checker_many.raw_hash[i]<<":"<<res_cpu->raw_hash[i]<<":"<<res_cpu2->raw_hash[i]<<endl;
    }
    if(bob){
        cout<<"hashmay passed!"<<endl;
    }

    printf("GPU hash:\n");
    for(int i=0; i<8; i++)
        printf("%02x", res_cpu->raw_hash[i]);
    printf("\n");

    printf("Real hash:\n");
    for(int i=0; i<8; i++)
        printf("%02x", checker_many.raw_hash[i]);
    printf("\n");

}