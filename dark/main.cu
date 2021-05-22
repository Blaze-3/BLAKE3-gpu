#include "blaze3.cuh"
#include <fstream>
#include <iomanip>
#include <filesystem>

// Do not change this, it has to be equal to chunk size
#define BUFFER_LEN CHUNK_LEN

int main(int argc, char *argv[]) {
    if(argc<2) {
        cout << "Usage: " << argv[0] << " test-file-name\n";
        return 1;
    }

    ifstream file(argv[1], ios::binary);
    if(file.fail()) {
        cout << "Could not read file " << argv[1] << endl;
        return 1;
    }

    filesystem::path fyle{argv[1]};
    u64 file_size = filesystem::file_size(fyle);

    Hasher hasher = Hasher::_new(file_size);
    // this must always be there
    hasher.init();

    char buffer[BUFFER_LEN] = {0};
    file.read(buffer, BUFFER_LEN);
    while(file.gcount()) {
        hasher.update(buffer, file.gcount());
        file.read(buffer, BUFFER_LEN);
    }

    vector<u8> hash_output(32);
    hasher.finalize(hash_output);

    for(auto e: hash_output)
        cout << hex << setfill('0') << setw(2) << (int)e;
    cout << endl;

    return 0;
}