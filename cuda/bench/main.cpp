#include "sequential.h"
#include <fstream>
#include <iomanip>

#define BUFFER_LEN 1024

int main(int argc, char *argv[]) {
    if(argc<2) {
        cout << "Usage: " << argv[0] << " test-file-name\n";
        return 1;
    }

    Hasher hasher = Hasher::_new();

    ifstream file(argv[1], ios::binary);
    if(file.fail()) {
        cout << "Could not read file " << argv[1] << endl;
        return 1;
    }
    char buffer[BUFFER_LEN] = {0};
    file.read(buffer, BUFFER_LEN);
    while(file.gcount()) {
        vector<u8> store(file.gcount());
        for(int i=0; i<store.size(); i++)
            store[i] = buffer[i];
        hasher.update(store);
        file.read(buffer, BUFFER_LEN);
    }

    vector<u8> hash_output(32);
    hasher.finalize(hash_output);

    for(auto e: hash_output)
        cout << hex << setfill('0') << setw(2) << (int)e;
    cout << endl;

    return 0;
}