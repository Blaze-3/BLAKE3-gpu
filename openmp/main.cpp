#include "blaze3.h"
#include <fstream>
#include <iomanip>
#include <omp.h>

// Do not change this, it has to be equal to chunk size
#define BUFFER_LEN CHUNK_LEN
// Max depth of thread nesting allowed. Should technically be log2(SNICKER)
#define NEST_LEVELS 3

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

    // Not really helpful since we're not using cin
    // ios_base::sync_with_stdio(false);
    // cin.tie(NULL);

    // open-mp settings
    #if defined(_OPENMP)
    // regions should execute with two threads
    omp_set_num_threads(2);
    // enable nested threading for N levels
    omp_set_max_active_levels(NEST_LEVELS);
    #endif

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