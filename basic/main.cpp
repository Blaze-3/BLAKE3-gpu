#include "reference.h"

int main() {
    cout << "Blake hasher in cpp\n";
    Hasher hasher = Hasher::_new();
    string input = "Good hash function";
    vector<u8> bytes(begin(input), end(input));
    vector<u8> hash_output(32);
    hasher.update(bytes);
    cout << "Updated hasher with data\n";
    hasher.finalize(hash_output);
    cout << "Hash of: " << input << ": \n";
    for(auto e: hash_output)
        cout << (int)e << " ";
    cout << endl;
    cout.flush();
}