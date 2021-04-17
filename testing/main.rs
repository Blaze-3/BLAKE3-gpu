fn main() {
    let mut hasher = reference_impl::Hasher::new();
    hasher.update(b"abc");
    hasher.update(b"def");
//    hasher.update(b"tres");
    let mut hash = [0; 32];
    hasher.finalize(&mut hash);
    let mut extended_hash = [0; 500];
    hasher.finalize(&mut extended_hash);
    assert_eq!(hash, extended_hash[..32]);
    // Extended output. OutputReader also implements Read and Seek.

// Print a hash as hex.
println!("{:?}", hash.to_vec());

}
