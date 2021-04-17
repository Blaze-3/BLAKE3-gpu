# Runing the reference implimentation

### Step1: make sure you have all the packages
> Make sure you have got ability to tun rust programs. Apart form that, you will also need `cargo`, a build tool (gets installed automatically with rust on linux)
### Step2: Create a project using cargo
> $ `cargo new ref_test`
>This will make a directory called 'ref_test '. 
### Step3: Edit the Cargo.toml file
> Under the `[dependencies]` section paste the following line.  
`reference_impl={path="/home/cicada3301/Desktop/BLAKE3/reference_impl"}`
Replace /home/cicada3301/Desktop/BLAKE3/reference_impl with your path, the reference_impl file is provided above.

### Step4: Edit the main.rs file in `ref_test/src' directory.
> Basically use the main.rs here.
 
### Step5: Build and Run
> Navigate to the `ref_test` directory, a src directory and Cargo.toml file must be visible.  
$ `cargo build`  
$ `cargo run`   


