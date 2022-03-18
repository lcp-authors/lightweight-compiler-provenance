# Code for Section 2.1

This is intended to be run on the dataset from `datasets/section-2-1`. Run me like the following:
* `python3 cfg-size.py <directory of binaries> <output filename> <number of jobs to run>`
    * This produces `amd64-cfg-blocksize.log` and `armhf-cfg-blocksize.log` in `outputs/section-2-1`.
* `python3 distinct-opcodes-registers.py <directory of binaries> <output filename> <number of jobs to run> <"opcodes" or "registers> <"amd64" or "armhf">`
    * This produces `amd64-opcode-counts.log`, `amd64-register-counts.log`, `armhf-opcode-counts.log`, and `armhf-register-counts.log` in `outputs/section-2-1`.