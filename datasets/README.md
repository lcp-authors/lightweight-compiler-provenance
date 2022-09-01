# Datasets

This directory holds binaries and their disassembly for our five experiments -- one to compare code generation on x86-64 against ARM in Section 3, and the rest to evaluate our model in Sections 5.1-5.4.

* `section-3` contains 976 pairs of (x86-64 binary, ARM32 binary) compiled from identical source code, plus their disassembly.
* `section-5-1` contains 14,140 ARM64 binaries and their objdumps, originally sourced from the artifacts of the paper _Identifying Compiler and Optimization Level in Binary Code from Multiple Architectures_ by Pizzolotto and Inoue.
* `section-5-2` contains 8,181 ARM32 binaries and their objdumps which we compiled from several open-source C and C++ projects, including the GNU `diffutils`, `findutils`, `coreutils`, and `inetutils` suites.
* `section-5-3` contains 2,058 ARM32 binaries and their objdumps which we compiled from a different pool of source code which contains projects from the pool of `section-5-2`, plus additional projects. Some binaries in this dataset come from CompCert, a formally verified C compiler.
* `section-5-4` contains 2,330 ARM32 binaries and their objdumps which ship with three Linux distributions (CentOS, Raspberry Pi OS, and Ubuntu). We also include a link to Zenodo with 39,937 more (binary, objdump) pairs which were too large for GitHub.
