# Prerequisites

For the `bench` program:

    - OpenBLAS (for verifying correctness and providing a baseline to compare against)

# Build

To build, use the given makefile, specifying your march. For example:

```bash
$ make TARGET=skylake
```

By default, OpenMP is enabled to take advantage of multi-core systems. To
disable OpenMP, pass `USE_OPENMP=no`:

```bash
$ make TARGET=skylake USE_OPENMP=no
```

# References

[1] U. Drepper, “What Every Programmer Should Know About Memory,” Nov. 2007, [Online]. Available: https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
[2] `__M512_REDUCE_OP` macro in GCC. Available: https://github.com/gcc-mirror/gcc/blob/9d7e19255c06e05ad791e9bf5aefc4783a12c4f9/gcc/config/i386/avx512fintrin.h#L15928
