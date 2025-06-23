## simd

High performance GEMM algorithm written in C using AVX
intrinsics. Currently achieves 75-80% performance of OpenBLAS (at least on my
system). 

Some optimization techniques were inspired by Salykov's article on the topic [2],
by the algorithm used here is different than the one described in the article.
On my system, I can achieve comparable performance to Salykov's code.

# Prerequisites

For the library:

* `pthreads` support (for multithreading)
* A version of `make` that supports the `shell` directive

For the `bench` program:

* OpenBLAS (for verifying correctness and providing a baseline to compare against)

# Build

To build, use the given makefile, specifying your march. For example:

```bash
$ make TARGET=skylake
```

By default, multithreading is enabled using as many threads as logical
processors on your system. To change the number of threads, pass the
`NUM_THREADS=<num>` option:

```bash
$ make TARGET=skylake NUM_THREADS=4
```

You'll need to `make clean` first before changing configurations.

# References

[1] U. Drepper, “What Every Programmer Should Know About Memory,” Nov. 2007, [Online]. Available: https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

[2] A. Salykov, “Advanced Matrix Multiplication Optimization on Modern Multi-Core Processors,” salykova. Accessed: Jun. 20, 2025. [Online]. Available: https://salykova.github.io/matmul-cpu
