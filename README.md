## simd

High performance GEMM algorithm written in C using AVX
intrinsics. Currently achieves 75-80% performance of OpenBLAS (at least on my
system). 

Some optimization techniques were inspired by Salykov's article on the topic [2],
by the algorithm used here is different than the one described in the article.
On my system, I can achieve comparable performance to Salykov's code.

# Benchmarks

(I know, fight me for testing on mobile CPUs)

Metrics are in GFLOPs.

**Single-threaded, n=4096, fp32,** average of 5 runs

| CPU | This algorithm | OpenBLAS | Ratio of OpenBLAS |
|:----|---------------:|---------:|:------------------|
**Skylake (Kaby Lake)** i5-8350u | 76 | 100 | 0.76 |
**Tiger Lake** i5-1135G7 | 100 | 122 | 0.82 |

**4 threads, n=4096, fp32,** average of 5 runs

| CPU | This algorithm | OpenBLAS | Ratio of OpenBLAS |
|:----|---------------:|---------:|:------------------|
**Tiger Lake** i5-1135G7 | 355 | 465 | 0.76 |

# Prerequisites

## Hardware

A modern x86 processor supporting AVX2 or AVX-512 must be used.

## Software

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

When compiling for a target that supports AVX-512, AVX-512 operations will 
automatically be used.

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
