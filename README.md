# GEMM 

High performance GEMM kernels written in C++ using AVX and AMX
intrinsics. Currently achieves 75-105% performance of OpenBLAS (at least on
systems I have tested with). Peak performance is than OpenBLAS on Granite Rapids! (* in
single-threaded applications, there are bugs in my multithreaded code currently)

Some optimization techniques were inspired by Salykov's article on the topic [2],
but the algorithm used here is different than the one described in the article.
On my systems, I can achieve comparable performance to Salykov's code.

# Benchmarks

Metrics are in GFLOPs; speedups are compared to OpenBLAS (we also
compared to Intel's MKL and used whichever library was faster as the baseline). Types are `fp32` for
AVX-based kernels and `bf16` input/`fp32` result for AMX.

**Single-threaded, n=4096** average of 10 runs

| Kernel | CPU | This algorithm | OpenBLAS/MKL | Speedup |
|:-------|:----|---------------:|---------:|:------------------|
AVX2 | **Skylake (Kaby Lake)** i5-8350u | 78 | 103 | 0.76 |
AVX-512 | **Tiger Lake** i5-1135G7 | 106 | 122 | 0.87 |
AMX | **Granite Rapids** Xeon 6972P | 1279 | 1229 | 1.04 |

**4 threads, n=4096** average of 10 runs

| Kernel | CPU | This algorithm | OpenBLAS/MKL | Speedup |
|:-------|:----|---------------:|---------:|:------------------|
AVX2 | **Skylake (Kaby Lake)** i5-8350u | 213 |  282 | 0.76 |
AVX-512 | **Tiger Lake** i5-1135G7 | 380 | 465 | 0.82 |
AMX | **Granite Rapids** Xeon 6972P | 4810 | 4675 | 1.03 |

# Prerequisites

## Hardware

A modern x86 processor supporting AVX2, AVX-512, or AMX must be used.

## Software

For the library:

* `pthreads` support (for multithreading)
* A version of `make` that supports the `shell` directive
* A modern C++ compiler. For AMX support, `clang` version 16+ or `gcc` version
  13+ must be used as these compilers support the `__bf16` type.

For the `bench` program:

* OpenBLAS (for verifying correctness and providing a baseline to compare against)
    - If using AMX, OpenBLAS must be compiled with `BUILD_BFLOAT16=1` (NOT the
      default)

# Build

To build, use the given makefile, specifying your march. For example:

```bash
$ make TARGET=skylake
```

The fastest available kernel (AVX2, AVX-512, or AMX) will be chosen based on the
features available in the `TARGET` architecture.

By default, multithreading is enabled using as many threads as logical
processors on your system. To change the number of threads, pass the
`NUM_THREADS=<num>` option:

```bash
$ make TARGET=skylake NUM_THREADS=4
```

You'll need to `make clean` first before changing configurations.

## Install

There is an `install` target provided in the makefile:

```bash
$ make install PREFIX=/your/sysroot/dir
```

This will copy the required libraries and header file(s) for use in other
applications.

# References

[1] U. Drepper, “What Every Programmer Should Know About Memory,” Nov. 2007, [Online]. Available: https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

[2] A. Salykov, “Advanced Matrix Multiplication Optimization on Modern Multi-Core Processors,” salykova. Accessed: Jun. 20, 2025. [Online]. Available: https://salykova.github.io/matmul-cpu

[3] “Intel Advanced Vector Extensions 512  (Intel AVX-512) - Permuting Data Within  and Between AVX Registers,” Intel. Accessed: Jun. 19, 2025. [Online]. Available: https://builders.intel.com/docs/networkbuilders/intel-avx-512-permuting-data-within-and-between-avx-registers-technology-guide-1668169807.pdf
