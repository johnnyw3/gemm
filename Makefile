# set up thread count for multithreaded baseline
OPENBLAS_NUM_THREADS?="$$(nproc)"

# for logging etc
TEMP_FNAME=$(shell date +%FT%H:%M:%S)_log.txt

TARGET?=skylake
CXXFLAGS?=-march=$(TARGET) -lopenblas -g -O3 -fsave-optimization-record

all: test

test: gemm.cxx gemm.h simd_common.h
	clang++ -o test gemm.cxx $(CXXFLAGS) 

.PHONY: clean perf_report
perf_report: test
	perf stat -d -d -d ./test mat1_4096.txt mat2_4096.txt 2>&1 | tee $(TEMP_FNAME)
	perf stat -e l2_lines_out.useless_hwpf,cycle_activity.stalls_l1d_miss ./test mat1_4096.txt mat2_4096.txt 2>&1 | tee -a $(TEMP_FNAME)

clean:
	rm -f test
