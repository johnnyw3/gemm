# set up thread count for multithreaded baseline
OPENBLAS_NUM_THREADS?="$$(nproc)"

TARGET?=skylake
CXXFLAGS?=-march=$(TARGET) -lopenblas -g


all: test

test: test.cxx
	clang++ -o test test.cxx $(CXXFLAGS) 

.PHONY: clean
clean:
	rm -f test
