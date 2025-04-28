# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = clang

.PHONY: all
all: src/main.c
	$(CC) -Ofast -march=native -g -o bin/plainllm src/main.c -lm

# Useful for testing - build + run with the small stories model.
run: all
	./bin/plainllm stories15M.bin

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
debug: src/main.c
	$(CC) -g -o bin/plainllm src/main.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: omp
omp: main.c
	$(CC) -Ofast -fopenmp -march=native src/main.c -lm -o bin/plainllm

# run all tests
.PHONY: test
test:
	pytest

# run only tests for main.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o bin/testc src/test.c -lm
	./bin/testc

.PHONY: clean
clean:
	rm -f bin
