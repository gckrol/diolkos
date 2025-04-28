# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = clang

.PHONY: all
all: run.c
	$(CC) -Ofast -march=native -g -o run run.c -lm

# Useful for testing - build + run with the small stories model.
run: all
	./run stories15M.bin

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
debug: run.c
	$(CC) -g -o run run.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: omp
omp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
