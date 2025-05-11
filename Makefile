# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
# Currently, gcc seems to be better at vectorizing.

WARN = -Wall -Wextra -Wpedantic -Wstrict-prototypes -Wpointer-arith -Wcast-qual -Wwrite-strings -Werror=implicit-function-declaration
# CC = gcc $(WARN) -fopt-info-vec
CC = clang $(WARN) -Wno-gnu-folding-constant -fno-unroll-loops # -Rpass=loop-vectorize  -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize


# Source files and object files
SRC = $(wildcard src/*.c)
OBJ = $(patsubst src/%.c,obj/%.o,$(SRC))
OBJ_OMP = $(patsubst src/%.c,obj/omp/%.o,$(SRC))
OPT_OMP = -Ofast -march=native -flto -fopenmp #-fopenmp-simd # -fopt-info-vec-missed # -fopenmp # -fopt-info-vec-missed
OPT = -Ofast -march=native -flto -fopenmp-simd -pthread # -fopt-info-vec-missed # -fopenmp # -fopt-info-vec-missed
INC = -Isrc

.PHONY: compile_commands
compile_commands:
	bear make clean all

.PHONY: all
all: bin/plainllm bin/stest bin/worker bin/client bin/benchmark

# Use OMP by default.
bin/%: obj/bin/omp/%.o $(OBJ_OMP)
	@mkdir -p bin
	$(CC) $(OPT_OMP) $(INC) -g -o $@ $^ -lm

# No OMP for the worker - it uses pthreads.
bin/worker: obj/bin/worker.o $(OBJ)
	@mkdir -p bin
	$(CC) $(OPT) $(INC) -g -o $@ $^ -lm

bin/benchmarkp: obj/bin/benchmarkp.o $(OBJ)
	@mkdir -p bin
	$(CC) $(OPT) $(INC) -g -o $@ $^ -lm

obj/%.o: src/%.c
	@mkdir -p obj
	$(CC) $(OPT) $(INC) -g -c -o $@ $<

obj/omp/%.o: src/%.c
	@mkdir -p obj
	@mkdir -p obj/omp
	$(CC) $(OPT_OMP) $(INC) -g -c -o $@ $<

obj/bin/%.o: src/bin/%.c
	@mkdir -p obj
	@mkdir -p obj/bin
	$(CC) $(OPT) $(INC) -g -c -o $@ $<

obj/bin/omp/%.o: src/bin/%.c
	@mkdir -p obj
	@mkdir -p obj/bin
	@mkdir -p obj/bin/omp
	$(CC) $(OPT_OMP) $(INC) -g -c -o $@ $<


# Useful for testing - build + run with the small stories model.
.PHONY: run
run: all
	./bin/plainllm python/model.bin

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
.PHONY: debug
debug: 
	@mkdir -p bin
	$(CC) -g -o bin/plainllm $(SRC) -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: omp
omp:
	@mkdir -p bin
	$(CC) -Ofast -fopenmp -march=native $(SRC) -lm -o bin/plainllm

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
	@mkdir -p bin
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o bin/testc src/test.c -lm
	./bin/testc

.PHONY: clean
clean:
	rm -rf bin obj
