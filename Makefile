# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
# Currently, gcc seems to be better at vectorizing.

WARN = -Wall -Wextra -Wpedantic -Wstrict-prototypes -Wpointer-arith -Wcast-qual -Wwrite-strings
# CC = gcc $(WARN) -fopt-info-vec
CC = clang $(WARN) -Wno-gnu-folding-constant -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -fno-unroll-loops


# Source files and object files
SRC = src/tokenizer.c src/sampler.c src/transformer.c src/utils.c src/safetensors.c src/parson.c src/tensor.c src/transformer_info.c src/net.c src/benchmarking.c
OBJ = $(patsubst src/%.c,obj/%.o,$(SRC))
OPT = -Ofast -march=native -flto -fopenmp #-fopenmp-simd # -fopt-info-vec-missed # -fopenmp # -fopt-info-vec-missed
INC = -Isrc

.PHONY: all
all: bin/plainllm bin/stest bin/worker bin/client bin/benchmark

bin/%: obj/bin/%.o $(OBJ)
	@mkdir -p bin
	$(CC) $(OPT) $(INC) -g -o $@ $^ -lm

obj/%.o: src/%.c
	@mkdir -p obj
	$(CC) $(OPT) $(INC) -g -c -o $@ $<

obj/bin/%.o: src/bin/%.c
	@mkdir -p obj
	@mkdir -p obj/bin
	$(CC) $(OPT) $(INC) -g -c -o $@ $<

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
