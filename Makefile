# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = clang

# Source files and object files
SRC = src/main.c src/tokenizer.c src/sampler.c src/transformer.c src/utils.c
OBJ = $(patsubst src/%.c,obj/%.o,$(SRC))

.PHONY: all
all: bin/plainllm

bin/plainllm: $(OBJ)
	@mkdir -p bin
	$(CC) -Ofast -march=native -g -o $@ $^ -lm

obj/%.o: src/%.c
	@mkdir -p obj
	$(CC) -Ofast -march=native -g -c -o $@ $<

# Useful for testing - build + run with the small stories model.
.PHONY: run
run: all
	./bin/plainllm stories15M.bin

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
