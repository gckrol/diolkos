#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>      // for close()
#include <arpa/inet.h>   // for sockaddr_in, inet_ntoa()
#include <assert.h>
#include <time.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <math.h>

#include "worker_commands.h"
#include "tensor.h"
#include "utils.h"
#include "net.h"
#include "fnv1a.h"
#include "threading.h"

// VS Code shows this as undefined.
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 0
#endif

typedef struct Slice {
    Tensor *matrix;
    Tensor *input_vector;
    Tensor *output_vector;
} Slice;

Slice *slices = NULL;
#define MAX_SLICES 1024

void load_matrix(int client_fd) {
    // printf("Loading matrix... ");
    uint32_t type;
    uint32_t dim_in;
    uint32_t dim_out;
    uint32_t slice_id;
    uint128_t hash;

    read_full(client_fd, &slice_id, sizeof(slice_id));
    read_full(client_fd, &type, sizeof(type));
    read_full(client_fd, &dim_in, sizeof(dim_in));
    read_full(client_fd, &dim_out, sizeof(dim_out));
    read_full(client_fd, &hash, sizeof(hash));

    assert(dim_in % 32 == 0);
    assert(dim_out % 32 == 0);

    Slice *s = &slices[slice_id];

    size_t dim = (size_t)dim_in * (size_t)dim_out;

    tensor_destroy(s->matrix);
    tensor_destroy(s->input_vector);
    tensor_destroy(s->output_vector);

    s->matrix = tensor_create(dim, type);
    s->input_vector = tensor_create(dim_in, Q8_0);
    s->output_vector = tensor_create(dim_out, Q8_0);

    // printf("#%d %u x %u %s\n", slice_id, dim_in, dim_out, quant_t_to_string(type));
    // printf("Reading %zu bytes\n", Tensor_storage_size(s->matrix));

    read_full(client_fd, s->matrix->data, Tensor_storage_size(s->matrix));

    read_end_marker(client_fd);

    // Write to cache file.
    char cache_name[512];
    snprintf(cache_name, sizeof(cache_name), "cache/%016llx%016llx.slice", hash.high, hash.low);
    mkdir("cache", 0755);
    FILE *cache_file = fopen(cache_name, "wb");
    if (cache_file == NULL) {
        fprintf(stderr, "Failed to open cache file for writing: %s\n", cache_name);
        exit(EXIT_FAILURE);
    }
    fwrite(s->matrix->data, Tensor_storage_size(s->matrix), 1, cache_file);
    fclose(cache_file);
    fprintf("Cache file written: %s\n", cache_name);
}

void load_matrix_hash(int client_fd) {
    // printf("Loading matrix hash... ");
    uint32_t slice_id;
    uint32_t type;
    uint32_t dim_in;
    uint32_t dim_out;
    uint128_t hash;

    read_full(client_fd, &slice_id, sizeof(slice_id));
    read_full(client_fd, &type, sizeof(type));
    read_full(client_fd, &dim_in, sizeof(dim_in));
    read_full(client_fd, &dim_out, sizeof(dim_out));
    read_full(client_fd, &hash, sizeof(hash));

    // printf("Loading matrix hash: %d %u x %u %s\n", slice_id, dim_in, dim_out, quant_t_to_string(type));

    assert(dim_in % 32 == 0);
    assert(dim_out % 32 == 0);

    Slice *s = &slices[slice_id];

    size_t dim = (size_t)dim_in * (size_t)dim_out;

    tensor_destroy(s->matrix);
    tensor_destroy(s->input_vector);
    tensor_destroy(s->output_vector);

    s->input_vector = tensor_create(dim_in, Q8_0);
    s->output_vector = tensor_create(dim_out, Q8_0);

    // Generate cache file name
    char cache_name[512];
    snprintf(cache_name, sizeof(cache_name), "cache/%016llx%016llx.slice", hash.high, hash.low);
    // printf("Cache name: %s\n", cache_name);
    // Check if the cache file exists
    struct stat st;
    if (stat(cache_name, &st) == 0) {
        // printf("Cache file found: %s\n", cache_name);
        s->matrix = calloc(1, sizeof(Tensor));
        // MMap the cache file
        int fd = open(cache_name, O_RDONLY);
        if (fd == -1) {
            fprintf(stderr, "Failed to open cache file: %s\n", cache_name);
            exit(EXIT_FAILURE);
        }
        // printf("Cache file size: %zu\n", st.st_size);
        void *cache_data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (cache_data == MAP_FAILED) {
            fprintf(stderr, "Failed to mmap cache file: %s\n", cache_name);
            close(fd);
            exit(EXIT_FAILURE);
        }
        s->matrix->type = type;
        s->matrix->data = (TensorData*)cache_data;
        s->matrix->hdim = dim_in;
        s->matrix->vdim = dim_out;
        s->matrix->dim = dim;
        s->matrix->fd = fd;
        if (type == Q8_0) {
            s->matrix->scale = (float*)((uint8_t*)cache_data + s->matrix->dim);
        }
        write_full(client_fd, "\x00", 1);
        return;
    }
    s->matrix = tensor_create(dim, type);
    write_full(client_fd, "\x01", 1);
}   

static double time_in_ms2(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

void read_tensors(int client_fd, Tensor **vector, int num_vectors) {
    int entries = num_vectors + 1;
    struct iovec iov[entries];
    int c = 0;

    for (int i = 0; i < num_vectors; i++) {
        iov[c].iov_base = vector[i]->data;
        iov[c++].iov_len = Tensor_storage_size(vector[i]);
    }

    uint32_t end_marker = 1;
    iov[c].iov_base = &end_marker;
    iov[c++].iov_len = sizeof(end_marker);

    assert(c == entries);
    readv_full(client_fd, iov, entries);

    // Verify the end marker
    if (end_marker != 0xCAFEF00D) {
        fprintf(stderr, "Error: expected end marker 0xCAFEF00D, got 0x%X\n",
                end_marker);
        close(client_fd);
        exit(EXIT_FAILURE);
    }
}

void write_tensors(int client_fd, Tensor **vector, int num_vectors) {
    int entries = num_vectors + 1;
    struct iovec iov[entries];
    int c = 0;

    for (int i = 0; i < num_vectors; i++) {
        iov[c].iov_base = vector[i]->data;
        iov[c++].iov_len = Tensor_storage_size(vector[i]);
    }

    uint32_t end_marker = 0xCAFEF00D;
    iov[c].iov_base = &end_marker;
    iov[c++].iov_len = sizeof(end_marker);

    assert(c == entries);
    writev_full(client_fd, iov, entries);
}

void multiply(int client_fd, bool perform_matmul) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t slice_id;
    read_full(client_fd, &slice_id, sizeof(slice_id));
    Slice *s = &slices[slice_id];
    // printf("Multiplying slice %d\n", slice_id);
    // printf("Input vector type: %s\n", quant_t_to_string(s->input_vector->type));
    // printf("Input vector dim: %zu\n", s->input_vector->dim);
    // printf("Reading %zu bytes\n", Tensor_storage_size(s->input_vector));

    // Read using the read_vectors function
    read_tensors(client_fd, &s->input_vector, 1);

    // printf("Input vector type: %s\n", quant_t_to_string(s->input_vector->type));
    // for (int i = 0; i < s->input_vector->dim / 32; i++) {
    //     // printf("%f\n", s->input_vector->scale[i]);
    //     assert(!reliable_isnan(s->input_vector->scale[i]));
    // }

    // printf("Ready for matrix multiplication.\n");
    // printf("Input vector type: %s\n", quant_t_to_string(s->input_vector->type));
    // printf("Input vector dim: %zu\n", s->input_vector->dim);
    // printf("Matrix type: %s\n", quant_t_to_string(s->matrix->type));
    // printf("Matrix dim: %zu\n", s->matrix->dim);
    // printf("Output vector type: %s\n", quant_t_to_string(s->output_vector->type));
    // printf("Output vector dim: %zu\n", s->output_vector->dim);

    if (perform_matmul) {
        matmul_parallel(s->output_vector, s->input_vector, s->matrix);
    } else {
        // Keep Valgrind happy.
        memset(s->output_vector->data, 0, Tensor_storage_size(s->output_vector));
    }

    write_tensors(client_fd, &s->output_vector, 1);

    clock_gettime(CLOCK_MONOTONIC, &end);
    // printf("Function took %.3f ms\n", time_in_ms2(&start, &end));    
}

void multiply_qkv(int client_fd) {
    // For this one we get 3 matrix ids, and 1 input vector.
    uint32_t slice_ids[3];
    read_full(client_fd, slice_ids, sizeof(slice_ids));

    Tensor *input_vector;
    // Borrow the input vector from the first slice.
    input_vector = slices[slice_ids[0]].input_vector;

    // Read using the read_vectors function
    read_tensors(client_fd, &input_vector, 1);

    Tensor *output_vectors[3];
    for (int i = 0; i < 3; i++) {
        output_vectors[i] = slices[slice_ids[i]].output_vector;
    }

    for (int i = 0; i < 3; i++) {
        Tensor *matrix = slices[slice_ids[i]].matrix;
        Tensor *output_vector = output_vectors[i];
        matmul_parallel(output_vector, input_vector, matrix);
    }

    write_tensors(client_fd, output_vectors, 3);
}

void ffn_silu(int client_fd) {
    // 1 input, 1 output, 3 matrices.
    uint32_t slice_ids[3];
    read_full(client_fd, slice_ids, sizeof(slice_ids));

    Slice *s1 = &slices[slice_ids[0]];
    Slice *s2 = &slices[slice_ids[1]];
    Slice *s3 = &slices[slice_ids[2]];

    read_tensors(client_fd, &s1->input_vector, 1);

    // TODO: memory allocation.
    Tensor *hb = tensor_create(s1->output_vector->dim, F32);
    Tensor *hb2 = tensor_create(s3->output_vector->dim, F32);

    matmul_parallel_f32(hb, s1->input_vector, s1->matrix);
    matmul_parallel_f32(hb2, s1->input_vector, s3->matrix);

    // SwiGLU non-linearity
    float *hb_data = data_f32(hb);
    float *hb2_data = data_f32(hb2);
    #pragma omp simd
    for (int i = 0; i < hb->dim; i++) {
        float val = hb_data[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= hb2_data[i];
        hb_data[i] = val;
    }

    // Quantize.
    convert_into(s2->input_vector, hb);

    matmul_parallel(s2->output_vector, s2->input_vector, s2->matrix);

    write_tensors(client_fd, &s2->output_vector, 1);

    tensor_destroy(hb);
    tensor_destroy(hb2);
}

int main(int argc, char *argv[]) {
    init_threads();
    init_utils(4096*4, 0);

    slices = calloc(MAX_SLICES, sizeof(Slice));

    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);

    // Listen on the port provided in argv[1]
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <port>\n", argv[0]);
        return 1;
    }
    int port = atoi(argv[1]);
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "Invalid port number: %d\n", port);
        return 1;
    }

    // 1. Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // 2. Bind to a port
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;  // 0.0.0.0
    server_addr.sin_port = htons(port);
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // 3. Listen for connections
    if (listen(server_fd, 1) < 0) {
        perror("listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }


    for (;;) {
        printf("Server listening on port %d...\n", port);

        // 4. Accept a connection
        client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) {
            perror("accept failed");
            close(server_fd);
            exit(EXIT_FAILURE);
        }
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int));

        // Busy wait (more CPU, but much lower latency).
        int val = 1;
#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 1
#endif
        setsockopt(client_fd, SOL_SOCKET, SO_BUSY_POLL, &val, sizeof(val));

        printf("Connected to client: %s\n", inet_ntoa(client_addr.sin_addr));

        for (;;) {
            // 5. Communicate
            // printf("Waiting for command...\n");
            uint16_t command;
            ssize_t bytes_read = read(client_fd, &command, sizeof(command));
            if (bytes_read <= 0) {
                printf("Client disconnected\n");
                break;
            }
            // printf("Received command: %u\n", command);
            if (command == CMD_LOAD_MATRIX) {
                load_matrix(client_fd);
            } else if (command == CMD_LOAD_MATRIX_HASH) {
                load_matrix_hash(client_fd);
            } else if (command == CMD_MULTIPLY) {
                multiply(client_fd, true);
            } else if (command == CMD_MULTIPLY_QKV) {
                multiply_qkv(client_fd);
            } else if (command == CMD_MULTIPLY_OVERHEAD) {
                multiply(client_fd, false);
            } else if (command == CMD_FFN_SILU) {
                ffn_silu(client_fd);
            } else if (command == CMD_PING) {
                // Simply send back a single byte for ping latency measurement
                uint8_t response = 0x01;
                write_full(client_fd, &response, sizeof(response));
            } else {
                printf("Unknown command: %u\n", command);
                break;
            }
            
        }

        // 6. Cleanup
        close(client_fd);

        printf("Client disconnected: %s\n", inet_ntoa(client_addr.sin_addr));

        // Free the slices
        for (int i = 0; i < MAX_SLICES; i++) {
            if (slices[i].matrix) {
                tensor_destroy(slices[i].matrix);
                slices[i].matrix = NULL;
            }
            if (slices[i].input_vector) {
                tensor_destroy(slices[i].input_vector);
                slices[i].input_vector = NULL;
            }
            if (slices[i].output_vector) {
                tensor_destroy(slices[i].output_vector);
                slices[i].output_vector = NULL;
            }
        }
    }

    close(server_fd);

    return 0;
}