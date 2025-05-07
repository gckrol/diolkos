#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>      // for close()
#include <arpa/inet.h>   // for sockaddr_in, inet_ntoa()
#include <assert.h>
#include <time.h>

#include "worker_commands.h"
#include "tensor.h"
#include "utils.h"
#include "net.h"
#include <netinet/tcp.h>

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
    printf("Loading matrix... ");
    uint32_t type;
    uint32_t dim_in;
    uint32_t dim_out;
    uint32_t slice_id;

    read_full(client_fd, &slice_id, sizeof(slice_id));
    read_full(client_fd, &type, sizeof(type));
    read_full(client_fd, &dim_in, sizeof(dim_in));
    read_full(client_fd, &dim_out, sizeof(dim_out));

    Slice *s = &slices[slice_id];

    size_t dim = (size_t)dim_in * (size_t)dim_out;

    tensor_destroy(s->matrix);
    tensor_destroy(s->input_vector);
    tensor_destroy(s->output_vector);

    s->matrix = tensor_create(dim, type);
    s->input_vector = tensor_create(dim_in, Q8_0);
    s->output_vector = tensor_create(dim_out, F32);

    printf("#%d %u x %u %s\n", slice_id, dim_in, dim_out, quant_t_to_string(type));
    printf("Reading %zu bytes\n", Tensor_storage_size(s->matrix));

    read_full(client_fd, s->matrix->data, Tensor_storage_size(s->matrix));

    read_end_marker(client_fd);
}

static double time_in_ms2(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

void multiply(int client_fd) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t slice_id;
    read_full(client_fd, &slice_id, sizeof(slice_id));
    Slice *s = &slices[slice_id];
    // printf("Multiplying slice %d\n", slice_id);
    // printf("Input vector type: %s\n", quant_t_to_string(s->input_vector->type));
    // printf("Input vector dim: %zu\n", s->input_vector->dim);
    // printf("Reading %zu bytes\n", Tensor_storage_size(s->input_vector));
    read_full(client_fd, s->input_vector->data, Tensor_storage_size(s->input_vector));
    read_end_marker(client_fd);

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

    matmul(s->output_vector, s->input_vector, s->matrix, s->input_vector->dim, s->output_vector->dim);

    // printf("Writing %zu bytes\n", s->output_vector->dim * quant_size(s->output_vector->type));
    write_full(client_fd, s->output_vector->data, s->output_vector->dim * quant_size(s->output_vector->type));
    write_end_marker(client_fd);

    clock_gettime(CLOCK_MONOTONIC, &end);
    // printf("Function took %.3f ms\n", time_in_ms2(&start, &end));    
}

int main(int argc, char *argv[]) {
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

        printf("Connected to client: %s\n", inet_ntoa(client_addr.sin_addr));

        for (;;) {
            // 5. Communicate
            // printf("Waiting for command...\n");
            uint16_t command;
            ssize_t bytes_read = read(client_fd, &command, sizeof(command));
            if (bytes_read <= 0) {
                printf("Reading command failed\n");
                break;
            }
            // printf("Received command: %u\n", command);
            if (command == CMD_LOAD_MATRIX) {
                load_matrix(client_fd);
            } else if (command == CMD_MULTIPLY) {
                multiply(client_fd);
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