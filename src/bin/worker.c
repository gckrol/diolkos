#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>      // for close()
#include <arpa/inet.h>   // for sockaddr_in, inet_ntoa()
#include <assert.h>

#include "worker_commands.h"
#include "tensor.h"
#include "utils.h"
#include "net.h"

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

    Tensor_destroy(s->matrix);
    Tensor_destroy(s->input_vector);
    Tensor_destroy(s->output_vector);

    s->matrix = Tensor_create(dim, type);
    s->input_vector = Tensor_create(dim_in, Q8_0);
    s->output_vector = Tensor_create(dim_out, F32);

    printf("#%d %u x %u %s\n", slice_id, dim_in, dim_out, quantization_type_to_string(type));
    printf("Reading %zu bytes\n", Tensor_storage_size(s->matrix));

    read_full(client_fd, s->matrix->data, Tensor_storage_size(s->matrix));

    uint32_t end;
    read_full(client_fd, &end, sizeof(end));
    if(end != 0xCAFEF00D) {
        fprintf(stderr, "Error: expected end marker 0xCAFEF00D, got 0x%X\n", end);
        close(client_fd);
        exit(EXIT_FAILURE);
    }

}

void multiply(int client_fd) {
    uint32_t slice_id;
    read_full(client_fd, &slice_id, sizeof(slice_id));
    Slice *s = &slices[slice_id];
    read_full(client_fd, s->input_vector->data, s->input_vector->dim * quant_size(s->input_vector->type));
    matmul(s->output_vector, s->input_vector, s->matrix, s->input_vector->dim, s->output_vector->dim);
    write_full(client_fd, s->output_vector->data, s->output_vector->dim * quant_size(s->output_vector->type));
}

int main(int argc, char *argv[]) {
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

        printf("Connected to client: %s\n", inet_ntoa(client_addr.sin_addr));

        for (;;) {
            // 5. Communicate
            printf("Waiting for command...\n");
            uint16_t command;
            ssize_t bytes_read = read(client_fd, &command, sizeof(command));
            if (bytes_read <= 0) {
                printf("Reading command failed\n");
                break;
            }
            printf("Received command: %u\n", command);
            if (command = CMD_LOAD_MATRIX) {
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
                Tensor_destroy(slices[i].matrix);
                slices[i].matrix = NULL;
            }
            if (slices[i].input_vector) {
                Tensor_destroy(slices[i].input_vector);
                slices[i].input_vector = NULL;
            }
            if (slices[i].output_vector) {
                Tensor_destroy(slices[i].output_vector);
                slices[i].output_vector = NULL;
            }
        }
    }

    close(server_fd);

    return 0;
}