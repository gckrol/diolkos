#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

size_t read_full(int fd, void *buf, size_t count) {
    size_t bytes_read = 0;
    while (bytes_read < count) {
        size_t r = read(fd, (char*)buf + bytes_read, count - bytes_read);
        if (r <= 0) return r; // error or EOF
        bytes_read += r;
    }
    return bytes_read;
}

size_t write_full(int fd, const void *buf, size_t count) {
    size_t bytes_written = 0;
    while (bytes_written < count) {
        size_t w = write(fd, (const char*)buf + bytes_written, count - bytes_written);
        if (w <= 0) return w; // error or closed socket
        bytes_written += w;
    }
    return bytes_written;
}

void read_end_marker(int client_fd) {
    uint32_t end = 0;
    if (read_full(client_fd, &end, sizeof(end)) != sizeof(end)) {
        fprintf(stderr, "Error: failed to read end marker\n");
        close(client_fd);
        exit(EXIT_FAILURE);
    }
    if (end != 0xCAFEF00D) {
        fprintf(stderr, "Error: expected end marker 0xCAFEF00D, got 0x%X\n", end);
        close(client_fd);
        exit(EXIT_FAILURE);
    }
}

void write_end_marker(int client_fd) {
    uint32_t end = 0xCAFEF00D;
    write_full(client_fd, &end, sizeof(end));
}
