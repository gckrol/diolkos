#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/uio.h>

size_t read_full(int fd, void *buf, size_t count) {
    size_t bytes_read = 0;
    while (bytes_read < count) {
        size_t r = read(fd, (char*)buf + bytes_read, count - bytes_read);
        if (r <= 0) {
            fprintf(stderr, "Error: read failed. Read %zu bytes of %zu\n", bytes_read, count);
            return r;
        }
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

ssize_t writev_full(int fd, struct iovec *iov, int iovcnt) {
    ssize_t total = 0;
    while (iovcnt > 0) {
        ssize_t n = writev(fd, iov, iovcnt);
        if (n <= 0) return -1; // error

        total += n;

        // Advance the iov array
        while (n > 0 && iovcnt > 0) {
            if (n >= (ssize_t)iov->iov_len) {
                n -= iov->iov_len;
                iov++;
                iovcnt--;
            } else {
                iov->iov_base = (char *)iov->iov_base + n;
                iov->iov_len -= n;
                n = 0;
            }
        }
    }
    return total;
}

ssize_t readv_full(int fd, struct iovec *iov, int iovcnt) {
    ssize_t total = 0;
    while (iovcnt > 0) {
        ssize_t n = readv(fd, iov, iovcnt);
        if (n <= 0) return -1; // EOF or error

        total += n;

        while (n > 0 && iovcnt > 0) {
            if (n >= (ssize_t)iov->iov_len) {
                n -= iov->iov_len;
                iov++;
                iovcnt--;
            } else {
                iov->iov_base = (char *)iov->iov_base + n;
                iov->iov_len -= n;
                n = 0;
            }
        }
    }
    return total;
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
