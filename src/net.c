#include <stddef.h>

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
