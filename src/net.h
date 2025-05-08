#ifndef NET_H
#define NET_H

#include <stddef.h>

struct iovec;

size_t read_full(int fd, void *buf, size_t count);
size_t write_full(int fd, const void *buf, size_t count);
ssize_t writev_full(int fd, struct iovec *iov, int iovcnt);
ssize_t readv_full(int fd, struct iovec *iov, int iovcnt);

void read_end_marker(int client_fd);
void write_end_marker(int client_fd);

#endif // NET_H
