#ifndef NET_H
#define NET_H

#include <stddef.h>

size_t read_full(int fd, void *buf, size_t count);
size_t write_full(int fd, const void *buf, size_t count);

#endif // NET_H
