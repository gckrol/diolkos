#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

Safetensors *load_safetensors(const char* dir) {
    // Allocate the Safetensors struct
    Safetensors *st = (Safetensors*)malloc(sizeof(Safetensors));
    if (st == NULL) {
        fprintf(stderr, "Failed to allocate memory for Safetensors struct\n");
        return NULL;
    }

    // Build the path to model.safetensors
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    
    // Open the file
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Failed to open safetensors file: %s\n", path);
        free(st);
        return NULL;
    }
    
    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        fprintf(stderr, "Failed to get file size\n");
        close(fd);
        free(st);
        return NULL;
    }
    
    // Memory map the file
    void *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap file\n");
        close(fd);
        free(st);
        return NULL;
    }
    
    // Store the file descriptor and mapped data
    st->fd = fd;
    st->data = data;
    st->size = sb.st_size;
    
    // Parse the header size (first 8 bytes, little endian)
    uint64_t header_size = 0;
    uint8_t *bytes = (uint8_t*)data;
    for (int i = 0; i < 8; i++) {
        header_size |= (uint64_t)bytes[i] << (i * 8);
    }
    
    // Store the header size and position
    st->header_size = header_size;
    st->header = (char*)data + 8;
    st->tensors = (uint8_t*)data + 8 + header_size;

    return st;
}

void free_safetensors(Safetensors *st) {
    if (st) {
        if (st->data) {
            munmap(st->data, st->size);
        }
        if (st->fd != -1) {
            close(st->fd);
        }
        free(st);
    }
}