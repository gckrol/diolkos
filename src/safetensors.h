#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int fd;            // file descriptor
    void *data;        // pointer to mmap'd data
    size_t size;       // total size of the file
    uint64_t header_size;  // size of the header in bytes
    char *header;      // pointer to the header (JSON)
    uint8_t *tensors;  // pointer to the start of tensor data
} Safetensors;

/**
 * Load a model.safetensors file from the given directory
 * 
 * @param dir Directory containing the model.safetensors file
 * @return Pointer to Safetensors struct or NULL on failure
 */
Safetensors *load_safetensors(const char* dir);

/**
 * Free resources associated with a Safetensors struct
 * 
 * @param st Pointer to Safetensors struct
 */
void free_safetensors(Safetensors *st);

#endif // SAFETENSORS_H