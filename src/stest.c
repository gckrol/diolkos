#include "safetensors.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <directory>\n", argv[0]);
        return 1;
    }

    // Load the safetensors file
    Safetensors *st = load_safetensors(argv[1]);
    if (st == NULL) {
        fprintf(stderr, "Failed to load safetensors file\n");
        return 1;
    }

    // Print the header size and contents
    printf("Header size: %lu bytes\n", st->header_size);
    printf("Header contents:\n");
    fwrite(st->header, 1, st->header_size, stdout);
    printf("\n");

    // Free the resources
    free_safetensors(st);

    return 0;
}