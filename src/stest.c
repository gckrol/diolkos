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

    return 0;
}