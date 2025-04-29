#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dirent.h>
#include <math.h>

#include "parson.h"
#include "transformer.h"

// Add at the end of unpermute_weights
void print_tensor_sample(const char* name, float* tensor, int count) {
    printf("Tensor %s - first %d values:\n", name, count);
    for (int i = 0; i < count; i++) {
        printf("  [%d] = %f\n", i, tensor[i]);
    }
    printf("\n");
}

void print_tensor_sum(const char* name, float* tensor, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += tensor[i];
    }
    printf("%s sum: %f first: %f last: %f\n", name, sum, tensor[0], tensor[size - 1]);
}

// BF16 to FP32 conversion function
float bf16_to_float(uint16_t bf16) {
    // BF16 has the same exponent bits as FP32 but only the top 7 mantissa bits
    // To convert: put the 16 bits in the top half of a 32-bit word, clear bottom 16 bits
    uint32_t bits = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));  // bit-level reinterpretation
    return result;
}

float f16_to_float(uint16_t f16) {
    // Extract components
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exponent = (f16 >> 10) & 0x1F;
    uint32_t mantissa = f16 & 0x3FF;
    
    // Special case: zero
    if (exponent == 0 && mantissa == 0) {
        return sign ? -0.0f : 0.0f;
    }
    
    // Special case: denormalized numbers
    if (exponent == 0) {
        float result = mantissa * powf(2.0f, -24.0f);
        return sign ? -result : result;
    }
    
    // Normalized number
    // Convert to F32 format: adjust exponent bias (15 -> 127) and shift mantissa
    uint32_t f32_bits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    float result;
    memcpy(&result, &f32_bits, sizeof(result));
    
    return result;
}

float *load_tensor(JSON_Object *o, void * data, const char *name, size_t expected_size) {
    JSON_Object *tensor_obj = json_object_get_object(o, name);
    if (tensor_obj == NULL) {
        fprintf(stderr, "Failed to get tensor object for %s\n", name);
        exit(EXIT_FAILURE);
    }
    const char *dtype = json_object_get_string(tensor_obj, "dtype");
    int item_size = 4;
    if (strcmp(dtype, "BF16") == 0) {
        item_size = 2;
    }
    if (strcmp(dtype, "F16") == 0) {
        item_size = 2;
    }
    JSON_Array *shape = json_object_get_array(tensor_obj, "shape");
    size_t n_dims = json_array_get_count(shape);
    size_t size = 1;
    for (size_t i = 0; i < n_dims; i++) {
        size *= json_array_get_number(shape, i);
    }
    if (size != expected_size) {
        fprintf(stderr, "Tensor size mismatch for %s: expected %zu, got %zu\n", name, expected_size, size);
        exit(EXIT_FAILURE);
    }
    JSON_Array *data_offsets = json_object_get_array(tensor_obj, "data_offsets");
    size_t start = json_array_get_number(data_offsets, 0);
    size_t end = json_array_get_number(data_offsets, 1);
    size_t data_size = end - start;
    if (data_size != size * item_size) {
        fprintf(stderr, "Data size mismatch for %s: expected %zu, got %zu\n", name, size * sizeof(float), data_size);
        exit(EXIT_FAILURE);
    }

    if (strcmp(dtype, "BF16") == 0) {
        // We now need to convert the half-precision floats to single-precision.
        // This is a bit tricky, as we need to read the data as uint16_t and then convert it.
        uint16_t *half_data = (uint16_t*)((uint8_t*)data + start);
        float *float_data = (float*)malloc(size * sizeof(float));
        if (float_data == NULL) {
            fprintf(stderr, "Failed to allocate memory for float data\n");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < size; i++) {
            uint16_t half = half_data[i];
            float_data[i] = bf16_to_float(half);
        }
        return float_data;
    }
    if (strcmp(dtype, "F16") == 0) {
        uint16_t *half_data = (uint16_t*)((uint8_t*)data + start);
        float *float_data = (float*)malloc(size * sizeof(float));
        if (float_data == NULL) {
            fprintf(stderr, "Failed to allocate memory for float data\n");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < size; i++) {
            uint16_t half = half_data[i];
            float_data[i] = f16_to_float(half);
        }
        return float_data;
    }

    return (float*)((uint8_t*)data + start);
}

// Process a single safetensors file and load its tensors
int process_safetensors_file(const char* filepath, Safetensors *st, Config *config) {
    int tensors_found = 0;
    Layer *layers = st->layers;
    int head_size = config->dim / config->n_heads;
    
    printf("Processing safetensors file: %s\n", filepath);
    
    // Open the file
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Failed to open safetensors file: %s\n", filepath);
        return 0;
    }
    
    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        fprintf(stderr, "Failed to get file size for: %s\n", filepath);
        close(fd);
        return 0;
    }
    
    // Memory map the file
    void *data = mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap file: %s\n", filepath);
        close(fd);
        return 0;
    }
    
    // Parse the header size (first 8 bytes, little endian)
    uint64_t header_size = 0;
    uint8_t *bytes = (uint8_t*)data;
    for (int i = 0; i < 8; i++) {
        header_size |= (uint64_t)bytes[i] << (i * 8);
    }
    
    // Get the tensors data position
    uint8_t *tensors = (uint8_t*)data + 8 + header_size;

    // Parse the header
    JSON_Value *header_value = json_parse_string((char*)data + 8);
    JSON_Object *header = json_value_get_object(header_value);
    
    // Try to load each tensor - if it exists in this file, we'll load it
    // Token embedding
    if (json_object_has_value(header, "model.embed_tokens.weight") && st->token_embedding_table == NULL) {
        st->token_embedding_table = load_tensor(header, tensors, "model.embed_tokens.weight", config->vocab_size * config->dim);
        tensors_found++;
    }
    
    // Final norm and classifier
    if (json_object_has_value(header, "model.norm.weight") && st->rms_final_weight == NULL) {
        st->rms_final_weight = load_tensor(header, tensors, "model.norm.weight", config->dim);
        tensors_found++;
    }
    
    if (json_object_has_value(header, "lm_head.weight") && st->wcls == NULL) {
        st->wcls = load_tensor(header, tensors, "lm_head.weight", config->vocab_size * config->dim);
        tensors_found++;
    }
    
    // Load all layer tensors that exist in this file
    for (int i = 0; i < config->n_layers; i++) {
        char layer_name[256];

        // Try to load each layer component
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.input_layernorm.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].rms_att_weight == NULL) {
            layers[i].rms_att_weight = load_tensor(header, tensors, layer_name, config->dim);
            tensors_found++;
        }
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.self_attn.q_proj.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].wq == NULL) {
            layers[i].wq = load_tensor(header, tensors, layer_name, config->dim * config->dim);
            tensors_found++;
        }
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.self_attn.k_proj.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].wk == NULL) {
            layers[i].wk = load_tensor(header, tensors, layer_name, config->dim * config->n_kv_heads * head_size);
            tensors_found++;
        }
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.self_attn.v_proj.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].wv == NULL) {
            layers[i].wv = load_tensor(header, tensors, layer_name, config->dim * config->n_kv_heads * head_size);
            tensors_found++;
        }
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.self_attn.o_proj.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].wo == NULL) {
            layers[i].wo = load_tensor(header, tensors, layer_name, config->dim * config->dim);
            tensors_found++;
        }
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.post_attention_layernorm.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].rms_ffn_weight == NULL) {
            layers[i].rms_ffn_weight = load_tensor(header, tensors, layer_name, config->dim);
            tensors_found++;
        }
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.mlp.gate_proj.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].w1 == NULL) {
            layers[i].w1 = load_tensor(header, tensors, layer_name, config->dim * config->hidden_dim);
            tensors_found++;
        }
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.mlp.down_proj.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].w2 == NULL) {
            layers[i].w2 = load_tensor(header, tensors, layer_name, config->dim * config->hidden_dim);
            tensors_found++;
        }
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.mlp.up_proj.weight", i);
        if (json_object_has_value(header, layer_name) && layers[i].w3 == NULL) {
            layers[i].w3 = load_tensor(header, tensors, layer_name, config->dim * config->hidden_dim);
            tensors_found++;
        }
    }
    
    // Clean up header but keep the mmap active
    json_value_free(header_value);
    close(fd);
    
    return tensors_found;
}

Safetensors *load_safetensors(const char* dir) {
    ///////////////////////////////
    // Load the config.json file
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", dir);

    Config *config = (Config*)malloc(sizeof(Config));
    JSON_Value *config_value = json_parse_file(config_path);
    JSON_Object *config_object = json_value_get_object(config_value);
    config->dim = json_object_get_number(config_object, "hidden_size");
    config->n_layers = json_object_get_number(config_object, "num_hidden_layers");
    config->n_heads = json_object_get_number(config_object, "num_attention_heads");
    config->n_kv_heads = json_object_get_number(config_object, "num_key_value_heads");
    config->vocab_size = json_object_get_number(config_object, "vocab_size");
    config->hidden_dim = json_object_get_number(config_object, "intermediate_size");
    config->seq_len = json_object_get_number(config_object, "max_position_embeddings");
    json_value_free(config_value);

    ///////////////
    // Allocate the Safetensors struct
    Safetensors *st = (Safetensors*)malloc(sizeof(Safetensors));
    if (st == NULL) {
        fprintf(stderr, "Failed to allocate memory for Safetensors struct\n");
        return NULL;
    }
    st->config = config;

    // Initialize layers
    Layer *layers = malloc(config->n_layers * sizeof(Layer));
    if (layers == NULL) {
        fprintf(stderr, "Failed to allocate memory for Layers\n");
        free(st);
        return NULL;
    }
    
    // Initialize all layer pointers to NULL
    memset(layers, 0, config->n_layers * sizeof(Layer));
    st->layers = layers;
    
    // Initialize st's top-level tensor pointers to NULL
    st->token_embedding_table = NULL;
    st->rms_final_weight = NULL;
    st->wcls = NULL;
    
    // Open the directory
    DIR *d = opendir(dir);
    if (d == NULL) {
        fprintf(stderr, "Failed to open directory: %s\n", dir);
        free(layers);
        free(st);
        return NULL;
    }
    
    // Count how many tensors we found
    int tensors_loaded = 0;
    
    // Scan for all .safetensors files
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        // Skip if not a .safetensors file
        if (strstr(entry->d_name, ".safetensors") == NULL) {
            continue;
        }
        
        // Build the full path to the safetensors file
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
        
        // Process this safetensors file and count found tensors
        tensors_loaded += process_safetensors_file(path, st, config);
    }
    
    closedir(d);
    
    if (tensors_loaded == 0) {
        fprintf(stderr, "No safetensors files found in directory: %s\n", dir);
        free(layers);
        free(st);
        return NULL;
    }
    
    printf("Loaded %d tensors from safetensors files\n", tensors_loaded);
    
    // Verify all required tensors were loaded
    // Check if we're missing any critical tensors
    if (st->token_embedding_table == NULL) {
        fprintf(stderr, "Missing required tensor: model.embed_tokens.weight\n");
        free(layers);
        free(st);
        return NULL;
    }
    
    if (st->rms_final_weight == NULL) {
        fprintf(stderr, "Missing required tensor: model.norm.weight\n");
        free(layers);
        free(st);
        return NULL;
    }
    
    if (st->wcls == NULL) {
        fprintf(stderr, "Missing required tensor: lm_head.weight\n");
        free(layers);
        free(st);
        return NULL;
    }
    
    // Check all layers
    for (int i = 0; i < config->n_layers; i++) {
        if (layers[i].rms_att_weight == NULL || 
            layers[i].wq == NULL ||
            layers[i].wk == NULL ||
            layers[i].wv == NULL ||
            layers[i].wo == NULL ||
            layers[i].rms_ffn_weight == NULL ||
            layers[i].w1 == NULL ||
            layers[i].w2 == NULL ||
            layers[i].w3 == NULL) {
            
            fprintf(stderr, "Missing required tensor in layer %d\n", i);
            free(layers);
            free(st);
            return NULL;
        }
    }

    return st;
}
