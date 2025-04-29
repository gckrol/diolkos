#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

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

    return (float*)((uint8_t*)data + start);
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
    void *data = mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0); // Write is to unpermute.
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

    JSON_Value *header_value = json_parse_string(st->header);
    JSON_Object *header = json_value_get_object(header_value);

    JSON_Object *lm_head = json_object_get_object(header, "lm_head.weight");
    const char *head_type = json_object_get_string(lm_head, "dtype");

    // load_tensor(header, st->tensors, "lm_head.weight", config->vocab_size * config->dim);
    st->token_embedding_table = load_tensor(header, st->tensors, "model.embed_tokens.weight", config->vocab_size * config->dim);
    st->rms_final_weight = load_tensor(header, st->tensors, "model.norm.weight", config->dim);
    st->wcls = load_tensor(header, st->tensors, "lm_head.weight", config->vocab_size * config->dim); // Sometimes tied to token_embedding_table!

    Layer *layers = malloc(config->n_layers * sizeof(Layer));
    st->layers = layers;
    
    // Load all layers
    int head_size = config->dim / config->n_heads;
    for (int i = 0; i < config->n_layers; i++) {
        char layer_name[256];

        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.input_layernorm.weight", i);
        layers[i].rms_att_weight = load_tensor(header, st->tensors, layer_name, config->dim);
        
        // (layer, dim, n_heads * head_size)
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.self_attn.q_proj.weight", i);
        layers[i].wq = load_tensor(header, st->tensors, layer_name, config->dim * config->dim);
        // Note: is permuted by hugginface
        // layers[i].wq = unpermute_weights(layers[i].wq, config->dim, config->dim, config->n_heads);
        
        // dim, n_kv_heads * head_size
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.self_attn.k_proj.weight", i);
        layers[i].wk = load_tensor(header, st->tensors, layer_name, config->dim * config->n_kv_heads * head_size);
        // Note: is permuted by hugginface
        // layers[i].wk = unpermute_weights(layers[i].wk, config->dim, config->n_kv_heads * head_size, config->n_kv_heads);
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.self_attn.v_proj.weight", i);
        layers[i].wv = load_tensor(header, st->tensors, layer_name, config->dim * config->n_kv_heads * head_size);
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.self_attn.o_proj.weight", i);
        layers[i].wo = load_tensor(header, st->tensors, layer_name, config->dim * config->dim);
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.post_attention_layernorm.weight", i);
        layers[i].rms_ffn_weight = load_tensor(header, st->tensors, layer_name, config->dim);
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.mlp.gate_proj.weight", i);
        layers[i].w1 = load_tensor(header, st->tensors, layer_name, config->dim * config->hidden_dim);
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.mlp.down_proj.weight", i);
        layers[i].w2 = load_tensor(header, st->tensors, layer_name, config->dim * config->hidden_dim);
        
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d.mlp.up_proj.weight", i);
        layers[i].w3 = load_tensor(header, st->tensors, layer_name, config->dim * config->hidden_dim);
    }

    json_value_free(header_value);

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