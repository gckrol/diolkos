#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dirent.h>
#include <math.h>

#include "safetensors.h"
#include "parson.h"
#include "transformer.h"
#include "tensor.h"
#include "utils.h"

Tensor *load_tensor(JSON_Object *o, const char *model_name, void * data, const char *name, size_t expected_size, quant_t target_type) {
    Tensor *tensor = calloc(1, sizeof(Tensor));

    JSON_Object *tensor_obj = json_object_get_object(o, name);
    if (tensor_obj == NULL) {
        fprintf(stderr, "Failed to get tensor object for %s\n", name);
        exit(EXIT_FAILURE);
    }
    const char *dtype = json_object_get_string(tensor_obj, "dtype");
    int item_size = -1;
    if (strcmp(dtype, "F32") == 0) {
        item_size = 4;
        tensor->type = F32;
    } else if (strcmp(dtype, "BF16") == 0) {
        item_size = 2;
        tensor->type = BF16;
    } else if (strcmp(dtype, "F16") == 0) {
        item_size = 2;
        tensor->type = F16;
    }
    JSON_Array *shape = json_object_get_array(tensor_obj, "shape");
    size_t n_dims = json_array_get_count(shape);

    tensor->vdim = 1;
    tensor->hdim = 1;

    for (size_t i = 0; i < n_dims; i++) {
        size_t d = json_array_get_number(shape, i);
        if (i == 0) {
            tensor->vdim = d;
        } else if (i == 1) {
            tensor->hdim = d;
        }
    }
    tensor->dim = tensor->vdim * tensor->hdim;
    if (tensor->dim != expected_size) {
        fprintf(stderr, "Tensor size mismatch for %s: expected %zu, got %zu\n", name, expected_size, tensor->dim);
        exit(EXIT_FAILURE);
    }

    // Generate cache file name
    char cache_name[512];
    snprintf(cache_name, sizeof(cache_name), "cache/%s/%s-%s.tensor", model_name, name, quant_t_to_string(target_type));
    // Check if the cache file exists
    struct stat st;
    if (stat(cache_name, &st) == 0) {
        // MMap the cache file
        int fd = open(cache_name, O_RDONLY);
        if (fd == -1) {
            fprintf(stderr, "Failed to open cache file: %s\n", cache_name);
            exit(EXIT_FAILURE);
        }
        void *cache_data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (cache_data == MAP_FAILED) {
            fprintf(stderr, "Failed to mmap cache file: %s\n", cache_name);
            close(fd);
            exit(EXIT_FAILURE);
        }
        tensor->type = target_type;
        tensor->data = (TensorData*)cache_data;
        if (target_type == Q8_0) {
            tensor->scale = (float*)((uint8_t*)cache_data + tensor->dim);
        }
        return tensor;
    }

    JSON_Array *data_offsets = json_object_get_array(tensor_obj, "data_offsets");
    size_t start = json_array_get_number(data_offsets, 0);
    size_t end = json_array_get_number(data_offsets, 1);
    size_t data_size = end - start;
    if (data_size != tensor->dim * item_size) {
        fprintf(stderr, "Data size mismatch for %s: expected %zu, got %zu\n", name, tensor->dim * sizeof(float), data_size);
        exit(EXIT_FAILURE);
    }

    tensor->data = (TensorData*)((uint8_t*)data + start);

    Tensor *result = convert(tensor, target_type);
    if (!result) {
        fprintf(stderr, "Unsupported conversion of tensor %s:  %d -> %d\n", name, tensor->type, target_type);
        exit(EXIT_FAILURE);
    }

    // Write cache file
    // Create cache directory if it doesn't exist
    char cache_dir[512];
    snprintf(cache_dir, sizeof(cache_dir), "cache/%s", model_name);
    mkdir(cache_dir, 0755);
    FILE *cache_file = fopen(cache_name, "wb");
    if (cache_file == NULL) {
        fprintf(stderr, "Failed to open cache file for writing: %s\n", cache_name);
        exit(EXIT_FAILURE);
    }
    fwrite(result->data, quant_size(target_type), result->dim, cache_file);
    if (result->scale) {
        fwrite(result->scale, sizeof(float), result->dim / 32, cache_file);
    }
    fclose(cache_file);

    return result;
}

// Process a single safetensors file and load its tensors
int process_safetensors_file(const char* model_name, const char* filepath, Model *st, Config *config) {
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
    void *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
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
    
    // First, iterate over all tensors in the file
    size_t count = json_object_get_count(header);
    for (size_t j = 0; j < count; j++) {
        const char *tensor_name = json_object_get_name(header, j);
        
        // Check for token embedding
        // TODO: llama.cpp quantizes this to 8b.
        if (strcmp(tensor_name, "model.embed_tokens.weight") == 0 && st->token_embedding_table == NULL) {
            st->token_embedding_table = load_tensor(header, model_name, tensors, tensor_name, config->vocab_size * config->dim, Q8_0);
            tensors_found++;
            continue;
        }
        
        // Check for final norm
        if (strcmp(tensor_name, "model.norm.weight") == 0 && st->rms_final_weight == NULL) {
            st->rms_final_weight = load_tensor(header, model_name, tensors, tensor_name, config->dim, F32);
            tensors_found++;
            continue;
        }
        
        // Check for classifier
        if (strcmp(tensor_name, "lm_head.weight") == 0 && st->wcls == NULL) {
            st->wcls = load_tensor(header, model_name, tensors, tensor_name, config->vocab_size * config->dim, Q8_0);
            tensors_found++;
            continue;
        }
        
        // Check for layer tensors using prefix matching
        const char *layer_prefix = "model.layers.";
        if (strncmp(tensor_name, layer_prefix, strlen(layer_prefix)) == 0) {
            // Extract the layer index
            int layer_idx = -1;
            char component[256];
            if (sscanf(tensor_name + strlen(layer_prefix), "%d.%255s", &layer_idx, component) == 2) {
                if (layer_idx >= 0 && layer_idx < config->n_layers) {
                    // Match different layer component patterns
                    
                    // Input layernorm
                    if (strcmp(component, "input_layernorm.weight") == 0 && layers[layer_idx].rms_att_weight == NULL) {
                        layers[layer_idx].rms_att_weight = load_tensor(header, model_name, tensors, tensor_name, config->dim, F32);
                        tensors_found++;
                        continue;
                    }
                    
                    // Q projection
                    if (strcmp(component, "self_attn.q_proj.weight") == 0 && layers[layer_idx].wq == NULL) {
                        layers[layer_idx].wq = load_tensor(header, model_name, tensors, tensor_name, config->dim * config->dim, Q8_0);
                        tensors_found++;
                        continue;
                    }
                    
                    // K projection
                    if (strcmp(component, "self_attn.k_proj.weight") == 0 && layers[layer_idx].wk == NULL) {
                        layers[layer_idx].wk = load_tensor(header, model_name, tensors, tensor_name, config->dim * config->n_kv_heads * head_size, Q8_0);
                        tensors_found++;
                        continue;
                    }
                    
                    // V projection
                    if (strcmp(component, "self_attn.v_proj.weight") == 0 && layers[layer_idx].wv == NULL) {
                        layers[layer_idx].wv = load_tensor(header, model_name, tensors, tensor_name, config->dim * config->n_kv_heads * head_size, Q8_0);
                        tensors_found++;
                        continue;
                    }
                    
                    // O projection
                    if (strcmp(component, "self_attn.o_proj.weight") == 0 && layers[layer_idx].wo == NULL) {
                        layers[layer_idx].wo = load_tensor(header, model_name, tensors, tensor_name, config->dim * config->dim, Q8_0);
                        tensors_found++;
                        continue;
                    }
                    
                    // Post attention layernorm
                    if (strcmp(component, "post_attention_layernorm.weight") == 0 && layers[layer_idx].rms_ffn_weight == NULL) {
                        layers[layer_idx].rms_ffn_weight = load_tensor(header, model_name, tensors, tensor_name, config->dim, F32);
                        tensors_found++;
                        continue;
                    }
                    
                    // MLP gate projection
                    if (strcmp(component, "mlp.gate_proj.weight") == 0 && layers[layer_idx].w1 == NULL) {
                        layers[layer_idx].w1 = load_tensor(header, model_name, tensors, tensor_name, config->dim * config->hidden_dim, Q8_0);
                        tensors_found++;
                        continue;
                    }
                    
                    // MLP down projection
                    if (strcmp(component, "mlp.down_proj.weight") == 0 && layers[layer_idx].w2 == NULL) {
                        layers[layer_idx].w2 = load_tensor(header, model_name, tensors, tensor_name, config->dim * config->hidden_dim, Q8_0);
                        tensors_found++;
                        continue;
                    }
                    
                    // MLP up projection
                    if (strcmp(component, "mlp.up_proj.weight") == 0 && layers[layer_idx].w3 == NULL) {
                        layers[layer_idx].w3 = load_tensor(header, model_name, tensors, tensor_name, config->dim * config->hidden_dim, Q8_0);
                        tensors_found++;
                        continue;
                    }
                }
            }
        }
    }
    
    // Clean up header but keep the mmap active
    json_value_free(header_value);
    close(fd);
    
    return tensors_found;
}

Model *load_safetensors(const char* dir) {

    // Extract model name from the directory
    const char *model_name = strrchr(dir, '/');
    if (model_name == NULL) {
        model_name = dir;
    } else {
        model_name++;
    }

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
    // Allocate the Model struct
    Model *st = calloc(1, sizeof(Model));
    if (st == NULL) {
        fprintf(stderr, "Failed to allocate memory for Model struct\n");
        exit(EXIT_FAILURE);
    }
    st->config = config;
    st->huggingface_rope = true;

    // Initialize layers
    Layer *layers = calloc(config->n_layers, sizeof(Layer));
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
        exit(EXIT_FAILURE);
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
        tensors_loaded += process_safetensors_file(model_name, path, st, config);
    }
    
    closedir(d);
    
    if (tensors_loaded == 0) {
        fprintf(stderr, "No safetensors files found in directory: %s\n", dir);
        free(layers);
        free(st);
        exit(EXIT_FAILURE);
    }
    
    printf("Loaded %d tensors from safetensors files\n", tensors_loaded);
    
    // Verify all required tensors were loaded
    // Check if we're missing any critical tensors
    if (st->token_embedding_table == NULL) {
        fprintf(stderr, "Missing required tensor: model.embed_tokens.weight\n");
        free(layers);
        free(st);
        exit(EXIT_FAILURE);
    }
    
    if (st->rms_final_weight == NULL) {
        fprintf(stderr, "Missing required tensor: model.norm.weight\n");
        free(layers);
        free(st);
        exit(EXIT_FAILURE);
    }
    
    if (st->wcls == NULL) {
        fprintf(stderr, "Missing required tensor: lm_head.weight\n");
        free(layers);
        free(st);
        exit(EXIT_FAILURE);
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
            exit(EXIT_FAILURE);
        }
    }

    init_utils(config->dim, config->hidden_dim);

    return st;
}
