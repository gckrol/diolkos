#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "transformer.h"
#include "utils.h"
#include "safetensors.h"

// Helper function to get bit depth from quantization type
int get_bit_depth(quantization_type qt) {
    switch(qt) {
        case F32: return 32;
        case F16: return 16;
        case BF16: return 16;
        case Q8_0: return 8;
        default: return 0;
    }
}

void print_transformer_memory(Transformer* transformer) {
    Config* p = &transformer->config;
    Model* st = transformer->safetensors;

    size_t runstate_memory = 0;
    runstate_memory += tensor_memory(transformer->state.x);
    runstate_memory += tensor_memory(transformer->state.xb);
    runstate_memory += tensor_memory(transformer->state.xb2);
    runstate_memory += tensor_memory(transformer->state.hb);
    runstate_memory += tensor_memory(transformer->state.hb2);
    runstate_memory += tensor_memory(transformer->state.q);
    runstate_memory += tensor_memory(transformer->state.att);
    runstate_memory += tensor_memory(transformer->state.logits);

    size_t kv_memory = 0;
    kv_memory += tensor_memory(transformer->state.key_cache);
    kv_memory += tensor_memory(transformer->state.value_cache);

    // Track individual layer types
    typedef struct {
        const char* name;
        size_t memory;
        size_t params;
        quantization_type type;
    } LayerTypeInfo;

    // Define layer types to track
    enum {
        TOKEN_EMBEDDING,
        RMS_ATT,
        WQ,
        WK,
        WV,
        WO,
        RMS_FFN,
        W1,
        W2,
        W3,
        RMS_FINAL,
        WCLS,
        K_CACHE,
        V_CACHE,
        LAYER_TYPE_COUNT
    };

    LayerTypeInfo layer_types[LAYER_TYPE_COUNT] = {
        {"Token embedding", 0, 0, F32},
        {"  RMS Attention", 0, 0, F32},
        {"  Query weights", 0, 0, F32},
        {"  Key weights", 0, 0, F32},
        {"  Value weights", 0, 0, F32},
        {"  Output weights", 0, 0, F32},
        {"  RMS FFN", 0, 0, F32},
        {"  W1 weights", 0, 0, F32},
        {"  W2 weights", 0, 0, F32},
        {"  W3 weights", 0, 0, F32},
        {"  Final RMS norm", 0, 0, F32},
        {"Classifier", 0, 0, F32},
        {"Key cache", 0, 0, F32},
        {"Value cache", 0, 0, F32}
    };

    // Calculate per layer type stats
    for (int l = 0; l < p->n_layers; l++) {
        Layer* layer = &st->layers[l];
        
        // Attention layers
        layer_types[RMS_ATT].memory += tensor_memory(layer->rms_att_weight);
        layer_types[RMS_ATT].params += layer->rms_att_weight->dim;
        layer_types[RMS_ATT].type = layer->rms_att_weight->type;
        
        layer_types[WQ].memory += tensor_memory(layer->wq);
        layer_types[WQ].params += layer->wq->dim;
        layer_types[WQ].type = layer->wq->type;
        
        layer_types[WK].memory += tensor_memory(layer->wk);
        layer_types[WK].params += layer->wk->dim;
        layer_types[WK].type = layer->wk->type;
        
        layer_types[WV].memory += tensor_memory(layer->wv);
        layer_types[WV].params += layer->wv->dim;
        layer_types[WV].type = layer->wv->type;
        
        layer_types[WO].memory += tensor_memory(layer->wo);
        layer_types[WO].params += layer->wo->dim;
        layer_types[WO].type = layer->wo->type;
        
        // FFN layers
        layer_types[RMS_FFN].memory += tensor_memory(layer->rms_ffn_weight);
        layer_types[RMS_FFN].params += layer->rms_ffn_weight->dim;
        layer_types[RMS_FFN].type = layer->rms_ffn_weight->type;
        
        layer_types[W1].memory += tensor_memory(layer->w1);
        layer_types[W1].params += layer->w1->dim;
        layer_types[W1].type = layer->w1->type;
        
        layer_types[W2].memory += tensor_memory(layer->w2);
        layer_types[W2].params += layer->w2->dim;
        layer_types[W2].type = layer->w2->type;
        
        layer_types[W3].memory += tensor_memory(layer->w3);
        layer_types[W3].params += layer->w3->dim;
        layer_types[W3].type = layer->w3->type;
    }
    
    // Non-layer weights
    layer_types[TOKEN_EMBEDDING].memory = tensor_memory(st->token_embedding_table);
    layer_types[TOKEN_EMBEDDING].params = st->token_embedding_table->dim;
    layer_types[TOKEN_EMBEDDING].type = st->token_embedding_table->type;
    
    layer_types[RMS_FINAL].memory = tensor_memory(st->rms_final_weight);
    layer_types[RMS_FINAL].params = st->rms_final_weight->dim;
    layer_types[RMS_FINAL].type = st->rms_final_weight->type;
    
    layer_types[WCLS].memory = tensor_memory(st->wcls);
    layer_types[WCLS].params = st->wcls->dim;
    layer_types[WCLS].type = st->wcls->type;
    
    layer_types[K_CACHE].memory = tensor_memory(transformer->state.key_cache);
    layer_types[K_CACHE].params = transformer->state.key_cache->dim;
    layer_types[K_CACHE].type = transformer->state.key_cache->type;

    layer_types[V_CACHE].memory = tensor_memory(transformer->state.value_cache);
    layer_types[V_CACHE].params = transformer->state.value_cache->dim;
    layer_types[V_CACHE].type = transformer->state.value_cache->type;

    // Calculate total memory and parameters
    size_t model_memory = 0;
    size_t total_params = 0;
    
    for (int i = 0; i < LAYER_TYPE_COUNT; i++) {
        model_memory += layer_types[i].memory;
        total_params += layer_types[i].params;
    }

    // Print per-layer-type breakdown
    fprintf(stderr, "\nMemory by layer type:\n");
    fprintf(stderr, "%-20s %-10s %-15s %-15s %-15s\n", "Layer Type", "Quant", "Memory", "Params (M)", "Bytes/Param");
    
    for (int i = 0; i < LAYER_TYPE_COUNT; i++) {
        LayerTypeInfo* lt = &layer_types[i];
        
        // Display memory in appropriate units
        char memory_str[32];
        if (lt->memory >= 1024 * 1024 * 1024) {
            sprintf(memory_str, "%.2f GB", (float)lt->memory / 1024 / 1024 / 1024);
        } else {
            sprintf(memory_str, "%.2f MB", (float)lt->memory / 1024 / 1024);
        }
        
        float bytes_per_param = (float)lt->memory / lt->params;
        
        fprintf(stderr, "%-20s %-10s %-15s %-15.2f %-15.2f\n", 
                lt->name,
                quantization_type_to_string(lt->type), 
                memory_str,
                (float)lt->params / 1000000, 
                bytes_per_param);
    }
    
    fprintf(stderr, "Total parameters: %.2f M\n", (float)total_params / 1000000);
    // Print memory usage summary
    fprintf(stderr, "Total memory: %.2f GB (runstate: %.2f MB, kv: %.2f GB, model: %.2f GB)\n\n",
            (float)(runstate_memory + kv_memory + model_memory) / 1024 / 1024 / 1024,
            (float)runstate_memory / 1024 / 1024,
            (float)kv_memory / 1024 / 1024 / 1024,
            (float)model_memory / 1024 / 1024 / 1024);
}
