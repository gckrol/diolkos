#include "transformer.h"

// Functions for safetensors loading
void build_transformer_from_safetensors(Transformer *t, const char* model_path);
void free_transformer_safetensors(Transformer* t);
float* forward_safetensors(Transformer* transformer, int token, int pos);
