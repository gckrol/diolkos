#include "transformer.h"

// Functions for safetensors loading
void build_transformer_from_safetensors(Transformer *t, const char* model_path);
float* forward_safetensors(Transformer* transformer, int token, int pos);
