from transformers import AutoModelForCausalLM
import torch
import numpy as np

# Load model
model_path = "Felladrin/Llama-160M-Chat-v1"
model = AutoModelForCausalLM.from_pretrained(model_path)

# Print the first few values of key tensors
def print_tensor(name, tensor, count=5):
    print(f"Tensor {name} - first {count} values:")
    flat_tensor = tensor.flatten()
    for i in range(count):
        print(f"  [{i}] = {flat_tensor[i].item()}")
    print()
    
def print_tensor_sum(name, tensor):
    flat_tensor = tensor.flatten()
    total_sum = torch.sum(flat_tensor)
    print(f"{name} sum: {total_sum.item()} first: {flat_tensor[0].item()} last: {flat_tensor[-1].item()}")

# Print tensor sums for debugging
print_tensor_sum("token_embedding_table", model.model.embed_tokens.weight)
print_tensor_sum("model.norm.weight", model.model.norm.weight)
print_tensor_sum("model.embed_tokens.weight", model.model.embed_tokens.weight)

for i in range(model.config.num_hidden_layers):
    print_tensor_sum(f"model.layers.{i}.input_layernorm.weight", model.model.layers[i].input_layernorm.weight)
    print_tensor_sum(f"model.layers.{i}.self_attn.q_proj.weight", model.model.layers[i].self_attn.q_proj.weight)
    print_tensor_sum(f"model.layers.{i}.self_attn.k_proj.weight", model.model.layers[i].self_attn.k_proj.weight)
    print_tensor_sum(f"model.layers.{i}.self_attn.v_proj.weight", model.model.layers[i].self_attn.v_proj.weight)
    print_tensor_sum(f"model.layers.{i}.self_attn.o_proj.weight", model.model.layers[i].self_attn.o_proj.weight)
    print_tensor_sum(f"model.layers.{i}.post_attention_layernorm.weight", model.model.layers[i].post_attention_layernorm.weight)
    print_tensor_sum(f"model.layers.{i}.mlp.gate_proj.weight", model.model.layers[i].mlp.gate_proj.weight)
    print_tensor_sum(f"model.layers.{i}.mlp.down_proj.weight", model.model.layers[i].mlp.down_proj.weight)
    print_tensor_sum(f"model.layers.{i}.mlp.up_proj.weight", model.model.layers[i].mlp.up_proj.weight)

# Print raw tensor values before permutation
layer_idx = 0
q_weight = model.model.layers[layer_idx].self_attn.q_proj.weight
print_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight (raw)", q_weight)

# Print values after permutation (if permuted)
def permute(w, n_heads):
    dim1, dim2 = w.shape
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

q_weight_perm = permute(q_weight, model.config.num_attention_heads)
print_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight (permuted)", q_weight_perm)

# For completion, print unpermuted version (should match your C unpermute result)
def unpermute(w, n_heads):
    dim1, dim2 = w.shape
    head_dim = dim1 // n_heads
    return w.view(n_heads, 2, head_dim // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

q_weight_unperm = unpermute(q_weight_perm, model.config.num_attention_heads)
print_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight (unpermuted)", q_weight_unperm)

# Also check original vs unpermuted
print("First few elements match:", torch.allclose(q_weight[:5,:5], q_weight_unperm[:5,:5]))