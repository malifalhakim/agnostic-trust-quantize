import torch
import gc
import os
import json

from torch import nn
from safetensors.torch import load_file
from hybrid_layer import SafetyHybridLayer


def dequantize_packed_layer(packed_weight, scales, group_size = 128):
    """
    Unpacks int32 weights into FP16/BF16 weights simulating quantization noise.
    
    Args:
        packed_weight (torch.Tensor): Shape [out_features, in_features // 8], int32
        scales (torch.Tensor): Shape [out_features, in_features // group_size], bf16
        group_size (int): Usually 128 for GPTQ/AWQ models.
    """
    out_features, packed_in = packed_weight.shape
    num_weight_per_int = 8
    original_in_features = packed_in * num_weight_per_int

    mask = torch.tensor(0xF, device=packed_weight.device, dtype=torch.int32)

    unpacked_cols = []
    for i in range(num_weight_per_int):
        shift = i * 4
        unpacked_segment = (packed_weight >> shift) & mask
        unpacked_cols.append(unpacked_segment)
    
    w_int = torch.stack(unpacked_cols, dim=-1).view(out_features, -1)
    w_fp = w_int.to(scales.dtype) - 8.0

    scales_expanded = torch.repeat_interleave(scales, group_size, dim=1)

    w_dequantized = w_fp * scales_expanded

    return w_dequantized

def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    elif torch.xpu.is_available():
        return "xpu:0"
    else:
        return "cpu"
    
def clear_memory(weight=None):
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()

def get_named_linears(model, modules_to_not_convert=None):
    modules = model.model.layers
    all_named_linears = {}
    for i in range(len(modules)):
        module = modules[i]
        named_linears = {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}
        named_linears = exclude_layers_to_not_quantize(named_linears, modules_to_not_convert)

        module_prefix = get_op_name(model, module) + "."
        for name, layer in named_linears.items():
            full_name = module_prefix + name
            all_named_linears[full_name] = layer
    return all_named_linears

def exclude_layers_to_not_quantize(linear_layers, modules_to_not_convert):
    if modules_to_not_convert is None:
        return linear_layers

    filtered_layers = {}
    for name, linear_layer in linear_layers.items():
        if not any(key in name for key in modules_to_not_convert):
            filtered_layers[name] = linear_layer
    return filtered_layers

def get_op_name(module, op):
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")

def apply_safety_split_to_model(model, safety_masks):
    """
    Replaces target layers in an LLM with SafetyHybridLayer.
    
    Args:
        model: The loaded LLM (in FP16/BF16).
        safety_masks: A dict { 'layer_name': mask_tensor }. 
                      Keys must match model.named_modules().
    """
    print(f"Applying safety split to {len(safety_masks)} layers...")
    model_device = model.device
    
    for layer_name, mask in safety_masks.items():
        # 1. Locate the layer and its parent in the model tree
        if '.' in layer_name:
            parent_name, child_name = layer_name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ""
            child_name = layer_name
            parent = model

        # Get the original layer
        original_layer = getattr(parent, child_name)
        
        # Ensure we are replacing a Linear layer
        if not isinstance(original_layer, nn.Linear):
            print(f"Skipping {layer_name}: Not a Linear layer.")
            continue

        # 2. Create the Hybrid Layer
        hybrid_layer = SafetyHybridLayer(original_layer, mask.to(model_device))
        
        # 3. The Hot-Swap
        setattr(parent, child_name, hybrid_layer)
        
        del original_layer
        torch.cuda.empty_cache()
        
        print(f"Converted: {layer_name}")

    return model

def save_hybrid_model(model, tokenizer, output_dir, safety_masks=None, push_to_hub=False, repo_id=None):
    """
    Saves the hybrid model, tokenizer, and hybrid layer metadata.
    Optionally pushes to Hugging Face Hub.
    """
    print(f"Saving hybrid model to {output_dir}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save Model and Tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata about which layers are hybrid (crucial for loading)
    if safety_masks is not None:
        hybrid_layers = list(safety_masks.keys())
        with open(os.path.join(output_dir, "hybrid_layers.json"), "w") as f:
            json.dump(hybrid_layers, f, indent=2)
    
    if push_to_hub and repo_id:
        print(f"Pushing to Hugging Face Hub: {repo_id}")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        
        if safety_masks is not None:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=os.path.join(output_dir, "hybrid_layers.json"),
                path_in_repo="hybrid_layers.json",
                repo_id=repo_id
            )
    
    print("Model saved successfully.")