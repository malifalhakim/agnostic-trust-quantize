import torch
import gc

from torch import nn
from safetensors.torch import load_file

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