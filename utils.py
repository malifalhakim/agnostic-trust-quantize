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

def get_named_linears(model):
    module = model.model.layers
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}