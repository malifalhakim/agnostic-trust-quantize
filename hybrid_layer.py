import torch
import torch.nn as nn
import bitsandbytes as bnb

class SafetyHybridLayer(nn.Module):
    def __init__(self, original_layer, safety_mask):
        super().__init__()
        
        self.safe_branch = nn.Linear(
            original_layer.in_features, 
            original_layer.out_features, 
            bias=(original_layer.bias is not None),
            dtype=torch.float16
        )
        
        self.bulk_branch = bnb.nn.Linear8bitLt(
            original_layer.in_features,
            original_layer.out_features,
            bias=(original_layer.bias is not None),
            has_fp16_weights=False,
            threshold=6.0
        )

        with torch.no_grad():
            w_full = original_layer.weight.data.clone()
            
            w_safe = w_full * safety_mask
            self.safe_branch.weight.copy_(w_safe)
            
            w_bulk = w_full * (1 - safety_mask)
            self.bulk_branch.weight.copy_(w_bulk)
            
            if original_layer.bias is not None:
                self.safe_branch.bias.copy_(original_layer.bias)
                self.bulk_branch.bias = None 

    def forward(self, x):
        # High Precision Safety Calculation
        out_safe = self.safe_branch(x)
        
        # Quantized Bulk Calculation
        out_bulk = self.bulk_branch(x)
        
        # Combine
        return out_safe + out_bulk