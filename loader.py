import os
import json
from safetensors import safe_open
from huggingface_hub import snapshot_download

class ShardedLoader:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.is_local = os.path.isdir(model_path)

        if self.is_local:
            self.base_path = model_path
        else:
            self.base_path = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors", "*.json"]
            )
        
        index_path = os.path.join(self.base_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = None

        self.file_handles = {}
    
    def get_tensor(self, tensor_name):
        """
        Automatically finds the right file and retrieves the tensor.
        """
        if self.weight_map:
            if tensor_name not in self.weight_map:
                raise ValueError(f"{tensor_name} not found in model index")
            filename = self.weight_map[tensor_name]
        else:
            filename = "model.safetensors"
        
        file_path = os.path.join(self.base_path, filename)

        if filename not in self.file_handles:
            self.file_handles[filename] = safe_open(file_path, framework="pt", device=self.device)
        
        return self.file_handles[filename].get_tensor(tensor_name)
    
    def keys(self):
        """
        Returns all available parameter names.
        """
        if self.weight_map:
            return list(self.weight_map.keys())
        else:
            f = self.get_tensor(list(self.file_handles.keys())[0]) if self.file_handles else None

            return []
            