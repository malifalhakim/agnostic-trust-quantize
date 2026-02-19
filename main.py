import os
import torch
import json
from safetensors.torch import save_file, load_file
from tqdm import tqdm

from utils import dequantize_packed_layer, get_named_linears
from loader import ShardedLoader
from scoring import TrustScoring

# --- CONFIGURATION ---
ORIGINAL_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
QUANTIZED_MODEL_PATH = "Amadeus99/Qwen2.5-7B-Instruct-GPTQ"
OUTPUT_DIR = "Qwen2.5-7B-Instruct-GPTQ-trust"
BETA = 1.0
TAU = 0.4
SCORE_SAVED = "./saved_scores"
# ---------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_save_hybrid_model():
    # 1. Load the Original Model's Index Map
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(ORIGINAL_MODEL_PATH, allow_patterns=["*.index.json"])
        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_data = json.load(f)
            weight_map = index_data["weight_map"]
    except:
        # Fallback if model is not sharded (single file)
        weight_map = None
        print("Model appears to be non-sharded (single file).")

    # 2. Group layers by file
    if weight_map:
        files_to_process = {}
        for layer_name, filename in weight_map.items():
            if filename not in files_to_process:
                files_to_process[filename] = []
            files_to_process[filename].append(layer_name)
    else:
        # Single file case
        files_to_process = {"model.safetensors": ["ALL_LAYERS"]}
    
    # 3. Initialize loaders
    loader_orig = ShardedLoader(ORIGINAL_MODEL_PATH)
    loader_quant = ShardedLoader(QUANTIZED_MODEL_PATH)

    # 4. Calculate trust score for all linear layers
    scorer = TrustScoring(ORIGINAL_MODEL_PATH, beta=BETA, tau=TAU)
    all_layers_scores = scorer.calculate_trustworthinesscore(cache_dir=SCORE_SAVED)
    threshold = scorer.get_threshold_score(all_layers_scores)

    # 5. Iterate over each Shard
    for filename, layers in files_to_process.items():
        shared_state_dict = {}

        orig_shard_path = os.path.join(loader_orig.base_path, filename)
        orig_tensors = load_file(orig_shard_path)

        for layer_name in tqdm(layers):
            orig_weight = orig_tensors[layer_name]

            if "mlp" in layer_name or "self_attn" in layer_name:
                if layer_name.endswith(".weight"):
                    base_name = layer_name.replace(".weight", "")
                    try:
                        packed = loader_quant.get_tensor(f"{base_name}.weight_packed")
                        scales = loader_quant.get_tensor(f"{base_name}.weight_scale")
                        
                        noisy_weight = dequantize_packed_layer(packed, scales)

                        scores = all_layers_scores[base_name]
                        hybrid_weight = TrustScoring.apply_trust_preservation(orig_weight, noisy_weight, scores, threshold)

                        shared_state_dict[layer_name] = hybrid_weight
                    except Exception as e:
                        print(f"Warning: Could not process {layer_name} (likely not quantized). Keeping original.")
                        shared_state_dict[layer_name] = orig_weight
                else:
                    # Keep bias as is
                    shared_state_dict[layer_name] = orig_weight
            else:
                # Keep Embeddings / Output Head as is
                shared_state_dict[layer_name] = orig_weight
    
        # 6. Save the modified shard
        save_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Saving modified shard to {save_path}")
        save_file(shared_state_dict, save_path)

        del shared_state_dict
        torch.cuda.empty_cache()

    if weight_map:
        with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=2)
    
    print("Copying config files...")
    os.system(f"cp {os.path.join(loader_orig.base_path, '*.json')} {OUTPUT_DIR}")
    print("Hybrid Model Saved Successfully!")

if __name__ == "__main__":
    process_save_hybrid_model()


