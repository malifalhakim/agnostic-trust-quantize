import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scoring import TrustScoring
from utils import apply_safety_split_to_model, save_hybrid_model

# --- CONFIGURATION ---
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "Qwen2.5-7B-Instruct-LLM8bit-trust"
BETA = 1.0
TAU = 0.4
SCORE_SAVED = "./saved_scores"
PUSH_TO_HUB = True
REPO_ID = "Amadeus99/Qwen2.5-7B-Instruct-LLM8bit-trust"
# ---------------------

def process_save_hybrid_model():
    # Step 1: Load the Model in FP16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Step 2: Generate Safety Masks
    scorer = TrustScoring(MODEL_PATH, beta=BETA, tau=TAU)
    all_layers_scores = scorer.calculate_trustworthinesscore(cache_dir=SCORE_SAVED)
    threshold = scorer.get_threshold_score(all_layers_scores)
    safety_masks = scorer.get_safety_masks(all_layers_scores, threshold)

    # Step 3: Apply the Split
    model = apply_safety_split_to_model(model, safety_masks)

    # Step 4: Save the Hybrid Model
    save_hybrid_model(model, tokenizer, OUTPUT_DIR, None, PUSH_TO_HUB, REPO_ID)