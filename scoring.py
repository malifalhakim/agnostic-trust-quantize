import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_best_device, clear_memory, get_named_linears
from calib_data import get_general_dataset, get_fairness_dataset, get_safety_dataset

class TrustScoring:
    def __init__(self, model_path, beta, tau):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.beta = beta
        self.tau = tau
        self.named_linears = get_named_linears(self.model)

    def calculate_trustworthinesscore(self, cache_dir="/saved_scores"):
        """
        Calculate Trustworthiness Score for each weight in the model.
        Trustworthiness_score = Fairness_score + Safety_score
        """
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        path_nsp = os.path.join(cache_dir, "score_nsp.pt")
        path_fair = os.path.join(cache_dir, "score_fairness.pt")
        path_dolly = os.path.join(cache_dir, "score_dolly.pt")
        path_safe = os.path.join(cache_dir, "score_safety.pt")

        device = get_best_device()
        criterion = torch.nn.CrossEntropyLoss()

        # =========================================================================
        # PHASE 1: General Score (NSP)
        # =========================================================================
        if not os.path.exists(path_nsp):
            print(f"Phase 1/4: Calculating general score from wikipedia NSP data...")
            general_data_nsp = get_general_dataset(tokenizer=self.tokenizer)
            input_general_nsp = [item[0] for item in general_data_nsp]

            accumulated_scores = {
                name: torch.zeros_like(module.weight, device='cpu')
                for name, module in self.named_linears.items()
            }
            num_data = 0

            self.model.to(device)
            clear_memory()

            for i in tqdm(range(0, len(input_general_nsp)), desc="Calculating General Sensitivity"):
                input_gen = input_general_nsp[i].to(device)
                num_data += 1

                self.model.train()
                self.model.zero_grad()
                for module in self.named_linears.values():
                    module.weight.requires_grad = True
                
                outputs = self.model(input_gen)
                logits = outputs.logits[:, :-1, :]
                labels = input_gen[:, 1:]

                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss.backward()

                for name, module in self.named_linears.items():
                    if module.weight.grad is not None:
                        accumulated_scores[name] += module.weight.grad.detach().pow(2).cpu()
                        module.weight.grad = None
                    module.weight.requires_grad = False
                
                clear_memory()
            
            score_nsp = {name: acc_score / num_data for name, acc_score in accumulated_scores.items()}
            torch.save(score_nsp, path_nsp)
            del accumulated_scores, score_nsp, input_general_nsp
            clear_memory()
        else:
            print(f"Phase 1/4: NSP score found in cache. Skipping.")
        
        # =========================================================================
        # PHASE 2: Fairness Score
        # =========================================================================
        if not os.path.exists(path_fair):
            print(f"Phase 2/4: Calculating & Saving Fairness score...")
            fairness_data = get_fairness_dataset(tokenizer=self.tokenizer)
            input_stereotypes = [x[0] for x in fairness_data]
            target_stereotypes = [x[1] for x in fairness_data]
            input_antistereotypes = [x[2] for x in fairness_data]
            target_antistereotypes = [x[3] for x in fairness_data]

            accumulated_scores = {name: torch.zeros_like(module.weight, device='cpu') for name, module in self.named_linears.items()}
            num_data = 0
            self.model.to(device)
            clear_memory()

            for i in tqdm(range(len(input_stereotypes)), desc="Calc Fairness Sensitivity"):
                inp_s = input_stereotypes[i].to(device)
                tar_s = target_stereotypes[i].to(device)
                inp_a = input_antistereotypes[i].to(device)
                tar_a = target_antistereotypes[i].to(device)
                
                num_data += 1
                self.model.train()
                self.model.zero_grad()
                for module in self.named_linears.values(): module.weight.requires_grad = True

                # Stereo Forward
                out_s = self.model(inp_s).logits[:, :-1, :]
                nll_s = criterion(out_s.reshape(-1, out_s.size(-1)), tar_s[:, 1:].reshape(-1))

                # Anti-Stereo Forward
                out_a = self.model(inp_a).logits[:, :-1, :]
                nll_a = criterion(out_a.reshape(-1, out_a.size(-1)), tar_a[:, 1:].reshape(-1))

                loss_fair = torch.abs(nll_s - nll_a)
                loss_fair.backward()

                for name, module in self.named_linears.items():
                    if module.weight.grad is not None:
                        accumulated_scores[name] += module.weight.grad.detach().pow(2).cpu()
                        module.weight.grad = None
                    module.weight.requires_grad = False

            score_fair = {name: acc / num_data for name, acc in accumulated_scores.items()}
            torch.save(score_fair, path_fair)
            del accumulated_scores, score_fair, input_stereotypes, input_antistereotypes
            clear_memory()
        else:
            print("Phase 2/4: Fairness score found in cache. Skipping.")


        # =========================================================================
        # PHASE 3: General Score (Dolly)
        # =========================================================================
        if not os.path.exists(path_dolly):
            print(f"Phase 3/4: Calculating & Saving Dolly score...")
            general_data_dolly = get_general_dataset(
                dataset_name="databricks/databricks-dolly-15k",
                subset=None, split="train", use_template=True,
                text_column="response", prompt_column="instruction", tokenizer=self.tokenizer
            )
            
            input_dolly = [x[0] for x in general_data_dolly]
            target_dolly = [x[1] for x in general_data_dolly]

            accumulated_scores = {name: torch.zeros_like(module.weight, device='cpu') for name, module in self.named_linears.items()}
            num_data = 0
            self.model.to(device)
            clear_memory()

            for i in tqdm(range(len(input_dolly)), desc="Calc Dolly Sensitivity"):
                inp = input_dolly[i].to(device)
                tar = target_dolly[i].to(device)
                
                num_data += 1
                self.model.train()
                self.model.zero_grad()
                for module in self.named_linears.values(): module.weight.requires_grad = True

                out = self.model(inp).logits[:, :-1, :]
                loss = criterion(out.reshape(-1, out.size(-1)), tar[:, 1:].reshape(-1))
                loss.backward()

                for name, module in self.named_linears.items():
                    if module.weight.grad is not None:
                        accumulated_scores[name] += module.weight.grad.detach().pow(2).cpu()
                        module.weight.grad = None
                    module.weight.requires_grad = False
            
            score_dolly = {name: acc / num_data for name, acc in accumulated_scores.items()}
            torch.save(score_dolly, path_dolly)
            del accumulated_scores, score_dolly, input_dolly
            clear_memory()
        else:
            print("Phase 3/4: Dolly score found in cache. Skipping.")

        # =========================================================================
        # PHASE 4: Safety Score
        # =========================================================================
        if not os.path.exists(path_safe):
            print(f"Phase 4/4: Calculating & Saving Safety score...")
            safety_data = get_safety_dataset(tokenizer=self.tokenizer)
            input_safe = [x[0] for x in safety_data]
            target_safe = [x[1] for x in safety_data]

            accumulated_scores = {name: torch.zeros_like(module.weight, device='cpu') for name, module in self.named_linears.items()}
            num_data = 0
            self.model.to(device)
            clear_memory()

            for i in tqdm(range(len(input_safe)), desc="Calc Safety Sensitivity"):
                inp = input_safe[i].to(device)
                tar = target_safe[i].to(device)
                
                num_data += 1
                self.model.train()
                self.model.zero_grad()
                for module in self.named_linears.values(): module.weight.requires_grad = True

                out = self.model(inp).logits[:, :-1, :]
                loss = criterion(out.reshape(-1, out.size(-1)), tar[:, 1:].reshape(-1))
                loss.backward()

                for name, module in self.named_linears.items():
                    if module.weight.grad is not None:
                        accumulated_scores[name] += module.weight.grad.detach().pow(2).cpu()
                        module.weight.grad = None
                    module.weight.requires_grad = False

            score_safe = {name: acc / num_data for name, acc in accumulated_scores.items()}
            torch.save(score_safe, path_safe)
            del accumulated_scores, score_safe, input_safe
            clear_memory()
        else:
            print("Phase 4/4: Safety score found in cache. Skipping.")
        
        # =========================================================================
        # PHASE 5: ASSEMBLY
        # =========================================================================
        print("Assembling final scores ...")
        
        # 1. Start with Fairness
        final_scores = torch.load(path_fair)
        
        # 2. Add Safety
        temp_scores = torch.load(path_safe)
        for name in final_scores:
            final_scores[name] += temp_scores[name]
        del temp_scores # Free RAM immediately
        
        # 3. Subtract Beta * NSP
        temp_scores = torch.load(path_nsp)
        for name in final_scores:
            final_scores[name] -= (self.beta * temp_scores[name])
        del temp_scores # Free RAM
        
        # 4. Subtract Beta * Dolly
        temp_scores = torch.load(path_dolly)
        for name in final_scores:
            final_scores[name] -= (self.beta * temp_scores[name])
        del temp_scores # Free RAM

        return final_scores

    def analyze_scores(self, scores):
        """
        Memory-efficient analysis of critical score distribution
        """
        print("\n=== Critical Score Analysis ===")

        try:
            total_weights = sum(score_tensor.numel() for score_tensor in scores.values())
            print(f"Total weights: {total_weights:,}")
            
            global_min = float('inf')
            global_max = float('-inf')

            for score_tensor in scores.values():
                global_min = min(global_min, score_tensor.min().item())
                global_max = max(global_max, score_tensor.max().item())
            
            print(f"Min score: {global_min:.6f}")
            print(f"Max score: {global_max:.6f}")
            
            running_sum = 0.0
            for score_tensor in scores.values():
                running_sum += score_tensor.sum().item()
            mean_score = running_sum / total_weights
            print(f"Mean score: {mean_score:.6f}")
            
            zero_count = 0
            near_zero_count = 0
            for score_tensor in scores.values():
                zero_count += (score_tensor == 0).sum().item()
                near_zero_count += (score_tensor.abs() < 1e-6).sum().item()
            
            print(f"Zero values: {zero_count:,} ({100*zero_count/total_weights:.2f}%)")
            print(f"Near-zero (<1e-6): {near_zero_count:,} ({100*near_zero_count/total_weights:.2f}%)")
            
            print("\n=== Creating Distribution Plot ===")
            sample_size = min(10_000_000, total_weights)
            print(f"Sampling {sample_size:,} weights for visualization...")
            
            sampled_scores = []
            seen = 0

            for score_tensor in scores.values():
                scores_flat = score_tensor.view(-1)
                layer_size = scores_flat.numel()
                
                if seen + layer_size <= sample_size:
                    sampled_scores.append(scores_flat.cpu())
                    seen += layer_size
                else:
                    remaining = sample_size - seen
                    if remaining > 0:
                        indices = torch.randperm(layer_size)[:remaining]
                        sampled_scores.append(scores_flat[indices].cpu())
                        seen = sample_size
                        break
            
            sampled_scores = torch.cat(sampled_scores)
            
            print(f"Computing percentiles on {sampled_scores.numel():,} samples...")
            percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]
            for p in percentiles:
                val = torch.quantile(sampled_scores.float(), p/100)
                print(f"{p}th percentile: {val.item():.6f}")
            
            print("\n=== Top 10 Layers by Mean Score ===")
            layer_stats = []
            for name, score_tensor in scores.items():
                layer_stats.append({
                    'name': name,
                    'max': score_tensor.max().item(),
                    'mean': score_tensor.mean().item(),
                })
            
            layer_stats.sort(key=lambda x: x['mean'], reverse=True)
            for i, stat in enumerate(layer_stats[:10]):
                print(f"{i+1}. {stat['name']}")
                print(f"   Max: {stat['max']:.6f}, Mean: {stat['mean']:.6f}")
            
        
            del sampled_scores
            clear_memory()
        except Exception as e:
            print(f"Error during score analysis: {e}")

        return torch.cat([score_tensor.view(-1) for score_tensor in scores.values()])

    def get_threshold_score(self, all_layers_scores):
        all_scores_tensor = self.analyze_scores(all_layers_scores)
        sample_size = min(1_000_000, all_scores_tensor.numel())
        indices = torch.randint(0, all_scores_tensor.numel(), (sample_size,))
        score_samples = all_scores_tensor.view(-1)[indices].cpu()

        print("Finding threshold via sampling and sorting...")
        tau = self.tau
        k = int(score_samples.numel() * tau)
        threshold = torch.topk(score_samples.float(), k, largest=True, sorted=False)[0].min()
        return threshold
        
    @staticmethod
    def apply_trust_preservation(orig_weight, noisy_weight, sensitivity_scores, threshold):
        """
        Restores original weights for the most critical weights.
        """
        mask = (sensitivity_scores >= threshold).float()

        final_weight = (orig_weight * mask) + (noisy_weight * (1 - mask))

        return final_weight