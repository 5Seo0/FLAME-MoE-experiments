import torch
from pathlib import Path
from typing import Tuple, List

# Refactored from empirical_analysis/work/expert_specialization.py
def load_expert_data_local(sample_path: Path, active_path: Path, expert_index: int, topk: int = 6):
    """
    Simplified version of job_step1 for interactive notebooks.
    Runs directly in the notebook process (no complex multiprocessing needed for small batches).
    """
    print(f"Loading sample: {sample_path.name} for Expert {expert_index}")
    
    # Load data (CPU to save GPU memory for other tasks)
    sample = torch.load(sample_path, map_location="cpu")
    active = torch.load(active_path, map_location="cpu")
    
    # Logic from original script
    sample_count = torch.bincount(sample, minlength=50277) # 50277 is FLAME vocabulary size
    
    # Filter for specific expert
    expert_indices = active.topk(k=topk).indices
    mask = (expert_indices == expert_index)
    
    # Get tokens that activated this expert
    activated_tokens = sample[mask.any(dim=1)]
    active_count = torch.bincount(activated_tokens, minlength=50277)
    
    return sample_count, active_count

def get_top_specialized_tokens(tokenizer, active_count, sample_count, top_n=10):
    """
    New helper function for students to visualize results immediately.
    """
    # Avoid division by zero
    safe_sample_count = sample_count.clone()
    safe_sample_count[safe_sample_count == 0] = 1
    
    # Calculate specialization ratio
    ratio = active_count.float() / safe_sample_count.float()
    
    # Get top tokens
    top_indices = torch.argsort(ratio, descending=True)[:top_n]
    
    results = []
    for idx in top_indices:
        token_str = tokenizer.decode([idx.item()])
        results.append((token_str, ratio[idx].item(), active_count[idx].item()))
        
    return results