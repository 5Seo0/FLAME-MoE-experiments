# 🚀 Quick Start: FLAME-MoE Linguistic Analysis

**Goal:** Analyze how a Mixture-of-Experts (MoE) model "routes" different languages.
**Task:** You will use a small, efficient model (FLAME-MoE-115M) to see if it sends Korean, Japanese, or English tokens to different experts.
**Hardware:** Adapted for the MLTGPU-2 Student Server (Single GPU).

---

## 1. Environment Setup

Open a terminal on the server and run these commands one by one to create your workspace.

```bash
# 1. Load the system Conda module
module load miniconda

# 2. Create a fresh environment named 'moe_lab'
conda create -n moe_lab python=3.10 -y

# 3. Activate the environment
source activate moe_lab

# 4. Install required AI and Plotting libraries
# (We use cu121 for the A30 GPUs on the student server)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers accelerate matplotlib seaborn jupyterlab
```

## 2. Start Your Notebook

1. In your terminal (inside the moe_lab environment), start Jupyter:

    ```bash
    jupyter lab --no-browser --port=8888
    ```

2. Follow the SSH tunneling instructions provided by the course to open the link in your browser.

3. Create a new Python 3 (ipykernel) notebook.

## 3. The "Spy" Code (Copy into Cell 1)

This code loads the model and attaches a "hook" to the router. This allows us to see which expert the model chooses for every single token.

    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- CONFIGURATION ---
    # We use the 115M version. It is small enough to fit on one MIG slice (12GB).
    MODEL_ID = "CMU-FLAME/FLAME-MoE-115M-459M"

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # trust_remote_code=True is REQUIRED because FLAME uses custom architecture code
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    print("Model loaded successfully!")

    # --- ROUTER SPY SETUP ---
    # This dictionary will store the router choices for the most recent pass
    router_logs = {}

    def get_router_hook(layer_name):
        """Creates a hook that saves the router's choices (logits) to our log."""
        def hook(module, inputs, outputs):
            # FLAME outputs are usually a tuple. The router logits are often the 2nd item.
            # Shape is typically [batch_size, seq_len, num_experts]
            if isinstance(outputs, tuple) and len(outputs) > 1:
                # We detach() and move to CPU immediately to save GPU memory
                router_logs[layer_name] = outputs[1].detach().cpu()
        return hook

    # Attach our spy hook to every MoE layer in the model
    print("Attaching hooks...")
    for name, module in model.named_modules():
        # We look for modules with 'moe' and 'gate' in their name
        if "moe" in name and "gate" in name:
            print(f" -> Hooked: {name}")
            module.register_forward_hook(get_router_hook(name))

    print("Ready to analyze!")
    ```

## 4. Run an Experiment (Copy into Cell 2)

Run this cell to test if different languages trigger different experts.

```python
    # 1. Define a multilingual test sentence
    # Feel free to change this to your target languages!
    text = "The quick brown fox. 안녕하세요. こんにちは." 

    # 2. Tokenize and Run Inference
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Clear old logs before running
    router_logs = {} 

    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Analyze the Results
    # Let's look at a middle layer (e.g., layer 5 or 6)
    layer_to_analyze = list(router_logs.keys())[5] 
    logits = router_logs[layer_to_analyze].float() # Shape: [1, seq_len, 64]

    # Find the "Top-1" expert for each token (the one with the highest score)
    top_experts = torch.argmax(logits, dim=-1).squeeze() 
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    # 4. Print Table
    print(f"\n--- Routing Pattern for {layer_to_analyze} ---")
    print(f"{'TOKEN':<15} | {'CHOSEN EXPERT'}")
    print("-" * 35)

    for t, expert in zip(tokens, top_experts):
        # Decode special characters for cleaner printing
        readable_token = tokenizer.convert_tokens_to_string([t]).strip()
        print(f"{readable_token:<15} | Expert {expert.item()}")

    # 5. Visualization (Heatmap)
    plt.figure(figsize=(12, 5))
    # We only plot the first 20 tokens to keep it readable
    sns.heatmap(logits[0, :20, :].numpy(), cmap="viridis", cbar_kws={'label': 'Router Activation Score'})
    plt.title(f"Expert Activation: {layer_to_analyze}")
    plt.xlabel("Expert ID (0-63)")
    plt.ylabel("Token Position")
    plt.tight_layout()
    plt.show()
```

### Student Tips

- `trust_remote_code=True`: You will see a warning about this. It is normal. FLAME is a research model and requires custom code to run the "Expert" routing logic.
- **Memory Management**: If you run out of GPU memory (CUDA Out of Memory), try restarting the notebook kernel. The router_logs can get big if you feed it huge books, so analyze one batch of sentences at a time.
- **Your Thesis Task**: Your goal is to replace the single text string in Step 4 with a loop that processes thousands of English vs. Korean sentences. You will then count: "Did Expert 5 activate 90% of the time for Korean, but only 10% for English?"
