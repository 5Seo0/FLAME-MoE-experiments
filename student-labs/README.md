# Test 1: Analyzing Mixture-of-Experts (MoE) Routing

**Objective:** In this lab, you will run a state-of-the-art MoE model (`OLMoE-1B-7B`) and "spy" on its internal router. Your goal is to visualize how the model assigns different experts to different languages (English vs. Korean/Japanese).

**Hardware:** This lab is optimized for the **MLTGPU-2** student server (A30 GPU slices).

---

## Step 1: Environment Setup

Open a terminal on the server and run the following commands to create your research environment.

### 1. Load System Modules

We need the base Conda module provided by the university server.

```bash
module load miniconda
```

### 2. Create the Conda Environment

We will create an isolated environment named moe_lab with Python 3.10.

```bash
conda create -n moe_lab python=3.10 -y
source activate moe_lab
```

### 3. Install Required Libraries

```bash
# Install PyTorch (CUDA 12.1 compatible for A30 GPUs)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install Hugging Face and Analysis Tools
# We force an upgrade to ensure OLMoE model support
pip install -U transformers accelerate bitsandbytes numpy matplotlib seaborn jupyterlab ipywidgets
```

## Step 2: Launch Jupyter

**Start the Lab:** Run this command inside your moe_lab environment:

```bash
jupyter lab --no-browser --port=8888
```

**Connect:** Follow the standard course instructions to SSH tunnel the port to your laptop (e.g., ssh -L 8888:localhost:8888 student@server...).

**Open:** Open the link provided in the terminal (usually <http://localhost:8888/lab>...) in your browser.

## Step 3: The Analysis Code

Create a new notebook (e.g., 01_Router_Analysis.ipynb) and paste the following code blocks.

Block 1: Load the Model (Quantized)
We use 4-bit quantization to fit the 7B parameter model into your 12GB GPU slice.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define the model (AllenAI OLMoE-1B-7B)
MODEL_ID = "AllenAI/OLMoE-1B-7B-0924"

# Configure 4-bit quantization to save memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto"
)
print(f"Success! Model loaded on {model.device}")
```

**Block 2: Attach the "Router Spy"**
This code attaches a hook to the model to capture the routing decisions during inference.

```python
# Dictionary to store the router outputs
router_logs = {}

def get_router_hook(layer_name):
    def hook(module, inputs, outputs):
        # Robustly handle tuple vs tensor outputs
        # OLMoE routers often return a tuple where [1] is the logits
        if isinstance(outputs, tuple):
            val = outputs[1] 
        else:
            val = outputs
        
        # Only capture if it matches the expert count (64 for OLMoE)
        if isinstance(val, torch.Tensor) and val.shape[-1] == 64:
            # Detach and move to CPU to save GPU memory
            router_logs[layer_name] = val.detach().float().cpu()
    return hook

# Clear old hooks (if re-running)
for name, module in model.named_modules():
    if hasattr(module, "_forward_hooks"):
        module._forward_hooks.clear()

# Attach new hooks to the router layers
print("Attaching spy hooks...")
hook_count = 0
for name, module in model.named_modules():
    # We target the specific gate layer in the OLMoE architecture
    if name.endswith("mlp.gate"): 
        module.register_forward_hook(get_router_hook(name))
        hook_count += 1

print(f"Attached {hook_count} hooks (Expect 32 for this model).")
```

**Block 3: Run & Visualize**
Run this block to see the expert specialization in action.

```python
# Run a sample inference
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10)

# Visualize the routing
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the routing distribution for each hook
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define a Multilingual Test Sentence
text = "The quick brown fox jumps. 안녕하세요. こんにちは."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 2. Run Inference
router_logs = {} # Clear previous logs
with torch.no_grad():
    outputs = model(**inputs)

# 3. Analyze a Deep Layer (Layer 14)
# Deep layers often show the strongest specialization
target_layer = "model.layers.14.mlp.gate"

if target_layer in router_logs:
    logits = router_logs[target_layer].numpy()
    
    # Handle potential shape differences (Batch vs No-Batch)
    if logits.ndim == 3:
        data = logits[0] # [Seq, Experts]
    else:
        data = logits

    # Get the text labels
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    token_labels = [tokenizer.convert_tokens_to_string([t]).strip() for t in tokens]
    
    # --- Visualization ---
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        data, 
        cmap="viridis", 
        yticklabels=token_labels,
        cbar_kws={'label': 'Router Score'}
    )
    plt.title(f"Linguistic Specialization in {target_layer}")
    plt.xlabel("Expert ID (0-63)")
    plt.ylabel("Token Sequence")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Print Top Experts
    print(f"{'TOKEN':<15} | {'TOP EXPERT'}")
    print("-" * 30)
    for t, expert_id in zip(token_labels, np.argmax(data, axis=-1)):
        print(f"{t:<15} | {expert_id}")

else:
    print(f"Layer {target_layer} not found. Captured: {list(router_logs.keys())[:3]}...")
```

**Next Steps for examination:**

**Scale Up:** Wrap Block 3 in a loop that feeds 1,000 sentences from your dataset.

**Count:** Instead of plotting a heatmap, count how many times Expert X is chosen for Language Y.

**Compare:** Does Korean use a smaller, more specific set of experts than English?
