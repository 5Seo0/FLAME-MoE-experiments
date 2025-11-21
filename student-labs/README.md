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
