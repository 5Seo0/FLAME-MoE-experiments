#!/bin/bash

# 1. Load System Modules (Specific to your MLTGPU-2 server)
# Based on MLTGPU-2 PDF, we know miniconda is available via modules
module load miniconda

# 2. Create Environment (Lightweight version)
# We remove the heavy compilation logic if possible, or ensure it runs on 1 core
conda create -n flame_student python=3.10 -y
source activate flame_student

# 3. Install PyTorch (Standard)
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install Dependencies without the heavy Apex/TransformerEngine build if strictly inference
# Note: FLAME-MoE code likely imports apex/TE. We will install pre-built if available or build locally.
pip install -r ../Megatron-LM/requirements/pytorch_24.10/requirements.txt
pip install transformers numpy pandas matplotlib seaborn jupyterlab

# 5. Handle the tricky submodules (Apex/TE)
# Strategy: Try to use standard library implementations where possible, 
# or compile them ONCE in a shared directory and PYTHONPATH them for students.
# For now, we let them compile but strictly locally (no sbatch):
echo "Compiling Apex (this may take a few minutes)..."
cd ../apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ../student_labs

echo "Environment 'flame_student' created. Run 'source activate flame_student' to use."