# Reinforcement learning and canonical forms in mathematics

This repository contains materials for the "Reinforcement Learning and Canonical Forms in Mathematics" team program at the 2025 KIAS Winter School on Mathematics and AI.


## Getting Started (Environment Setup)

If you are not familiar with Python or local environment setup, you can run most of the `.py` files in this repository directly on Google Colab (e.g., by uploading the script or opening it from GitHub in Colab) without installing anything on your own machine.

For those who prefer a local setup, the instructions below assume a Linux environment (Debian/Ubuntu family). On Windows, open a WSL2 terminal and run the same commands.

0. Prepare a working directory and move into it:
```bash
mkdir rl_canonical_forms
cd rl_canonical_forms
```

1. Clone this repository (HTTPS) into the current directory:
```bash
git clone https://github.com/chlee-0/RL_canonical_forms_math .
```

2. Install the Python virtual environment package (e.g., on Debian/Ubuntu):
```bash
sudo apt install python3.12-venv
```

3. Create a Python virtual environment and activate it (recommended). For example, in your working directory:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install Python dependencies from this repository:
```bash
pip install -r requirements.txt
