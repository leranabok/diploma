______________________________________________________________________
<div align="center">

# 🤖 Code for diploma project


______________________________________________________________________
[![ci-testing](https://github.com/leranabok/diploma/actions/workflows/ci-testing.yml/badge.svg?branch=main&event=push)](https://github.com/leranabok/diploma/actions/workflows/ci-testing.yml)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/leranabok/diploma/blob/master/LICENSE)

</div>

## 💎 Installation with `pip`

Installation is as simple as running:

```bash
pip install git+https://github.com/leranabok/diploma.git
```

## ⚙️ Setup

**Before setting up a project,** make sure that the `setup.py` script has all setup parameters adjusted and change an `ai_awesome` directory name to your project name. Also, if you want to use conda environment, you may want to change its name in a `conda_env.yml` file. Then, follow the instruction:

```bash
# Clone repo
git clone https://github.com/leranabok/diploma.git

# Go to repo directory
cd diploma

# (Optional) Create virtual environment
python -m venv venv
source ./venv/bin/activate

# Install project in editable mode
pip install -e .[dev]

# (Optional but recommended) Install pre-commit hooks to preserve code format consistency
pre-commit install
```

## 🐍 Setup with Anaconda or Miniconda

**Before setting up a project,** make sure that the `setup.py` script has all setup parameters adjusted and change an `ai_awesome` directory name to your project name. Additionally, you may want to change a default (_ai_awesome_env_) conda environment name in a `conda_env.yml` file. Then, follow the instruction:

```bash
# Clone repo
git clone https://github.com/leranabok/diploma.git

# Go to repo directory
cd diploma

# Create and activate conda environment
conda env create -f ./conda_env.yml
conda activate ai_awesome_env

# (Optional but recommended) Install pre-commit hooks to preserve code format consistency
pre-commit install
```

<div align="center">

### 🤗 Happy coding !

</div>
