# Project Setup Guide

## What is uv?

uv is a fast Python package and project manager written in Rust. It's designed to be a drop-in replacement for pip and pip-tools, but much faster. Think of it as a modern alternative to pip that handles both package installation and virtual environment management.

Key benefits of uv:
- Much faster than pip (10-100x faster in many cases)
- Better dependency resolution
- Built-in virtual environment management
- Compatible with existing Python packaging standards

## Setting up the Virtual Environment

### 1. Install uv (if not already installed)
```bash
# Install uv (works on both Windows and macOS)
pip install uv

# Alternative for macOS users with Homebrew:
# brew install uv
```

### 2. No pyproject.toml. Initialize uv project (creates pyproject.toml)
```bash
# Initialize a new uv project (this creates pyproject.toml)
uv init

# This creates a pyproject.toml file needed for package management
```


### 3. Create and activate virtual environment
```bash
# Create a new virtual environment (same on both systems)
uv venv

# Activate the virtual environment:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate

# Note: You should see (.venv) at the beginning of your terminal prompt when activated
```

### 4. Install project dependencies (AFTER creating and activating venv)
```bash
# Make sure your virtual environment is activated first!
# You should see (.venv) at the beginning of your terminal prompt

# Install dependencies from pyproject.toml (if it exists)
uv sync

# Or install specific packages for data science projects
uv add jupyterlab ipykernel
```

## Using Virtual Environment in Jupyter Notebooks

### Method 1: VS Code (Recommended)
1. Open VS Code
2. Open a .ipynb file
3. Click on the kernel selector (top right of notebook)
4. Choose "Select Another Kernel"
5. Select "Python Environments"
4. Choose the Python interpreter from your .venv folder:
   - Windows: `.venv\Scripts\python.exe`
   - macOS/Linux: `.venv/bin/python`

### Method 2: Command Line
1. Activate your virtual environment (see step 2 above)
2. Install jupyter and data science packages in the virtual environment:
   ```bash
   uv add jupyterlab ipykernel pandas numpy matplotlib scipy seaborn
   ```
3. Start Jupyter:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

### Method 3: Install Kernel Manually
1. Activate your virtual environment
2. Install ipykernel:
   ```bash
   uv add ipykernel
   ```
3. Register the kernel:
   ```bash
   python -m ipykernel install --user --name=your-project-name --display-name="Python (your-project-name)"
   ```
4. In Jupyter, select your custom kernel from the kernel menu

## Required Data Files

**IMPORTANT:** Some data files are not included in this repository due to copyright restrictions.

For **oblig2_2025** project:
- Download `Ruter-data.csv` from Canvas
- Place it in the `oblig2_2025` folder
- The file is required for the passenger prediction assignment

## Tips for Beginners

- **IMPORTANT:** Always create and activate your virtual environment BEFORE installing packages
- Always activate your virtual environment before working on the project
- Use `uv add package-name` instead of `pip install package-name`
- The .venv folder contains your virtual environment - don't delete it!
- If you see import errors in Jupyter, check that you're using the correct kernel
- Run `uv sync` to install all project dependencies after cloning the repository

## Common Commands

```bash
# Initialize uv project (creates pyproject.toml)
uv init

# Create virtual environment (same on both systems)
uv venv

# Activate virtual environment
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Install dependencies (same on both systems)
uv sync

# Add new package (same on both systems)
uv add package-name

# Remove package (same on both systems)
uv remove package-name

# Show installed packages (same on both systems)
uv pip list

# Deactivate virtual environment (same on both systems)
deactivate
```