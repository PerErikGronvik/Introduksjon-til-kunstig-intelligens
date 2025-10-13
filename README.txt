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

### 2. Create and activate virtual environment
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

### 3. Install project dependencies
```bash
# Install dependencies from pyproject.toml
uv sync

# Or install specific packages
uv add package-name
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
2. Install jupyter in the virtual environment:
   ```bash
   uv add jupyter
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
   python -m ipykernel install --user --name=your-project-name
   ```
4. In Jupyter, select your custom kernel from the kernel menu

## Tips for Beginners

- Always activate your virtual environment before working on the project
- Use `uv add package-name` instead of `pip install package-name`
- The .venv folder contains your virtual environment - don't delete it!
- If you see import errors in Jupyter, check that you're using the correct kernel
- Run `uv sync` to install all project dependencies after cloning the repository

## Common Commands

```bash
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