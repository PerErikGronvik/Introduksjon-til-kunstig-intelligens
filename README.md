# Content of this file:

## Multiple choice quiz for exam.

#Multiple Labs

1. Lab 1: Introduction to Python Labs/Lab1.ipynb
2. Lab 2: Data Analysis with Pandas Labs/Lab2.ipynb
3. Lab 3: Machine Learning Basics Labs/Lab3.ipynb
4. Lab 4: Model Evaluation and Selection Labs/Lab4.ipynb

# Project 1: Data Analysis and Visualization oblig1_2025
oblig1_2025/MA1.ipynb
# Project 2: Passenger Prediction with Machine Learning oblig2_2025
oblig2_2025/'Mandatory Assignment 2.ipynb'
# Project 3: Vibe coding
oblig3_2025/'README.md'







# Project Setup Guide - FOOLPROOF INSTRUCTIONS

## IMPORTANT: Navigation First!

**BEFORE YOU DO ANYTHING ELSE**, you must navigate to the specific project folder:

### For oblig1_2025:
```bash
# Windows PowerShell:
cd "C:\Users\YourUsername\Documents\Github\Dave3625-Intro-ki\oblig1_2025"

# Or if you're already in the main repo folder:
cd oblig1_2025
```

### For oblig2_2025:
```bash
# Windows PowerShell:
cd "C:\Users\YourUsername\Documents\Github\Dave3625-Intro-ki\oblig2_2025"

# Or if you're already in the main repo folder:
cd oblig2_2025
```

**⚠️ CRITICAL:** You MUST be in the project folder (oblig1_2025 or oblig2_2025) for ALL the commands below to work!

## What is uv?

uv is a fast Python package and project manager written in Rust. It's designed to be a drop-in replacement for pip and pip-tools, but much faster. Think of it as a modern alternative to pip that handles both package installation and virtual environment management.

Key benefits of uv:
- Much faster than pip (10-100x faster in many cases)
- Better dependency resolution
- Built-in virtual environment management
- Compatible with existing Python packaging standards

## Setting up the Virtual Environment

### STEP 1: Install uv (ONLY if not already installed)
```bash
# Check if uv is installed first:
uv --version

# If you get an error, install uv:
pip install uv

# Alternative for macOS users with Homebrew:
# brew install uv
```

### STEP 2: Navigate to YOUR PROJECT FOLDER (CRITICAL!)
```bash
# Navigate to the project you want to work on:
# For oblig1_2025:
cd oblig1_2025


# Verify you're in the right place - you should see pyproject.toml:
ls *.toml
# This should show: pyproject.toml
```

### STEP 3: Install ALL Dependencies (One Command!)
```bash
# This installs everything you need automatically if:
uv sync

# If you get an error "No pyproject.toml found", you're in the WRONG FOLDER!
# Go back to STEP 2 and navigate properly!
```

### STEP 4: Set Up Jupyter Kernel (For VS Code)
```bash
# Register your project as a Jupyter kernel:
uv run python -m ipykernel install --user --name PROJECT_NAME --display-name "Python (PROJECT_NAME)"

# Replace PROJECT_NAME with oblig1_venv or oblig2_venv
# Example for oblig2:
uv run python -m ipykernel install --user --name oblig2_venv --display-name "Python (oblig2_venv)"
```

## Using Jupyter in VS Code (EASIEST METHOD)

### STEP 5: Open Notebook in VS Code
1. **Open VS Code**
2. **Navigate to your project folder** (oblig1_2025 or oblig2_2025)
3. **Open any .ipynb file** (or create a new one)
4. **VS Code will ask you to select a kernel** - this is NORMAL!

### STEP 6: Select the Correct Kernel
1. **Click on "Select Kernel"** (top-right corner of notebook)
2. **Look for your project kernel**: "Python (oblig1_venv)" or "Python (oblig2_venv)"
3. **If you don't see it:**
   - Click "Select Another Kernel"
   - Choose "Python Environments" 
   - Browse to: `YourProject\.venv\Scripts\python.exe`
   - Example: `oblig2_2025\.venv\Scripts\python.exe`

### STEP 7: Test Everything Works
Run this in a notebook cell:
```python
import sys
print(f"Python path: {sys.executable}")
print("✅ Jupyter is working!")

# Test if packages are available
import pandas as pd
import numpy as np
print("✅ Data science packages loaded!")
```

## If VS Code Can't Find Your Kernel (TROUBLESHOOTING)

### Option A: Force Kernel Registration
```bash
# Make sure you're in your project folder first!
cd oblig2_2025  # or oblig1_2025

# Register the kernel manually:
uv run python -m ipykernel install --user --name oblig2_venv --display-name "Python (oblig2_venv)"

# Restart VS Code completely
# Open your notebook again - kernel should now appear
```

### Option B: Manual Python Path Selection
1. In VS Code, open Command Palette (`Ctrl+Shift+P`)
2. Type: "Python: Select Interpreter"
3. Browse to your project's Python:
   - `C:\Users\YourUsername\Documents\Github\Dave3625-Intro-ki\oblig2_2025\.venv\Scripts\python.exe`
4. Open your notebook - it should use this interpreter

## Required Data Files

**IMPORTANT:** Some data files are not included in this repository due to copyright restrictions.

For **oblig2_2025** project:
- Download `Ruter-data.csv` from Canvas
- Place it in the `oblig2_2025` folder
- The file is required for the passenger prediction assignment

## CRITICAL TIPS (READ THIS!)

### ⚠️ MOST COMMON MISTAKES:
1. **NOT navigating to project folder first** - ALL commands will fail!
2. **Using pip instead of uv** - packages won't install in the right place
3. **Wrong kernel in VS Code** - imports will fail even if packages are installed
4. **Deleting .venv folder** - you'll lose your entire environment!

### ✅ GOLDEN RULES:
1. **ALWAYS navigate to project folder first**: `cd oblig1_2025` or `cd oblig2_2025`
2. **Use `uv sync` to install everything** - it's automatic and fast
3. **In VS Code, select the project kernel**: "Python (oblig1_venv)" or "Python (oblig2_venv)"
4. **If kernel missing, run**: `uv run python -m ipykernel install --user --name PROJECT_venv`
5. **Never delete the .venv folder** - it contains your Python environment!

## Quick Reference Commands (AFTER navigating to project folder!)

### Essential Commands (Use These):
```bash
# 1. Navigate to your project (FIRST!)
cd oblig1_2025  # or cd oblig2_2025

# 2. Install everything automatically
uv sync

# 3. Register Jupyter kernel
uv run python -m ipykernel install --user --name oblig2_venv --display-name "Python (oblig2_venv)"

# 4. Add new packages if needed
uv add package-name

# 5. Run Python commands through uv
uv run python your_script.py
```

### Advanced Commands (If Needed):
```bash
# Check what's installed
uv run python -m pip list

# Remove package
uv remove package-name

# Get Python path (for VS Code kernel selection)
uv run python -c "import sys; print(sys.executable)"

# Activate venv manually (usually not needed with uv)
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # macOS/Linux
```

### Emergency Reset (If Everything Breaks):
```bash
# Navigate to project folder
cd oblig2_2025

# Delete virtual environment
rm -rf .venv  # or rmdir /s .venv on Windows

# Recreate everything
uv sync
uv run python -m ipykernel install --user --name oblig2_venv --display-name "Python (oblig2_venv)"

# Restart VS Code
```