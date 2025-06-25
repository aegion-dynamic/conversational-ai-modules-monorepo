# Conversation-ai-modules

## Overview
This monorepo is used to hold all the python modules used in the conversational chatbots. By consolidating all these projects under a single repository, we hope to reduce version conflicts and better manage the codebase.

## Code guidelines:
- Use **Poetry** for dependency management. **uv** can be used as a faster alternative for installations while keeping Poetry for dependency updates. If Poetry is unable to add your dependency, this means that your project will not work with the rest of packages. Poetry is meant to solve problems for you after you finish fighting with it.
- Add type annotations for all functions arguments and returns
- Add docstrings for all functions in Google format (use the plugin - `njpwerner.autodocstring`)
- Do not commit log files into the source tree and watch out what you commit, source tree should only include documentation, code and absolutely necessary resources that will prevent the code from running
- Linearize algorithm code into a `workflow` method where each step happens sequentially.
- The way to write algorithms is to : 1) Write the overall agorithm in comments 2) write down each of the steps 3) write the code (keep updating comments if you change the way you do things)
- Name variables properly so that we can keep track of whats going on in the algorithm
- Try to name functions as `<verb>_<subject>` as much as possible
- Name all the files in snake case
- Name all classes CamelCase with captials starts
- Avoid having `global` variables for libaries. Only applications are allowed to have them

## Quick Start

### Option 1: Traditional Poetry Setup
```powershell
# Install dependencies with Poetry
poetry install
poetry shell
```

### Option 2: Fast Setup with uv (Recommended)
```powershell
# For Windows (PowerShell)
.\setup_uv.ps1

# Or using Python script (cross-platform)
python setup_uv.py
```

### Hybrid Workflow (Recommended)
```powershell
# Add new dependencies with Poetry (maintains lock file)
poetry add <package-name>

# Fast installation/sync with uv
uv pip install -r requirements.txt

# Create virtual environment with uv (much faster)
uv venv
```

## Exporting the Environemtn variables for the backend
```sh
set -o allexport; source .env; set +o allexport
```

## Dependency Management Migration Guide

### Why uv?
- **10-100x faster** than Poetry for installations
- **Drop-in replacement** for pip with better dependency resolution
- **Compatible** with existing Poetry workflows
- **Better performance** for large monorepos like this one

### Migration Strategy
We're using a **gradual transition approach**:

1. **Keep Poetry** for dependency management (adding/removing packages)
2. **Use uv** for fast installations and virtual environments
3. **Maintain compatibility** with existing workflows

### Quick Commands
```powershell
# Windows users
.\run.bat help                    # Show all available commands
.\run.bat setup-uv               # One-time setup
.\run.bat install                # Smart install (uv if available, Poetry fallback)
.\run.bat sync                   # Sync uv with Poetry changes

# Cross-platform
python setup_uv.py              # One-time setup
make help                        # Show all available commands (Unix-like)
```

### Daily Workflow
```powershell
# Adding a new dependency (use Poetry to maintain lock file)
poetry add requests

# Sync the change to uv
.\run.bat sync

# Fast installation for other team members
.\run.bat install-uv

# Run tests (automatically uses uv if available)
.\run.bat test
```
