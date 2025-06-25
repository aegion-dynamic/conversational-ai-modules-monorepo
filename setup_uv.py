#!/usr/bin/env python3
"""
Setup script for transitioning from Poetry to uv
Provides commands for both Poetry and uv usage
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_tool_installed(tool_name):
    """Check if a tool is installed"""
    try:
        subprocess.run([tool_name, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_uv():
    """Install uv if not already installed"""
    if check_tool_installed("uv"):
        print("✅ uv is already installed")
        return True
    
    print("Installing uv...")
    if sys.platform.startswith("win"):
        # Windows installation
        cmd = ["pip", "install", "uv"]
    else:
        # Unix-like systems
        cmd = ["pip", "install", "uv"]
    
    return run_command(cmd, "uv installation")

def sync_with_poetry():
    """Sync uv with Poetry's lock file"""
    if not Path("poetry.lock").exists():
        print("❌ poetry.lock not found. Run 'poetry install' first.")
        return False
    
    print("Syncing uv with Poetry lock file...")
    # Export Poetry dependencies to requirements format for uv
    export_cmd = ["poetry", "export", "-f", "requirements.txt", "--output", "requirements.txt", "--with", "dev"]
    if run_command(export_cmd, "Export Poetry dependencies"):
        # Install using uv
        install_cmd = ["uv", "pip", "install", "-r", "requirements.txt"]
        return run_command(install_cmd, "Install dependencies with uv")
    return False

def main():
    """Main setup function"""
    print("🚀 Setting up uv alongside Poetry for faster dependency management")
    print("="*60)
    
    # Check Poetry installation
    if not check_tool_installed("poetry"):
        print("❌ Poetry not found. Please install Poetry first.")
        sys.exit(1)
    
    print("✅ Poetry is installed")
    
    # Install uv
    if not install_uv():
        print("❌ Failed to install uv")
        sys.exit(1)
    
    # Sync with Poetry
    if not sync_with_poetry():
        print("❌ Failed to sync with Poetry")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nUsage recommendations:")
    print("- Use Poetry for adding/removing dependencies: poetry add <package>")
    print("- Use uv for fast installations: uv pip install <package>")
    print("- Use uv for virtual environments: uv venv")
    print("- To sync uv with Poetry changes, run this script again")

if __name__ == "__main__":
    main()
