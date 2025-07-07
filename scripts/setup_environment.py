#!/usr/bin/env python3
"""
Virtual Environment Setup Script for RAG Publication Explorer
Provides commands and guidance for creating and managing virtual environments.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


def print_platform_commands():
    """Print platform-specific commands for virtual environment management."""
    
    print("=" * 70)
    print("🐍 VIRTUAL ENVIRONMENT SETUP GUIDE")
    print("=" * 70)
    
    print("\n📋 WHY USE VIRTUAL ENVIRONMENTS?")
    print("-" * 40)
    print("Virtual environments are crucial for Python projects because they:")
    print("• Isolate project dependencies from system-wide Python packages")
    print("• Prevent version conflicts between different projects")
    print("• Ensure reproducible deployments across different machines")
    print("• Allow you to test different versions of libraries safely")
    print("• Keep your global Python installation clean and minimal")
    print("• Enable easy sharing of exact dependency versions with team members")
    
    current_os = platform.system().lower()
    
    print(f"\n🖥️  DETECTED OPERATING SYSTEM: {platform.system()}")
    print("-" * 40)
    
    print("\n1️⃣  CREATE VIRTUAL ENVIRONMENT")
    print("-" * 30)
    if current_os == "windows":
        print("# Using python (Windows)")
        print("python -m venv venv")
        print("\n# Alternative using python3 if available")
        print("python3 -m venv venv")
    else:
        print("# Using python3 (macOS/Linux)")
        print("python3 -m venv venv")
        print("\n# Alternative using python if it points to Python 3")
        print("python -m venv venv")
    
    print("\n2️⃣  ACTIVATE VIRTUAL ENVIRONMENT")
    print("-" * 30)
    
    if current_os == "windows":
        print("# Windows Command Prompt")
        print("venv\\Scripts\\activate")
        print("\n# Windows PowerShell")
        print("venv\\Scripts\\Activate.ps1")
        print("\n# Git Bash on Windows")
        print("source venv/Scripts/activate")
    else:
        print("# macOS/Linux (bash/zsh)")
        print("source venv/bin/activate")
        print("\n# Fish shell")
        print("source venv/bin/activate.fish")
        print("\n# Csh/tcsh")
        print("source venv/bin/activate.csh")
    
    print("\n3️⃣  VERIFY ACTIVATION")
    print("-" * 30)
    print("# Your prompt should show (venv) at the beginning")
    print("# Check Python location:")
    if current_os == "windows":
        print("where python")
    else:
        print("which python")
    print("\n# Check pip location:")
    if current_os == "windows":
        print("where pip")
    else:
        print("which pip")
    
    print("\n4️⃣  DEACTIVATE VIRTUAL ENVIRONMENT")
    print("-" * 30)
    print("# When you're done working (same for all platforms)")
    print("deactivate")
    
    print("\n5️⃣  REMOVE VIRTUAL ENVIRONMENT")
    print("-" * 30)
    if current_os == "windows":
        print("# Windows")
        print("rmdir /s venv")
    else:
        print("# macOS/Linux")
        print("rm -rf venv")
    
    print("\n🔧 COMMON TROUBLESHOOTING")
    print("-" * 30)
    print("• If 'python' command not found, try 'python3'")
    print("• If venv module not found, install it: pip install virtualenv")
    print("• On Ubuntu/Debian: sudo apt-get install python3-venv")
    print("• On macOS with Homebrew: brew install python")
    print("• Always activate before installing packages!")
    
    print("\n📦 NEXT STEPS AFTER ACTIVATION")
    print("-" * 30)
    print("1. Upgrade pip: python -m pip install --upgrade pip")
    print("2. Install project dependencies: pip install -r requirements/requirements.txt")
    print("3. Verify installation: pip list")
    
    print("\n💡 PRO TIPS")
    print("-" * 30)
    print("• Add 'venv/' to your .gitignore file (already done in this project)")
    print("• Use 'pip freeze > requirements.txt' to save current dependencies")
    print("• Consider using tools like pipenv or poetry for advanced dependency management")
    print("• Create different virtual environments for different projects")


def create_activation_scripts():
    """Create platform-specific activation scripts."""
    
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    
    # Windows batch script
    windows_script = """@echo off
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment. Trying python3...
    python3 -m venv venv
)

echo.
echo Virtual environment created successfully!
echo.
echo To activate the environment, run:
echo venv\\Scripts\\activate
echo.
echo Then install dependencies with:
echo pip install -r requirements\\requirements.txt
"""
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Creating virtual environment..."
if command -v python3 &> /dev/null; then
    python3 -m venv venv
elif command -v python &> /dev/null; then
    python -m venv venv
else
    echo "Error: Python not found. Please install Python 3.7 or higher."
    exit 1
fi

echo ""
echo "Virtual environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Then install dependencies with:"
echo "pip install -r requirements/requirements.txt"
"""
    
    # Create Windows script
    with open(scripts_dir / "create_venv.bat", 'w') as f:
        f.write(windows_script)
    
    # Create Unix script
    with open(scripts_dir / "create_venv.sh", 'w') as f:
        f.write(unix_script)
    
    # Make Unix script executable
    try:
        os.chmod(scripts_dir / "create_venv.sh", 0o755)
    except:
        pass  # Might fail on Windows, that's okay
    
    print(f"✓ Created: scripts/create_venv.bat (Windows)")
    print(f"✓ Created: scripts/create_venv.sh (Unix/Linux/macOS)")


def main():
    """Main function to display environment setup information."""
    print_platform_commands()
    print("\n" + "=" * 70)
    
    # Ask if user wants to create helper scripts
    try:
        response = input("\n📝 Create helper scripts for virtual environment setup? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            create_activation_scripts()
            print("\n✅ Helper scripts created in the scripts/ directory!")
    except KeyboardInterrupt:
        print("\n\n👋 Setup guide displayed. You can run this script again anytime!")


if __name__ == "__main__":
    main()
