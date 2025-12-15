#!/usr/bin/env python3

import os
import sys
import runpy
from pathlib import Path
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
FIG_PKG = "et_precip_ppe.figures"

# Add src to path if needed
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def main():
    figures_dir = REPO_ROOT / "scripts" / "figures"
    print(figures_dir)
    
    # Get all Python files in figures directory
    fig_scripts = sorted(figures_dir.glob("supp*.py"))
    print(REPO_ROOT)

    # Save original working directory
    original_cwd = os.getcwd()
    
    for fig_path in fig_scripts:
        print("#######################################################################")
        print(f"##### Running {fig_path.name}...")
        print("#######################################################################")
        
        try:
            # Change to the script's directory
            os.chdir(fig_path.parent)
            runpy.run_path(str(fig_path), run_name="__main__")
        except Exception as e:
            print(f"  Error running {fig_path.name}: {e}")
            traceback.print_exc()
        finally:
            # Restore original directory
            os.chdir(original_cwd)

if __name__ == "__main__":
    main()
