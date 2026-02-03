# install_exact_min_versions.py
# Run this script with: python install_exact_min_versions.py
# Recommended: run it inside a fresh virtual environment!
#
# Purpose: Installs at least the requested versions of the packages needed for
# tools like NTTuner / LoRA trainers (DearPyGui GUI + PEFT/LoRA stack).

import subprocess
import sys

# Minimum versions you requested
MIN_VERSIONS = {
    "dearpygui":   "1.10.0",
    "transformers": "4.36.0",
    "datasets":     "2.14.0",
    "trl":          "0.7.4",
    "peft":         "0.7.0",
    "accelerate":   "0.25.0",
}

# Optional: common extras for LoRA/QLoRA (uncomment if needed)
# EXTRA_PACKAGES = ["bitsandbytes", "torch", "torchvision", "torchaudio"]

def run_pip(cmd):
    """Run a pip command and print output."""
    full_cmd = [sys.executable, "-m", "pip"] + cmd
    print(f"\n→ Executing: {' '.join(full_cmd)}")
    try:
        result = subprocess.run(
            full_cmd,
            check=True,
            text=True,
            capture_output=True,
            encoding="utf-8"
        )
        print(result.stdout.strip())
        if result.stderr:
            print("Warnings/Errors:", result.stderr.strip())
    except subprocess.CalledProcessError as e:
        print(f"Failed with code {e.returncode}:")
        print(e.stderr)
        sys.exit(1)


def main():
    print("=== Installing minimum required versions for LoRA/NTTuner stack ===")
    print("This will install (or upgrade to) at least these versions:\n")
    for pkg, ver in MIN_VERSIONS.items():
        print(f"  • {pkg} >= {ver}")
    print("\nNote: pip may install newer compatible versions (recommended).\n")

    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # Step 1: Upgrade pip (helps avoid dependency resolver issues)
    print("\n[1/3] Upgrading pip...")
    run_pip(["install", "--upgrade", "pip", "setuptools", "wheel"])

    # Step 2: Install each package with >= constraint
    print("\n[2/3] Installing packages...")
    for pkg, min_ver in MIN_VERSIONS.items():
        req = f"{pkg}>={min_ver}"
        run_pip(["install", "--upgrade", req])

    # Optional: Install extras if you uncommented EXTRA_PACKAGES
    # print("\n[3/3] Installing extras...")
    # for extra in EXTRA_PACKAGES:
    #     run_pip(["install", extra])

    print("\n=== Installation finished! ===")
    print("Installed / upgraded versions:\n")
    run_pip(["list", "--format=freeze"])

    print("\nNext steps:")
    print("  • Activate your virtual env if using one")
    print("  • Run: python NTTuner.py   (or your trainer script)")
    print("  • If still 'missing peft' → verify with:")
    print("    python -c \"import peft; print(peft.__version__)\"")


if __name__ == "__main__":
    main()
