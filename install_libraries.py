import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = [
    "pandas",
    "numpy",
    "matplotlib",
    "torch"
]

for package in packages:
    install(package)

print("All packages installed successfully.")