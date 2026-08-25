"""
Quick setup verification — not a pytest suite, just a manual script to check
required packages are installed (run directly with `python check_setup.py`).
Renamed from test_setup.py so `pytest` doesn't try to collect it as a test
module. `tests/` is currently empty — no automated tests exist yet.
"""
import sys
print(f"Python version: {sys.version}\n")

packages = [
    ('pandas', 'pd'),
    ('numpy', 'np'),
    ('sklearn', None),
    ('websockets', None),
]

for package, alias in packages:
    try:
        if alias:
            exec(f"import {package} as {alias}")
            version = eval(f"{alias}.__version__")
        else:
            exec(f"import {package}")
            version = eval(f"{package}.__version__") if hasattr(eval(package), '__version__') else "installed"
        print(f"✓ {package} {version}")
    except ImportError:
        print(f"✗ {package} not installed")

try:
    from binance.client import Client
    print("✓ python-binance installed")
except ImportError:
    print("✗ python-binance not installed")

print("\n✅ Setup complete! Ready to start coding.")