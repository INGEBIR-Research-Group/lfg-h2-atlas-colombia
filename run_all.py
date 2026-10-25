"""Runs the numbered scripts of scripts/ in their order.

Usage, from the repository root:
    python run_all.py           # 10,000 Monte Carlo realizations, as in the article
    python run_all.py 500       # a quicker check with fewer realizations

The number of realizations is passed to the scripts that draw them (11, 12, 14 and 20). The seed is fixed (20260910), so two runs with the same number of realizations give identical outputs. With 10,000 realizations the whole chain takes about one hour on a laptop, most of it in scripts/20_variants.py. No step needs network access.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
NDRAW = sys.argv[1] if len(sys.argv) > 1 else "10000"
TAKES_NDRAW = {"11", "12", "14", "20"}

for script in sorted((ROOT / "scripts").glob("[0-9][0-9]_*.py")):
    args = [NDRAW] if script.name[:2] in TAKES_NDRAW else []
    print(f"== {script.name} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", str(script), *args], cwd=ROOT, check=True)
print("pipeline complete")
