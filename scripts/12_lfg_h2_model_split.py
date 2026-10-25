"""Monte Carlo model with the reformer capital split into a common level and a site residual.

Runs scripts/11_lfg_h2_model.py with the option --split, which draws the reformer capital as a common level (standard error of the fitted mean) times an independent residual per site. It writes only results/mc_national_split.csv and leaves every other output untouched.

Run from the project root after scripts/11_lfg_h2_model.py:
    python scripts/12_lfg_h2_model_split.py [NDRAW]
"""
import pathlib
import runpy
import sys

MODEL = pathlib.Path(__file__).resolve().parent / "11_lfg_h2_model.py"
sys.argv = [str(MODEL), *sys.argv[1:], "--split"]
runpy.run_path(str(MODEL), run_name="__main__")
