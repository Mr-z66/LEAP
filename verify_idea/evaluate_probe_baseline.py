import os
import runpy


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_SCRIPT = os.path.join(PROJECT_ROOT, "evaluate_probe_baseline.py")


if __name__ == "__main__":
    runpy.run_path(TARGET_SCRIPT, run_name="__main__")
