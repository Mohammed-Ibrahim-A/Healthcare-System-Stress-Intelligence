import subprocess
import sys

scripts = [
    "src/collect.py",
    "src/preprocess.py",
    "src/feature_engineering.py",
    "src/regression.py",
    "src/classification.py",
    "src/clustering.py",
    "src/feature_importance.py",
    "src/visualize.py",
    "src/evaluate.py",
    "src/insights.py",
    "src/trend_analysis.py",
    "src/predict.py"
]

print("=" * 70)
print("Healthcare System Stress Intelligence Pipeline")
print("=" * 70)

for script in scripts:
    print(f"\nRunning {script}...\n")

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\nError while running {script}")
        break

print("\nPipeline Finished Successfully!")