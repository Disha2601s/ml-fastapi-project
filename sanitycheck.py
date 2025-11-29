# Simple sanity check script
import subprocess, sys
print('Running pytest...')
r = subprocess.run(['pytest','-q'], capture_output=True, text=True)
print(r.stdout)
print('Run flake8 (may show style issues)...')
r2 = subprocess.run(['flake8'], capture_output=True, text=True)
print(r2.stdout)
