"""
Run all 6 domain adaptation methods on all 4 target domains.
Source: 晴天 (Sunny), Targets: 逆光, 雨天, 雾天, 黑天
"""
import subprocess
import os
import re

PYTHON = r'D:\Anaconda\envs\pytorch\python.exe'
WORK_DIR = r'd:\Code\JMDA_Com'

methods = [
    ('DANN', os.path.join(WORK_DIR, 'DANN', 'train_dann.py'), 3),
    ('DeepCORAL', os.path.join(WORK_DIR, 'DeepCORAL', 'train_deepcoral.py'), 3),
    ('CDAN', os.path.join(WORK_DIR, 'CDAN', 'train_cdan.py'), 3),
    ('AD-CLIP', os.path.join(WORK_DIR, 'AD-CLIP', 'train_adclip.py'), 3),
    ('CoOp', os.path.join(WORK_DIR, 'CoOp', 'train_coop.py'), 3),
    ('TransAdapter', os.path.join(WORK_DIR, 'TransAdapter', 'train_transadapter.py'), 3),
]

targets = ['backlight', 'rainy', 'foggy', 'dark']

results = {}

for method_name, script, epochs in methods:
    results[method_name] = {}
    for target in targets:
        print(f'\n{"="*60}')
        print(f'Running {method_name} on target={target} (epochs={epochs})')
        print(f'{"="*60}')

        cmd = [PYTHON, script, '--target', target, '--epochs', str(epochs)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=600, cwd=os.path.dirname(script))
            output = proc.stdout + proc.stderr

            # Extract best accuracy from output
            match = re.search(r'Best target val accuracy.*?:\s*([0-9.]+)', output)
            if match:
                acc = float(match.group(1))
                results[method_name][target] = acc
                print(f'  Result: {acc:.4f}')
            else:
                results[method_name][target] = 'N/A'
                print(f'  Result: N/A (could not parse accuracy)')
                # Print last 20 lines for debugging
                lines = output.strip().split('\n')
                for line in lines[-10:]:
                    print(f'    {line}')
        except subprocess.TimeoutExpired:
            results[method_name][target] = 'TIMEOUT'
            print(f'  Result: TIMEOUT')
        except Exception as e:
            results[method_name][target] = f'ERROR: {e}'
            print(f'  Result: ERROR: {e}')

# Print summary table
print('\n\n' + '='*80)
print('SUMMARY OF RESULTS (Target Val Accuracy)')
print('='*80)
print(f'{"Method":<15} {"backlight":>12} {"rainy":>12} {"foggy":>12} {"dark":>12}')
print('-'*80)
for method_name in results:
    row = f'{method_name:<15}'
    for target in targets:
        val = results[method_name][target]
        if isinstance(val, float):
            row += f' {val:>11.4f}'
        else:
            row += f' {str(val):>11}'
    print(row)

# Save results to file
with open(os.path.join(WORK_DIR, 'results_summary.txt'), 'w') as f:
    f.write('Domain Adaptation Methods - Results Summary\n')
    f.write('Source: 晴天 (Sunny), Targets: 逆光(backlight), 雨天(rainy), 雾天(foggy), 黑天(dark)\n')
    f.write('='*80 + '\n')
    f.write(f'{"Method":<15} {"backlight":>12} {"rainy":>12} {"foggy":>12} {"dark":>12}\n')
    f.write('-'*80 + '\n')
    for method_name in results:
        row = f'{method_name:<15}'
        for target in targets:
            val = results[method_name][target]
            if isinstance(val, float):
                row += f' {val:>11.4f}'
            else:
                row += f' {str(val):>11}'
        f.write(row + '\n')

print(f'\nResults saved to {os.path.join(WORK_DIR, "results_summary.txt")}')
