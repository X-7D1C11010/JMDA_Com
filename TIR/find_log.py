import os, sys

log = sys.argv[1]
dataset = sys.argv[2]
root = 'output/log'
d = root + '/' + dataset + '/coop_lora/'

for f in os.listdir(d):
    f = os.path.join(d, f)
    lines = open(f, 'r', encoding='latin-1').readlines()
    if len(lines) == 0:
        continue
    if log in lines[-1]:
        print(f, lines[-1])
