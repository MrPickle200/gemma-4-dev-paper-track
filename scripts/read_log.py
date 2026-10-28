import os
import sys
import unittest.mock

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Patch open in kaggle to always use utf-8
orig_open = open
def utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if 'b' not in mode:
        if encoding is None:
            encoding = 'utf-8'
    else:
        encoding = None
        errors = None
        newline = None
    return orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

import builtins
builtins.open = utf8_open

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--slug", type=str, default="daoviet/gemma-4-e4b-m4-sft")
args, _ = parser.parse_known_args()

slug = args.slug
kernel_slug = slug.split("/")[1]

if "e4b" in slug:
    out_dir = r'notebooks/sft_e4b_kernel/log_output'
else:
    out_dir = r'notebooks/sft_kernel/log_output'

os.makedirs(out_dir, exist_ok=True)
api.kernels_output(slug, path=out_dir)

log_file = os.path.join(out_dir, f'{kernel_slug}.log')
if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
        print('Length:', len(text))
        print(text[-4000:])
else:
    print(f'Log file not found at: {log_file}')
    print('Contents of out_dir:', os.listdir(out_dir))
