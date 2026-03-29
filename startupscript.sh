#!/bin/bash

python3.12 -m venv .venv
source .venv/bin/activate
rm -rf analysis_outputs watermarked_audios
pip install -r requirements.txt
python3.12 src/main.py