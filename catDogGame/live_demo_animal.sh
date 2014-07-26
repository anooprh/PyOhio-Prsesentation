#!/bin/bash

filename=$1

ch_wave $filename -scaleN 0.65 -c 0 -F 16000 -o temp.wav

python extract_feats.py temp.wav

python detect_animal.py -d mel_cep.data 5

rm temp.wav