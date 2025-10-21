#!/bin/bash

COMMON="--data_dir data/windows_ieee33 --tag W96 \
  --epochs 60 \
  --lr 1e-3 3e-4 \
  --hidden 256 --layers 2 --nhead 4 \
  --bias_scale 0.0 3.0 \
  --trust_lambda 0.0 1e-3 \
  --w_temp_th 1e-4 --w_temp_vm 1e-4"

# 1) use_mask=ON,  bus_smooth=ON（默认开启）
python tools/train_sweep_and_compare.py $COMMON \
  --use_mask \
  --out_dir results/W96_maskON_busON

# 2) use_mask=ON,  bus_smooth=OFF（显式关闭）
python tools/train_sweep_and_compare.py $COMMON \
  --use_mask --no_bus_smooth \
  --out_dir results/W96_maskON_busOFF

# 3) use_mask=OFF, bus_smooth=ON
python tools/train_sweep_and_compare.py $COMMON \
  --out_dir results/W96_maskOFF_busON

# 4) use_mask=OFF, bus_smooth=OFF
python tools/train_sweep_and_compare.py $COMMON \
  --no_bus_smooth \
  --out_dir results/W96_maskOFF_busOFF