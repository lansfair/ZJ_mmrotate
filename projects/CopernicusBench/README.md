# CopernicusBench Detection Project

This project adapts the Copernicus-FM backbone from the MMSegmentation
CopernicusBench project for MMRotate oriented-box detection experiments. It
provides Oriented R-CNN configs for DOTA and DIOR-R.

The RGB configs treat ordinary imagery as R/G/B spectral bands and feed those
bands into Copernicus-FM through `RGBToCopernicusFM`.

MMRotate examples:

```bash
python tools/train.py \
  projects/CopernicusBench/configs/copernicus-fm-base_oriented-rcnn_1x_dota-rgb.py

python tools/train.py \
  projects/CopernicusBench/configs/copernicus-fm-base_oriented-rcnn_1x_dior-rgb.py
```
