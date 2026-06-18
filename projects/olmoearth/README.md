# OLMoEarth for MMRotate

This project adds a non-invasive MMRotate 1.x integration for OLMoEarth.

## Which OpenMMLab Project

Use the OLMoEarth project that matches the downstream task:

| Task | Project | Typical data |
| --- | --- | --- |
| Semantic segmentation | MMSegmentation | masks, valid masks, GeoTIFF manifests |
| Horizontal-box detection | MMDetection | rslearn detection manifest, VOC/XML DIOR |
| Oriented-box detection | MMRotate | DOTA txt, DIOR-R oriented XML |

## What Is Reused

- `OlmoEarthBackbone` builds the released OLMoEarth encoder from `config.json`
  and loads the released `.pth` weights through native `init_cfg`.
- `RGBToOlmoEarthRGB` maps ordinary RGB images into the native OLMoEarth
  `rgb` modality. The model-side RGB modality has `B/G/R/NIR` bands; DIOR-R
  and DOTA RGB inputs fill `B/G/R` and mark `NIR` as missing.
- `RGBToOlmoEarthS2` keeps a Sentinel-2 adapter path for released/open-source
  checkpoints that expect the `sentinel2_l2a` modality. It maps RGB images into
  Sentinel-2 band slots and leaves the unavailable bands empty.
- `OlmoEarthMultiLevelNeck` turns the single dense OLMoEarth feature map into
  feature pyramid levels for oriented detectors.
- `OlmoEarthFasterRCNN` forwards batch metainfo such as `present_bands` and
  timestamps into the backbone before normal Faster/Oriented R-CNN logic.

The implementation is registered under `mmrotate.registry` and keeps changes
inside `projects/olmoearth`.

## Config Layout

- `configs/dior/` contains only the six main DIOR-R experiment entrypoints.
- `configs/dota/` contains lightweight DOTA examples for RGB and S2-adapter
  checks.
- `configs/_base_/` contains shared DIOR-R templates used by the six
  entrypoints. These are not intended as primary launch configs.

`configs/dota/olmoearth_oriented-rcnn_1x_dota_rgb.py` uses MMRotate
`DOTADataset` for standard DOTA-style split data:

  ```text
  data/split_ss_dota/
    trainval/
      images/*.png
      annfiles/*.txt
  ```

The six `configs/dior/*.py` entrypoints use MMRotate `DIORDataset` for DIOR-R
oriented XML:

  ```text
  DIOR/
    JPEGImages-trainval/*.jpg
    JPEGImages-test/*.jpg
    ImageSets/Main/train.txt
    ImageSets/Main/test.txt
    Annotations/Oriented Bounding Boxes/*.xml
  ```

Use the direct `*-rgb.py` configs for OLMoEarth 10m/2m RGB-modality
experiments. Use the native `*-rgb-s2adapter.py` configs when evaluating the
released OLMoEarth Sentinel-2 checkpoint on RGB imagery. All provided
oriented-RCNN configs use `patch_size = 16`.

## OLMoEarth Version Matrix

For DIOR-R oriented detection, the project provides one frozen-backbone config
and one full-finetuning config for each OLMoEarth variant:

| Variant | Modality path | Frozen backbone | Full finetune |
| --- | --- | --- | --- |
| native | RGB-to-Sentinel-2 adapter | `configs/dior/olmoearth-native_oriented-rcnn_1x_dior-rgb-s2adapter-frozen.py` | `configs/dior/olmoearth-native_oriented-rcnn_1x_dior-rgb-s2adapter-finetune.py` |
| 10m | native RGB modality | `configs/dior/olmoearth-10m_oriented-rcnn_1x_dior-rgb-frozen.py` | `configs/dior/olmoearth-10m_oriented-rcnn_1x_dior-rgb-finetune.py` |
| 2m | native RGB modality | `configs/dior/olmoearth-2m_oriented-rcnn_1x_dior-rgb-frozen.py` | `configs/dior/olmoearth-2m_oriented-rcnn_1x_dior-rgb-finetune.py` |

The 10m and 2m configs rely on the active Python environment to provide the
intended `olmoearth_pretrain` package. Before running a variant, install that
variant's source tree with `pip install -e -v .` in the corresponding
`olmoearth_pretrain-main` directory. Use one OLMoEarth source variant per
environment or reinstall the editable package before switching variants. The
config defaults expect each model directory to contain `config.json` and
`weights.pth`; if your released checkpoints are stored elsewhere, update
`olmoearth_model_dir` in the corresponding config.

## Run

```bash
python tools/train.py \
  projects/olmoearth/configs/dior/olmoearth-10m_oriented-rcnn_1x_dior-rgb-frozen.py
```

For your server paths, edit these variables at the top of the config:

```python
data_root = "/path/to/data"
olmoearth_model_dir = "/path/to/OlmoEarth-v1-Base"
```

or override the already-expanded fields from the command line:

```bash
python tools/train.py \
  projects/olmoearth/configs/dior/olmoearth-10m_oriented-rcnn_1x_dior-rgb-frozen.py \
  --cfg-options \
  train_dataloader.dataset.data_root="/mnt/ht2-nas2/EO test/zyf/data/DIOR" \
  val_dataloader.dataset.data_root="/mnt/ht2-nas2/EO test/zyf/data/DIOR" \
  test_dataloader.dataset.data_root="/mnt/ht2-nas2/EO test/zyf/data/DIOR" \
  model.backbone.model_config_path="/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/config.json" \
  model.backbone.init_cfg.checkpoint="/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/weights.pth"
```

## DIOR vs DOTA

DIOR itself is not necessarily DOTA format. Original DIOR commonly uses XML
annotations, while DIOR-R provides oriented XML boxes. Some experiments convert
DIOR or DIOR-R into DOTA-like text files. Pick the config according to the
actual annotation files on disk:

- `Annotations/*.xml` with horizontal `bndbox`:
  use the MMDetection DIOR config instead.
- `Annotations/Oriented Bounding Boxes/*.xml` with `robndbox`:
  use MMRotate `DIORDataset`.
- `annfiles/*.txt` where each row is
  `x1 y1 x2 y2 x3 y3 x4 y4 class difficult`:
  use `DOTADataset`.

For a more detailed Chinese walkthrough, see
`projects/olmoearth/docs/mmrotate_migration_zh.md`.
