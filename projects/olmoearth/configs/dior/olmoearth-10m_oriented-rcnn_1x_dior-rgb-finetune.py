_base_ = ["../_base_/olmoearth_oriented-rcnn_dior_rgb.py"]

olmoearth_model_dir = (
    "/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/"
    "new_olmoearth/olmoearth_10m/weight"
)
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "/mnt/qh2-nas3/EO_test/wyf/scale-model-test/dior-r/olmoearth-10m_oriented-rcnn_dior-r_rgb-finetune"
patch_size = 4

model = dict(
    backbone=dict(
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        patch_size=patch_size,
        frozen=False,
    ),
)

optim_wrapper = dict(
    clip_grad=dict(_delete_=True, max_norm=1.0, norm_type=2),
)
