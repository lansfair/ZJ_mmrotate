_base_ = ["../_base_/olmoearth_oriented-rcnn_dior_rgb.py"]

olmoearth_model_dir = (
    "D:/ZJ_projects/model_code/code_release/olmoearth10m/OlmoEarth-v1-Base"
)
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "./work_dirs/olmoearth-10m_oriented-rcnn_dior-rgb-frozen"
patch_size = 4

model = dict(
    backbone=dict(
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        patch_size=patch_size,
        frozen=True,
    ),
)
