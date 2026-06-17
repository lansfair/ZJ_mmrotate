_base_ = ["./olmoearth-base_oriented-rcnn_1x_dior-rgb.py"]

olmoearth_model_dir = (
    "D:/ZJ_projects/model_code/code_release/olmoearth2m/OlmoEarth-v1-Base"
)
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "./work_dirs/olmoearth-2m_oriented-rcnn_dior-rgb-frozen"

model = dict(
    backbone=dict(
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        frozen=True,
    ),
)
