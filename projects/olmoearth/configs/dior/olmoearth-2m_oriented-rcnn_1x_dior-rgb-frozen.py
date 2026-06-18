_base_ = ["../_base_/olmoearth_oriented-rcnn_dior_rgb.py"]

olmoearth_model_dir = (
    "D:/ZJ_projects/model_code/code_release/olmoearth2m/OlmoEarth-v1-Base"
)
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "/mnt/qh2-nas3/EO_test/wyf/scale-model-test/dior-r/olmoearth-2m_oriented-rcnn_dior-r_rgb-frozen"

model = dict(
    backbone=dict(
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        frozen=True,
    ),
)
