_base_ = ["./olmoearth-base_oriented-rcnn_1x_dior-rgb-s2adapter.py"]

olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"
work_dir = "./work_dirs/olmoearth-native_oriented-rcnn_dior-rgb-s2adapter-frozen"

model = dict(
    backbone=dict(
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        frozen=True,
    ),
)
