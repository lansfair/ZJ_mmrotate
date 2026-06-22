_base_ = ["./copernicus-fm-base_oriented-rcnn_1x_dota-rgb.py"]

data_root = "/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/dior-r"
work_dir = (
    "/mnt/qh2-nas3/EO_test/wyf/scale-model-test/dior-r/"
    "copernicus-fm-base_oriented-rcnn_dior-rgb"
)

classes = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)
metainfo = dict(classes=classes)
num_classes = len(classes)

dataset_type = "DOTADataset"
img_suffix = "jpg"
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="trainval/labelTxt/",
        data_prefix=dict(img_path="trainval/images/"),
        img_suffix=img_suffix,
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=_base_.train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="test/labelTxt/",
        data_prefix=dict(img_path="test/images/"),
        img_suffix=img_suffix,
        metainfo=metainfo,
        test_mode=True,
        pipeline=_base_.val_pipeline,
    ),
)
test_dataloader = val_dataloader

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
    ),
)
