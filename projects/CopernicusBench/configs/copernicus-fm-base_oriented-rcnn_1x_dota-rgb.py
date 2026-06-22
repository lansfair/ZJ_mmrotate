_base_ = [
    "../../../configs/_base_/schedules/schedule_1x.py",
    "../../../configs/_base_/default_runtime.py",
]

custom_imports = dict(
    imports=["projects.CopernicusBench.copernicusbench"],
    allow_failed_imports=False,
)

data_root = "data/split_ss_dota/"
copernicus_fm_checkpoint = (
    "https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/"
    "CopernicusFM_ViT_base_varlang_e100.pth"
)
work_dir = "./work_dirs/copernicus-fm-base_oriented-rcnn_dota-rgb"

angle_version = "le90"
patch_size = 16
out_channels = 768
fpn_channels = 256
featmap_strides = [4, 8, 16, 32]
backend_args = None

dataset_type = "DOTADataset"
train_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="mmdet.RandomFlip",
        prob=0.75,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(
        type="RGBToCopernicusFM",
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "copernicus_meta",
            "copernicus_rgb_adapter",
        ),
    ),
]
val_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(
        type="RGBToCopernicusFM",
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "copernicus_meta",
            "copernicus_rgb_adapter",
        ),
    ),
]
test_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="RGBToCopernicusFM",
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "copernicus_meta",
            "copernicus_rgb_adapter",
        ),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="trainval/annfiles/",
        data_prefix=dict(img_path="trainval/images/"),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
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
        ann_file="trainval/annfiles/",
        data_prefix=dict(img_path="trainval/images/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="DOTAMetric", metric="mAP")
test_evaluator = val_evaluator

model = dict(
    type="CopernicusFasterRCNN",
    data_preprocessor=dict(
        type="mmdet.DetDataPreprocessor",
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        boxtype2tensor=False,
    ),
    backbone=dict(
        type="CopernicusFMBackbone",
        arch="base",
        frozen_exclude=[],
        norm_eval=True,
        init_cfg=dict(type="Pretrained", checkpoint=copernicus_fm_checkpoint),
        band_wavelengths=(665, 560, 490),
        band_bandwidths=(30, 35, 65),
        var_option="spectrum",
        input_mode="spectral",
        kernel_size=patch_size,
        patch_area=640000,
    ),
    neck=dict(
        type="CopernicusFeature2Pyramid",
        embed_dim=out_channels,
        out_channels=fpn_channels,
        rescales=[4, 2, 1, 0.5],
    ),
    rpn_head=dict(
        type="OrientedRPNHead",
        in_channels=fpn_channels,
        feat_channels=fpn_channels,
        anchor_generator=dict(
            type="mmdet.AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=featmap_strides,
            use_box_type=True,
        ),
        bbox_coder=dict(
            type="MidpointOffsetCoder",
            angle_version=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type="mmdet.SmoothL1Loss",
            beta=0.1111111111111111,
            loss_weight=1.0,
        ),
    ),
    roi_head=dict(
        type="mmdet.StandardRoIHead",
        bbox_roi_extractor=dict(
            type="RotatedSingleRoIExtractor",
            roi_layer=dict(
                type="RoIAlignRotated",
                out_size=7,
                sample_num=2,
                clockwise=True,
            ),
            out_channels=fpn_channels,
            featmap_strides=featmap_strides,
        ),
        bbox_head=dict(
            type="mmdet.Shared2FCBBoxHead",
            predict_box_type="rbox",
            in_channels=fpn_channels,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            reg_predictor_cfg=dict(type="mmdet.Linear"),
            cls_predictor_cfg=dict(type="mmdet.Linear"),
            bbox_coder=dict(
                type="DeltaXYWHTRBBoxCoder",
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
            ),
            reg_class_agnostic=True,
            loss_cls=dict(
                type="mmdet.CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            loss_bbox=dict(
                type="mmdet.SmoothL1Loss",
                beta=1.0,
                loss_weight=1.0,
            ),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="mmdet.MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type="RBbox2HBboxOverlaps2D"),
            ),
            sampler=dict(
                type="mmdet.RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.8),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="mmdet.MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type="RBboxOverlaps2D"),
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="mmdet.RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.8),
            min_bbox_size=0,
        ),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type="nms_rotated", iou_threshold=0.1),
            max_per_img=2000,
        ),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    clip_grad=None,
)
default_hooks = dict(logger=dict(type="LoggerHook", interval=50))
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
auto_scale_lr = dict(enable=False, base_batch_size=16)
