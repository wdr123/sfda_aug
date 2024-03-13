_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn_aug.py',
     '../_base_/datasets/night_rainy1.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
style_stats_path_src = './data/night_rainy_aug/dusk_rainy'
style_stats_path_tgt = './data/night_rainy_aug/night_sunny'

src_ckp_list = [
    # './work_dirs/coco_pre/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # './work_dirs/faster-rcnn_r50_fpn_1x_source_day_clear/epoch_8.pth',
    # './work_dirs/faster-rcnn_r50_fpn_1x_source_day_foggy/epoch_8.pth',
    './work_dirs/faster-rcnn_r50_fpn_1x_source_dusk_rainy/epoch_8.pth',
    # './work_dirs/faster-rcnn_r50_fpn_1x_source_night_rainy/epoch_8.pth',
    # './work_dirs/faster-rcnn_r50_fpn_1x_source_night_sunny/epoch_8.pth'
]
load_from = './work_dirs/faster-rcnn_r50_fpn_1x_source_night_sunny/epoch_8.pth'
model = dict(
    backbone=dict(frozen_stages=3,init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=7,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            style_stats = style_stats_path_tgt,
            batch_size = 1,
            random_prob=0.5)))
src_model = dict(
    backbone=dict(frozen_stages=3,init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=7,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            style_stats = style_stats_path_src,
            batch_size = 1,
            random_prob=0.5)))
# optimizer
# lr is set for a batch size of 8
optim_wrapper = dict(optimizer=dict(lr=0.01))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        # [7] yields higher performance than [6]
        milestones=[7],
        gamma=0.1)
]

test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,#0.05
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
# actual epoch = max_epochs * 8 = 64
# model training and testing settings
train_cfg = dict(type='CoTEpochBasedTrainLoop', ckpt_list=src_ckp_list,num_src=1, max_epochs=4)

# For better, more stable performance initialize from COCO

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
# TODO: support auto scaling lr
# auto_scale_lr = dict(base_batch_size=8)

# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
