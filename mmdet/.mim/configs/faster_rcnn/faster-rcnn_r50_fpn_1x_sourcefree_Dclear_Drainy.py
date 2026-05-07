_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
     '../_base_/datasets/dusk_rainy.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# _base_ = [
#     '../_base_/models/faster-rcnn_r50_fpn.py',
#     '../_base_/datasets/cityscapes_detection.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
# ]
model = dict(
    backbone=dict(frozen_stages=1,init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=7,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
src_model = dict(
    backbone=dict(frozen_stages=1,init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=7,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
# optimizer
# lr is set for a batch size of 8
optim_wrapper = dict(optimizer=dict(lr=0.01))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        # [7] yields higher performance than [6]
        milestones=[7],
        gamma=0.1)
]
src_ckp_list = [
    #'./work_dirs/faster-rcnn_r50_fpn_1x_source_cityscapes/epoch_8.pth',
                './work_dirs/faster-rcnn_r50_fpn_1x_source_day_clear/epoch_7.pth',
                # './work_dirs/faster-rcnn_r50_fpn_1x_source_dusk/epoch_8.pth',
                # './work_dirs/faster-rcnn_r50_fpn_1x_source_day_foggy/epoch_8.pth',
                # './work_dirs/faster-rcnn_r50_fpn_1x_source_foggy_cityscapes/epoch_8.pth'
                # './work_dirs/faster-rcnn_r50_fpn_1x_source_night_sunny/epoch_8.pth',
]
# actual epoch = 8 * 8 = 64
train_cfg = dict(type='SingleSourceEpochBasedTrainLoop', ckpt_list=src_ckp_list,num_src=1, max_epochs=8)

# For better, more stable performance initialize from COCO

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
# TODO: support auto scaling lr
# auto_scale_lr = dict(base_batch_size=8)



# dataset_type = 'CityscapesDataset'
# data_root = 'data/daytime_foggy/VOC2007/'
# backend_args = None
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa

load_from = './work_dirs/faster-rcnn_r50_fpn_1x_source_day_clear/epoch_7.pth'
# load_from = './checkpoints/regionclip_pretrained-cc_rn50_mmdet.pth'
# load_from = './work_dirs/faster-rcnn_r50_fpn_1x_source_dusk/epoch_8.pth'