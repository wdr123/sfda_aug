_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
     '../_base_/datasets/cityscapes_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# _base_ = [
#     '../_base_/models/faster-rcnn_r50_fpn.py',
#     '../_base_/datasets/cityscapes_detection.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
# ]
model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=8,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
##source models
src_model = model
src_ckp_list = [
    # './work_dirs/faster-rcnn_r50_fpn_1x_cityscapes/epoch_8.pth',
                './work_dirs/faster-rcnn_r50_fpn_1x_day_clear/epoch_8.pth' ,
                # './work_dirs/faster-rcnn_r50_fpn_1x_day_foggy/epoch_8.pth' ,
                './work_dirs/faster-rcnn_r50_fpn_1x_dusk_rainy/epoch_8.pth',
                './work_dirs/faster-rcnn_r50_fpn_1x_night_rainy/epoch_8.pth',
                './work_dirs/faster-rcnn_r50_fpn_1x_night_sunny/epoch_8.pth'
]
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

# actual epoch = 8 * 8 = 64

#modify training loop
ckpt_pth_list = [
    # './work_dirs/faster-rcnn_r50_fpn_1x_cityscapes/epoch_8.pth',
                './work_dirs/faster-rcnn_r50_fpn_1x_day_clear/epoch_8.pth' ,
                # './work_dirs/faster-rcnn_r50_fpn_1x_day_foggy/epoch_8.pth' ,
                './work_dirs/faster-rcnn_r50_fpn_1x_dusk_rainy/epoch_8.pth',
                './work_dirs/faster-rcnn_r50_fpn_1x_night_rainy/epoch_8.pth',
                './work_dirs/faster-rcnn_r50_fpn_1x_night_sunny/epoch_8.pth'
]
num_src=len(ckpt_pth_list)
train_cfg = dict(type='MSFEpochBasedTrainLoop',ckpt_list=ckpt_pth_list,num_src=num_src,max_epochs=8,val_interval=1)

# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
# TODO: support auto scaling lr
# auto_scale_lr = dict(base_batch_size=8)


data_root = 'data/cityscapes/'
dataset_type = 'CityscapesDataset'
backend_args = None
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instancesonly_filtered_gtFine_val.json',
#         data_prefix=dict(img='leftImg8bit/val/'),
#         test_mode=True,
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=test_pipeline,
#         backend_args=backend_args))
data_root = 'data/daytime_foggy/VOC2007/'
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations/output.json',
        data_prefix=dict(img='JPEGImages'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

custom_hooks = [
    dict(type='MixupPseudoLabelHook', ckpt_pth_list=ckpt_pth_list, num_src = num_src, data_loader=test_dataloader)
]

# load_from = './work_dirs/faster-rcnn_r50_fpn_1x_cityscapes/epoch_8.pth'
