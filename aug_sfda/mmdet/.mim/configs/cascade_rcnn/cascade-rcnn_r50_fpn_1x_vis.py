_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/visdrone_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model = dict(
#     backbone=dict(init_cfg=None),
#     roi_head=dict(
#         bbox_head=dict(
#             num_classes=8,
#             loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
