# The new config inherits a base config to highlight the necessary modification
# _base_ = 'retinanet_r50_fpn_ms-640-800-3x_coco.py'

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/cityscapes_detection.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=30))

# optimizer
# lr is set for a batch size of 8
# optim_wrapper = dict(optimizer=dict(lr=0.01))

# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=8,
#         by_epoch=True,
#         # [7] yields higher performance than [6]
#         milestones=[7],
#         gamma=0.1)
# ]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))
# # We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth'
