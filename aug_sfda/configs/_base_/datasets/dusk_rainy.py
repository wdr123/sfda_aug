# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/dusk_rainy/VOC2007/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/segmentation/cityscapes/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#          'data/': 's3://openmmlab/datasets/segmentation/'
#      }))
backend_args = None
classes = ('car', 'person', 'bus','bike','motor','rider','truck',)
data=dict(train=dict(classes=classes),  val=dict(classes=classes),  test=dict(classes=classes))
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(2048, 800), (2048, 1024)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

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

# metainfo = {
#     'classes': ('balloon', ),
#     'palette': [
#         (220, 20, 60),
#     ]
# }
# train_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='train.json',
#         data_prefix=dict(img='VisDrone2019-DET-train\\images')))
# val_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='val.json',
#         data_prefix=dict(img='VisDrone2019-DET-val\\images')))
# test_dataloader = val_dataloader

# # Modify metric related settings
# val_evaluator = dict(ann_file=data_root + 'val.json')
# test_evaluator = val_evaluator

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='Annotations/output.json',
            data_prefix=dict(img='JPEGImages'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=8,
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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Annotations/output.json',
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator
