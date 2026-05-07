src_ckp_list = [
    './work_dirs/faster-rcnn_r50_fpn_1x_source_coco/coco_pre.pth',
]

train_cfg = dict(type='SingleSourceEpochBasedTrainLoop', ckpt_list=src_ckp_list,num_src=1, max_epochs=4)