from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet1.engine.runner.runner_msfda_super_one import Runner_msfda_super_one
from mmengine.runner import EpochBasedTrainLoop
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict, load_from_local
from mmengine.runner.checkpoint import _load_checkpoint_to_model
import torch
import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.registry import MODELS
import torch
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
from mmdet.core.post_processing import multiclass_nms

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor

# from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor


# @MODELS.register_module()
# class MixupDataPreprocessor(BaseDataPreprocessor):

@LOOPS.register_module()
class SingleSourceEpochBasedTrainLoop(EpochBasedTrainLoop):
    def __init__(
            self,
            runner:Runner_msfda_super_one,
            dataloader: Union[DataLoader, Dict],
            ckpt_list:List,
            num_src: int = 1,
            max_epochs: int = 8,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_epochs)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        self.num_src=num_src
        self.ckpt_list=ckpt_list
        self.src_model_list =[]
        self._min_th = 5
        # self.data_preprocessor = BaseDataPreprocessor
        # if data_preprocessor is None:
        data_preprocessor = dict(type='BaseDataPreprocessor')
        if isinstance(data_preprocessor, torch.nn.Module):
            self.data_preprocessor = data_preprocessor
        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')
        # self.data_preprocessor = dict(type='BaseDataPreprocessor')
        _load_checkpoint_to_model(self.runner.src_model, torch.load(self.ckpt_list[0])['state_dict'])
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)
        

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    

    def update_pseudo_labels_teacher(self, data_batch):
        src_pred0 = self.runner.src_model0.val_step(data_batch)
        src_pred1 = self.runner.src_model1.val_step(data_batch)
        src_pred2 = self.runner.src_model2.val_step(data_batch)
        src_pred3 = self.runner.src_model3.val_step(data_batch)   

        src_pred = multiclass_nms(torch.concatenate([src_pred0, src_pred1, src_pred2, src_pred3], dim=0))

        scores = src_pred[0].pred_instances['scores']
        labels = src_pred[0].pred_instances['labels']
        bboxes = src_pred[0].pred_instances['bboxes']
        #filter low confidence instances lower than average
        min_confidence = np.mean(src_pred[0].pred_instances['scores'].detach().cpu().numpy())-0.0001
        num = len(torch.where(scores>min_confidence))
        # num = len(torch.where(scores>min_confidence)[0])
        if num>self._min_th:
            sorted_scores = torch.sort(scores)
            min_confidence = sorted_scores.values[-self._min_th-1]

        scores_new = scores[torch.where(scores>min_confidence)].clone().detach()
        labels_new = labels[torch.where(scores>min_confidence)].clone().detach()
        bboxes_new = bboxes[torch.where(scores>min_confidence)].clone().detach()
        bboxes_new.requires_grad_()
        
        del data_batch['data_samples'][0].gt_instances.bboxes,data_batch['data_samples'][0].gt_instances.labels
        
        data_batch['data_samples'][0].gt_instances.bboxes = bboxes_new
        data_batch['data_samples'][0].gt_instances.labels = labels_new
        # breakpoint()
        
        return data_batch
    

    def update_pseudo_labels_student(self, data_batch):
        src_pred = self.runner.model.val_step(data_batch)
        
        scores = src_pred[0].pred_instances['scores']
        labels = src_pred[0].pred_instances['labels']
        bboxes = src_pred[0].pred_instances['bboxes']
        #filter low confidence instances lower than average
        min_confidence = np.mean(src_pred[0].pred_instances['scores'].detach().cpu().numpy())-0.0001
        num = len(torch.where(scores>min_confidence))
        # num = len(torch.where(scores>min_confidence)[0])
        if num>self._min_th:
            sorted_scores = torch.sort(scores)
            min_confidence = sorted_scores.values[-self._min_th-1]
        # breakpoint()
        # min_confidence = np.sort(src_pred[0].pred_instances['scores'].detach().cpu().numpy())[threshold]
        scores_new = scores[torch.where(scores>min_confidence)].clone().detach()
        labels_new = labels[torch.where(scores>min_confidence)].clone().detach()
        bboxes_new = bboxes[torch.where(scores>min_confidence)].clone().detach()
        bboxes_new.requires_grad_()
        
        del data_batch['data_samples'][0].gt_instances.bboxes,data_batch['data_samples'][0].gt_instances.labels
        
        data_batch['data_samples'][0].gt_instances.bboxes = bboxes_new
        data_batch['data_samples'][0].gt_instances.labels = labels_new
        # breakpoint()
        
        return data_batch
    
    def EMAupdate(self, selected_model, momentum):
        with torch.no_grad():
            for name, value in self.runner.model.named_parameters():
                temp = (1-momentum) * selected_model[name].data + momentum * value.data.clone()
                value.data.copy_(temp)


    def run_iter(self, idx, data_batch) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        #obtain pseudo labels from multiple source models
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        data_batch_new = self.update_pseudo_labels_teacher(data_batch)
        data_batch_new1 = self.update_pseudo_labels_student(data_batch)  
        selected_index = random.randint(4)
        selected_model = self.srclist[selected_index]
        
        unfreeze_list = ['roi_head.bbox_head.fc_cls.weight', 'roi_head.bbox_head.fc_cls.bias',
                          'roi_head.bbox_head.fc_reg.weight', 'roi_head.bbox_head.fc_reg.bias']
        
        for name, param in self.runner.model.named_parameters():
            # not freeze head
            if name in unfreeze_list:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # with torch.no_grad():
        #     for (param, param_src) in zip(self.runner.model.module.backbone.parameters(), self.runner.src_model.module.backbone.parameters()):
        #         param.data.copy_(param_src.detach().clone())
        
        # self.EMAupdate(selected_model, momentum=0.95)
        tgt_outputs = self.runner.model.train_step(data_batch_new, optim_wrapper=self.runner.optim_wrapper)

        tgt_outputs_1 = selected_model.train_step(data_batch_new1, optim_wrapper=self.runner.optim_wrapper)

        #mmengine/mmengine/model/base_model/base_model.py def train_step()
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch_new,
            outputs=[tgt_outputs, tgt_outputs_1]
            )
        
        self._iter += 1
        # if self._iter % 5000: 
        #     self._min_th += 5
 

