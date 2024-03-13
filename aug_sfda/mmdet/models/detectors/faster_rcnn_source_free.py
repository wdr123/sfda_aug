# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.registry import MODELS
from typing import Dict, Optional, Tuple, Union
from mmengine.optim import OptimWrapper
import torch
import torch.nn as nn
import copy
@MODELS.register_module()
class FasterRCNN_source_free(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        if data_preprocessor is None:
            data_preprocessor = dict(type='BaseDataPreprocessor')
        if isinstance(data_preprocessor, nn.Module):
            self.data_preprocessor = data_preprocessor
        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')
    def src_val_step(self, data: Union[dict, tuple, list]):
    # def src_val_step(self, data: Union[dict, tuple, list],
    #                optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """

        data = self.data_preprocessor(data, True)
        losses = self._run_forward(data, mode='loss')
        
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        
        src_pred =  self._run_forward(data, mode='predict')
        
        
        return parsed_losses, log_vars, src_pred

    # def tgt_train_step(self, data: Union[dict, tuple, list],src_pred,
    #                optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    #     """Gets the predictions of given data.

    #     Calls ``self.data_preprocessor(data, False)`` and
    #     ``self(inputs, data_sample, mode='predict')`` in order. Return the
    #     predictions which will be passed to evaluator.

    #     Args:
    #         data (dict or tuple or list): Data sampled from dataset.

    #     Returns:
    #         list: The predictions of given data.
    #     """

    #     with optim_wrapper.optim_context(self):
    #         data = self.data_preprocessor(data, True)
    #         losses = self._run_forward_tgt(data, src_pred, mode='loss')  # type: ignore
    #     parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
    #     optim_wrapper.update_params(parsed_losses)
        
    #     return log_vars
   
    # def _run_forward_tgt(self, data: Union[dict, tuple, list], src_pred,
    #                  mode: str) -> Union[Dict[str, torch.Tensor], list]:
    #     """Unpacks data for :meth:`forward`

    #     Args:
    #         data (dict or tuple or list): Data sampled from dataset.
    #         mode (str): Mode of forward.

    #     Returns:
    #         dict or list: Results of training or testing mode.
    #     """
    #     if isinstance(data, dict):
    #         results = self(**data, mode=mode)
    #     elif isinstance(data, (list, tuple)):
    #         results = self(*data, mode=mode)
    #     else:
    #         raise TypeError('Output of `data_preprocessor` should be '
    #                         f'list, tuple or dict, but got {type(data)}')
    #     return results
