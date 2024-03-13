import mmengine
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.runner import Runner
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.runner.base_loop import BaseLoop
from mmengine.config import Config, ConfigDict
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.config import Config
from mmengine.runner import Runner

# config = Config.fromfile('configs/faster_rcnn/faster-rcnn_r50_fpn_1x_diverse.py')
# runner = Runner.from_cfg(config)
# runner.train()                            
# ConfigType = Union[Dict, Config, ConfigDict]
# ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,List[_ParamScheduler]]]
# OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]
@HOOKS.register_module()
class MixupPseudoLabelHook(Hook):
    def __init__(self,data_batch, src_outputs, tgt_outputs):
        # self.model = model
        # self.optimizer = optimizer
        self.data_batch = data_batch
        self.src_outputs = src_outputs
        self.tgt_outputs = tgt_outputs
        
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch,
                         src_outputs: Optional[dict] = None,
                         tgt_outputs: Optional[dict] = None) -> None:
        model = runner.model.loss(data_batch, src_outputs)
