from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.runner.checkpoint import _load_checkpoint
import torch
@HOOKS.register_module()
class LoadModelHook(Hook):
    def __init__(self,ckpt_pth_list):
        # self.model = model
        # self.optimizer = optimizer
        self.ckpt_pth_list = ckpt_pth_list

    def before_train_epoch(self, runner):
        
        checkpoint0 = _load_checkpoint(self.ckpt_pth_list[0])
        checkpoint1 = _load_checkpoint(self.ckpt_pth_list[1])
        checkpoint2 = _load_checkpoint(self.ckpt_pth_list[2])
        checkpoint3 = _load_checkpoint(self.ckpt_pth_list[3])
        checkpoint4 = _load_checkpoint(self.ckpt_pth_list[4])
        checkpoint5 = _load_checkpoint(self.ckpt_pth_list[5])
        # model = runner.model
        # optimizer = runner.optimizer
        # torch.load(self.ckpt_pth)
        # model.load_state_dict(torch.load(self.ckpt_pth))
        # optimizer.load_state_dict(torch.load(self.ckpt_pth))