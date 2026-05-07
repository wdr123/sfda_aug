# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import os.path as osp
import pickle
import platform
import time
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.device import get_device
from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
                           init_dist, is_distributed, master_only)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger, print_log
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)
from mmengine.visualization import Visualizer
# from mmengine.runner.activation_checkpointing import turn_on_activation_checkpointing
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, save_checkpoint,
                         weights_to_cpu)
from mmengine.runner.log_processor import LogProcessor
from mmengine.runner.loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
# from .customized_training_loops import  MSFEpochBasedTrainLoop,MSFMixupEpochBasedTrainLoop
from mmengine.runner.priority import Priority, get_priority
from mmengine.runner.utils import  set_random_seed
from mmengine.runner.runner import Runner
# from mmengine.runner import Runner

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]


@RUNNERS.register_module()
class Runner_msfda_super_one(Runner):
    """A training helper for PyTorch.

    Runner object can be built from config by ``runner = Runner.from_cfg(cfg)``
    where the ``cfg`` usually contains training, validation, and test-related
    configurations to build corresponding components. We usually use the
    same config to launch training, testing, and validation tasks. However,
    only some of these components are necessary at the same time, e.g.,
    testing a model does not need training or validation-related components.

    To avoid repeatedly modifying config, the construction of ``Runner`` adopts
    lazy initialization to only initialize components when they are going to be
    used. Therefore, the model is always initialized at the beginning, and
    training, validation, and, testing related components are only initialized
    when calling ``runner.train()``, ``runner.val()``, and ``runner.test()``,
    respectively.

    Args:
        model (:obj:`torch.nn.Module` or dict): The model to be run. It can be
            a dict used for build a model.
        src_model (:obj:`torch.nn.Module` or dict): The trained source model 
        used to predict pseudo labels. It can be the same architecture of the tgt model.
        work_dir (str): The working directory to save checkpoints. The logs
            will be saved in the subdirectory of `work_dir` named
            :attr:`timestamp`.
        train_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping training steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        val_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping validation steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        test_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping test steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        train_cfg (dict, optional): A dict to build a training loop. If it does
            not provide "type" key, it should contain "by_epoch" to decide
            which type of training loop :class:`EpochBasedTrainLoop` or
            :class:`IterBasedTrainLoop` should be used. If ``train_cfg``
            specified, :attr:`train_dataloader` should also be specified.
            Defaults to None. See :meth:`build_train_loop` for more details.
        val_cfg (dict, optional): A dict to build a validation loop. If it does
            not provide "type" key, :class:`ValLoop` will be used by default.
            If ``val_cfg`` specified, :attr:`val_dataloader` should also be
            specified. If ``ValLoop`` is built with `fp16=True``,
            ``runner.val()`` will be performed under fp16 precision.
            Defaults to None. See :meth:`build_val_loop` for more details.
        test_cfg (dict, optional): A dict to build a test loop. If it does
            not provide "type" key, :class:`TestLoop` will be used by default.
            If ``test_cfg`` specified, :attr:`test_dataloader` should also be
            specified. If ``ValLoop`` is built with `fp16=True``,
            ``runner.val()`` will be performed under fp16 precision.
            Defaults to None. See :meth:`build_test_loop` for more details.
        auto_scale_lr (dict, Optional): Config to scale the learning rate
            automatically. It includes ``base_batch_size`` and ``enable``.
            ``base_batch_size`` is the batch size that the optimizer lr is
            based on. ``enable`` is the switch to turn on and off the feature.
        optim_wrapper (OptimWrapper or dict, optional):
            Computing gradient of model parameters. If specified,
            :attr:`train_dataloader` should also be specified. If automatic
            mixed precision or gradient accmulation
            training is required. The type of ``optim_wrapper`` should be
            AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
            examples. Defaults to None.
        param_scheduler (_ParamScheduler or dict or list, optional):
            Parameter scheduler for updating optimizer parameters. If
            specified, :attr:`optimizer` should also be specified.
            Defaults to None.
            See :meth:`build_param_scheduler` for examples.
        val_evaluator (Evaluator or dict or list, optional): A evaluator object
            used for computing metrics for validation. It can be a dict or a
            list of dict to build a evaluator. If specified,
            :attr:`val_dataloader` should also be specified. Defaults to None.
        test_evaluator (Evaluator or dict or list, optional): A evaluator
            object used for computing metrics for test steps. It can be a dict
            or a list of dict to build a evaluator. If specified,
            :attr:`test_dataloader` should also be specified. Defaults to None.
        default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks to
            execute default actions like updating model parameters and saving
            checkpoints. Default hooks are ``OptimizerHook``,
            ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook`` and
            ``CheckpointHook``. Defaults to None.
            See :meth:`register_default_hooks` for more details.
        custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
            custom actions like visualizing images processed by pipeline.
            Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. If the ``model`` argument is a dict
            and doesn't contain the key ``data_preprocessor``, set the argument
            as the ``data_preprocessor`` of the ``model`` dict.
            Defaults to None.
        load_from (str, optional): The checkpoint file to load from.
            Defaults to None.
        resume (bool): Whether to resume training. Defaults to False. If
            ``resume`` is True and ``load_from`` is None, automatically to
            find latest checkpoint from ``work_dir``. If not found, resuming
            does nothing.
        launcher (str): Way to launcher multi-process. Supported launchers
            are 'pytorch', 'mpi', 'slurm' and 'none'. If 'none' is provided,
            non-distributed environment will be launched.
        env_cfg (dict): A dict used for setting environment. Defaults to
            dict(dist_cfg=dict(backend='nccl')).
        log_processor (dict, optional): A processor to format logs. Defaults to
            None.
        log_level (int or str): The log level of MMLogger handlers.
            Defaults to 'INFO'.
        visualizer (Visualizer or dict, optional): A Visualizer object or a
            dict build Visualizer object. Defaults to None. If not
            specified, default config will be used.
        default_scope (str): Used to reset registries location.
            Defaults to "mmengine".
        randomness (dict): Some settings to make the experiment as reproducible
            as possible like seed and deterministic.
            Defaults to ``dict(seed=None)``. If seed is None, a random number
            will be generated and it will be broadcasted to all other processes
            if in distributed environment. If ``cudnn_benchmark`` is
            ``True`` in ``env_cfg`` but ``deterministic`` is ``True`` in
            ``randomness``, the value of ``torch.backends.cudnn.benchmark``
            will be ``False`` finally.
        experiment_name (str, optional): Name of current experiment. If not
            specified, timestamp will be used as ``experiment_name``.
            Defaults to None.
        cfg (dict or Configdict or :obj:`Config`, optional): Full config.
            Defaults to None.

    Note:
        Since PyTorch 2.0.0, you can enable ``torch.compile`` by passing in
        `cfg.compile = True`. If you want to control compile options, you
        can pass a dict, e.g. ``cfg.compile = dict(backend='eager')``.
        Refer to `PyTorch API Documentation <https://pytorch.org/docs/
        master/generated/torch.compile.html#torch.compile>`_ for more valid
        options.

    Examples:
        >>> from mmengine.runner import Runner
        >>> cfg = dict(
        >>>     model=dict(type='ToyModel'),
        >>>     work_dir='path/of/work_dir',
        >>>     train_dataloader=dict(
        >>>     dataset=dict(type='ToyDataset'),
        >>>     sampler=dict(type='DefaultSampler', shuffle=True),
        >>>     batch_size=1,
        >>>     num_workers=0),
        >>>     val_dataloader=dict(
        >>>         dataset=dict(type='ToyDataset'),
        >>>         sampler=dict(type='DefaultSampler', shuffle=False),
        >>>        batch_size=1,
        >>>        num_workers=0),
        >>>     test_dataloader=dict(
        >>>         dataset=dict(type='ToyDataset'),
        >>>         sampler=dict(type='DefaultSampler', shuffle=False),
        >>>         batch_size=1,
        >>>         num_workers=0),
        >>>     auto_scale_lr=dict(base_batch_size=16, enable=False),
        >>>     optim_wrapper=dict(type='OptimizerWrapper', optimizer=dict(
        >>>         type='SGD', lr=0.01)),
        >>>     param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
        >>>     val_evaluator=dict(type='ToyEvaluator'),
        >>>     test_evaluator=dict(type='ToyEvaluator'),
        >>>     train_cfg=dict(by_epoch=True, max_epochs=3, val_interval=1),
        >>>     val_cfg=dict(),
        >>>     test_cfg=dict(),
        >>>     custom_hooks=[],
        >>>     default_hooks=dict(
        >>>         timer=dict(type='IterTimerHook'),
        >>>         checkpoint=dict(type='CheckpointHook', interval=1),
        >>>         logger=dict(type='LoggerHook'),
        >>>         optimizer=dict(type='OptimizerHook', grad_clip=False),
        >>>         param_scheduler=dict(type='ParamSchedulerHook')),
        >>>     launcher='none',
        >>>     env_cfg=dict(dist_cfg=dict(backend='nccl')),
        >>>     log_processor=dict(window_size=20),
        >>>     visualizer=dict(type='Visualizer',
        >>>     vis_backends=[dict(type='LocalVisBackend',
        >>>                        save_dir='temp_dir')])
        >>>    )
        >>> runner = Runner.from_cfg(cfg)
        >>> runner.train()
        >>> runner.test()
    """
    cfg: Config
    _train_loop: Optional[Union[BaseLoop, Dict]]
    _val_loop: Optional[Union[BaseLoop, Dict]]
    _test_loop: Optional[Union[BaseLoop, Dict]]

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        src_model: Union[nn.Module, Dict],
        work_dir: str,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        auto_scale_lr: Optional[Dict] = None,
        optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        data_preprocessor: Union[nn.Module, Dict, None] = None,
        load_from: Optional[str] = None,
        src_ckp_list: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
        default_scope: str = 'mmengine',
        randomness: Dict = dict(seed=None),
        experiment_name: Optional[str] = None,
        cfg: Optional[ConfigType] = None,
    ):
        super().__init__(model, work_dir, train_dataloader,val_dataloader,test_dataloader,train_cfg,val_cfg,test_cfg,\
        auto_scale_lr,optim_wrapper,param_scheduler,val_evaluator,test_evaluator,default_hooks, custom_hooks, data_preprocessor,\
        load_from,resume,launcher,env_cfg,log_processor,log_level,visualizer,default_scope,randomness,experiment_name,cfg)
        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
        # `dict` with single optimizer, parsed param_scheduler will be a
        # list of parameter schedulers. If `optim_wrapper` is
        # a `dict` with multiple optimizers, parsed `param_scheduler` will be
        # dict with multiple list of parameter schedulers.
        self._check_scheduler_cfg(param_scheduler)
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)
        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        if default_scope is not None:
            default_scope = DefaultScope.get_instance(  # type: ignore
                self._experiment_name,
                scope_name=default_scope)
        self.default_scope = default_scope

        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        self.logger = self.build_logger(log_level=log_level)

        # Collect and log environment information.
        self._log_env(env_cfg)

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        # build a model
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        # if isinstance(src_model, dict) and data_preprocessor is not None:
        #     # Merge the data_preprocessor to model config.
        #     src_model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)
        self.src_model = self.build_model(model)
        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)
        self.src_model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.src_model)
        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)
        # log hooks information
        self.logger.info(f'Hooks will be executed in the following '
                         f'order:\n{self.get_hooks_info()}')

        # dump `cfg` to `work_dir`
        self.dump_config()
    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            src_model=cfg['src_model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner