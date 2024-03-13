# Copyright (c) OpenMMLab. All rights reserved.
from .loops import TeacherStudentValLoop
from .single_source_domain_adaptation_training_loops import SingleSourceEpochBasedTrainLoop, CoTEpochBasedTrainLoop


__all__ = ['TeacherStudentValLoop','CoTEpochBasedTrainLoop']
