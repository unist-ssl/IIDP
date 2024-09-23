from contextlib import contextmanager

import torch

from iidp.profiler.comp.profile_data_schema import CompProfileData
from iidp.profiler.profile_trainer import IIDPSingleGPUProfileTrainer


WARMUP_STEP = 10
COMP_PROFILE_STEP = 90


class IIDPCustomProfilerHelper(object):
    def __init__(self, lbs, num_models):
        """
        The below member variables are required for ComputationProfiler
            ```model_name```
            ```lbs```
            ```num_models```
            ```profile_data```
        """
        self.gpu = 0
        self.model_name = None
        self.lbs = lbs
        self.num_models = num_models
        self.accum_step = 0
        self.weight_sync_method = 'sequential'

        self.trainer = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.param_groups_func = None
        self.warmup_step = WARMUP_STEP
        self.num_minibatches = COMP_PROFILE_STEP

        # Profile Data
        self.profile_data = CompProfileData()
        self.auxiliary_profile_data = {}

        self.cuda_time = 0

    def set_optimizer(self):
        raise NotImplementedError

    def prepare(self):
        torch.manual_seed(31415)
        self.trainer = IIDPSingleGPUProfileTrainer(
            self.gpu, self.lbs, self.num_models, self.accum_step, self.weight_sync_method)
        self.trainer.prepare_stream_parallel(self.model, self.criterion)
        self.set_optimizer()
        self.trainer.prepare_weight_sync_method(self.optimizer, None, self.param_groups_func)

    def run(self):
        raise NotImplementedError

    @contextmanager
    def record_cuda_time(self):
        try:
            self.cuda_time = 0
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            yield
        finally:
            event_end.record()
            torch.cuda.synchronize()
            self.cuda_time = event_start.elapsed_time(event_end)