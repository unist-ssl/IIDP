
from contextlib import contextmanager
import time
import datetime

import torch
import torch.distributed as dist

import threading
import inspect
import copy

from iidp.utils.distributed import get_allgather_value, print_one_rank
from iidp.utils.global_vars import REGISTERED_WEIGHT_SYNC_METHODS, WEIGHT_SYNC_METHOD_SELECTION_THRESHOLD
from iidp.train.epoch import EpochIterator


def calculate_bucket_size_distribution(iidp_module):
    assert type(iidp_module) == torch.nn.parallel.IndependentIdenticalDataParallel
    bucket_size_distribution = []
    parameter_size_distribution = []
    for _, param in enumerate(iidp_module.ddp_register_params):
        if hasattr(param, 'index'):
            param_mem_value = round(param.nelement() * param.element_size() / (1024 ** 2), 2)
            parameter_size_distribution.append(param_mem_value)

    for bucket in iidp_module.bucket_indices:
        bucket_size = 0
        for param_index in bucket:
            param_size = parameter_size_distribution[param_index]
            bucket_size += param_size
        bucket_size_distribution.append(round(bucket_size, 2))

    return bucket_size_distribution


def select_weight_sync_method(bucket_size_distribution, bukcet_capacity):
    def get_avg(val):
        if isinstance(val, (list, tuple)):
            return int(sum(val) / len(val))

    weight_sync_method = '' # return value
    # == [Main algorithm for weight synchronization method selection] ==
    allreduce_overlap_bucket_distribution = bucket_size_distribution[:-1] # Last bucket doesn't overlap with all-reduce kernel
    potential_interference_bucket = allreduce_overlap_bucket_distribution
    avg_outlier_bucket_size = get_avg(potential_interference_bucket)
    norm_avg_outlier_bucket_size = avg_outlier_bucket_size / bukcet_capacity
    """
    If the distribution of bucket size is not uniform,
    overlapping optimizer is not recommended
    due to interference with all-reduce NCCL kernel
    """
    if norm_avg_outlier_bucket_size > WEIGHT_SYNC_METHOD_SELECTION_THRESHOLD: # Threshold is heuristic
        weight_sync_method = 'sequential'
    else:
        weight_sync_method = 'overlap'

    log = f'Recommend ```{weight_sync_method}``` as weight sync method ' \
            f'as uniformity of bucket size is {norm_avg_outlier_bucket_size}'
    print_one_rank(log)

    return weight_sync_method


class GlobalTrainerState(object):
    def __init__(self):
        self.partition_size = []
        self.is_accum_mode = False
        self.global_batch_size = 0


class LocalTrainerState(object):
    def __init__(self):
        self.local_batch_size = 0
        self.num_models = 0
        self.accum_step = 0


GLOBAL_TRAINER_STATE = GlobalTrainerState()
LOCAL_TRAINER_STATE = LocalTrainerState()


class TrainerHelper(object):
    def __init__(self, gpu, local_batch_size, num_models, accum_step=0, weight_sync_method='recommend'):
        self.gpu = gpu
        self.local_batch_size = local_batch_size
        self.num_models = num_models
        self.accum_step = accum_step
        self.max_accum_step = -1
        self.batch_size_per_gpu = self.local_batch_size * self.num_models
        if self.batch_size_per_gpu % local_batch_size != 0:
            raise ValueError('Local batch size must be dividied by batch size per GPU')
        if weight_sync_method not in REGISTERED_WEIGHT_SYNC_METHODS:
            raise ValueError(f'Not support unregisted weight_sync_method: {weight_sync_method}')
        self.weight_sync_method = weight_sync_method

        self.model_streams = []

        self.original_local_models = []
        self.local_models = []
        self.local_optimizers = []
        self.local_schedulers = []
        self.criterion = None
        self.output_as_loss = False
        self.losses = {}
        self.sampler = None

        # For overlapping optimizer
        self.prepared_for_ddp = False
        self.hooks = []
        self.optimizer_stream = None
        # One DDP model's bucket_indices (All of local model's bucket indices is same)
        self.ddp_bucket_indices = []
        self.is_rebuilt_ddp_bucket_indices = False

        self.all_num_models_in_process_group = self._get_all_num_models_in_process_group()
        self._get_total_num_decoupled_workers()
        self.global_batch_size = self.local_batch_size * self.total_num_decoupled_workers

        self.all_partition_size_in_process_group = []
        self._get_all_partition_size_in_process_group()

        self.all_accum_step_in_process_group = []
        self._get_all_accum_step_in_process_group()
        self.max_accum_step = max(self.all_accum_step_in_process_group)

        self.is_accum_mode = True if self.max_accum_step > 0 else False
        # Used in seq_parallel_compute() for being block different number of VSWs on inter-node
        self.sync_accum_barrier = threading.Barrier(self.num_models)

        # It is used for _sync_params() in [torch/nn/parallel/distributed.py]
        self._sync_buffer_barrier = [None, None]
        if self.num_models > 1:
            self._sync_buffer_barrier = [threading.Barrier(self.num_models) for i in range(2)]

        self._set_trainer_state()

        self.local_accum_step = -1
        self.sync_step = 0
        self.epoch_iterator = EpochIterator()
        self.elapsed_time = 0
        self.total_epoch_time = 0

    def _set_trainer_state(self):
        GLOBAL_TRAINER_STATE.partition_size = self.all_partition_size_in_process_group
        GLOBAL_TRAINER_STATE.is_accum_mode = self.is_accum_mode
        GLOBAL_TRAINER_STATE.global_batch_size = self.global_batch_size

        LOCAL_TRAINER_STATE.local_batch_size = self.local_batch_size
        LOCAL_TRAINER_STATE.num_models = self.num_models
        LOCAL_TRAINER_STATE.accum_step = self.accum_step

    def _get_total_num_decoupled_workers(self):
        if dist.is_initialized():
            tensor = torch.tensor([self.num_models * (self.accum_step+1)], dtype=torch.int64).to(self.gpu)
            dist.all_reduce(tensor) # Default op is SUM
            self.total_num_decoupled_workers = tensor.item()
            tensor.cpu()
            del tensor
        else:
            self.total_num_decoupled_workers = self.num_models * (self.accum_step+1)

    def _get_all_partition_size_in_process_group(self):
        local_partition_size = (self.batch_size_per_gpu * (self.accum_step+1)) / self.global_batch_size
        self.all_partition_size_in_process_group = get_allgather_value(local_partition_size, self.gpu)

    def _get_all_accum_step_in_process_group(self):
        self.all_accum_step_in_process_group = get_allgather_value(self.accum_step, self.gpu)

    def _get_all_num_models_in_process_group(self):
        return get_allgather_value(self.num_models, self.gpu)

    def set_original_local_models(self, models):
        """Set the compelete local models by users"""
        if models is None:
            raise ValueError(f"Argument is None: {models}")
        else:
            if not isinstance(models, (list, tuple)):
                raise ValueError(
                    f"Argument models must be list or tuple type: {type(models)}")
        self.original_local_models = models
        assert len(self.original_local_models) == self.num_models

    def set_local_optimizers(self, optimizers):
        """Set the compelete local optimizers by users"""
        if optimizers is None:
            raise ValueError(f"Argument is None: {optimizers}")
        else:
            if not isinstance(optimizers, (list, tuple)):
                raise ValueError(
                    f"Argument optimizers must be list or tuple type: {type(optimizers)}")
        self.local_optimizers = optimizers
        assert len(self.local_optimizers) == self.num_models

    def set_local_schedulers(self, schedulers=None):
        """Set the compelete local schedulers by users"""
        # LR scheduler is optional
        if schedulers is not None:
            if not isinstance(schedulers, (list, tuple)):
                raise ValueError(
                    f"Argument optimizers must be list or tuple type: {type(schedulers)}")
            self.local_schedulers = schedulers
            assert len(self.local_schedulers) == self.num_models

    def _set_original_local_models(self, model):
        if model is None:
            raise ValueError(f"Argument is None: {model}")
        is_set_by_user = (len(self.original_local_models) == self.num_models)
        if not is_set_by_user:
            self.original_local_models = [model]
            for _ in range(1, self.num_models):
                copied_model = copy.deepcopy(model)
                self.original_local_models.append(copied_model)

    def _set_criterion(self, criterion):
        if criterion is None:
            raise ValueError(f"Argument is None: {criterion}")
        self.criterion = criterion
        if hasattr(self.criterion, 'forward'):
            args_of_criterion = inspect.getfullargspec(getattr(self.criterion, 'forward')).args
        else:
            args_of_criterion = inspect.getfullargspec(self.criterion).args
        if 'self' in args_of_criterion:
            args_of_criterion.remove('self')
        num_args = len(args_of_criterion)
        if num_args == 1:
            self.output_as_loss = True
        elif num_args == 2: # We expect arguments as output (y) and target (y^)
            self.output_as_loss = False
        else:
            raise ValueError(
                f"Not support number of arguments in criterion function > 2: {num_args}")

    def _get_required_args_value(self, instance):
        """Helper function for _set_local_optimizers() and _set_local_schedulers()"""
        removable_args = ['self', 'optimizer', 'params', 'lr']
        args_inspect = inspect.getfullargspec(instance.__init__)
        args_of_instace = args_inspect.args
        filtered_args_of_instance = [x for x in args_of_instace if x not in removable_args]
        is_defaults_exists = (args_inspect.defaults is not None and len(args_inspect.defaults) > 1)
        if is_defaults_exists:
            required_args = filtered_args_of_instance[:-len(args_inspect.defaults)]
        else:
            required_args = filtered_args_of_instance
        args = []
        for arg_name in required_args:
            try:
                # NOTE: In torch/optim/lr_scheduler.py, ```LambdaLR``` class has self.lr_lambdas,
                # but argument is lr_lambda
                if arg_name == 'lr_lambda':
                    args.append(instance.__dict__['lr_lambdas'][0])
                else:
                    args.append(instance.__dict__[arg_name])
            except KeyError:
                raise KeyError(f'[ERROR] instance.__dict__: {instance.__dict__} \n'
                               f'This might happen if argument is not registered by '
                               f'member variable of instance.')
        return args

    def _set_local_optimizers(self, optimizer, param_groups_func=None):
        """
        NOTE: Even main optimizer only updates globally aggregated gradients,
        optimizer.zero_grad() is efficient for parallel_compute().
        That's why we keep the individual optimizer for each local model.
        """
        if not issubclass(type(optimizer), torch.optim.Optimizer):
            raise TypeError(
                f'To set local optimizers for copy (use _set_local_optimizers()), original optimizer type: '
                f'{type(optimizer)} '
                f'must be sub-class of torch.optim.Optimizer')
        if optimizer is None:
            raise TypeError(f"Argument optimizer must be configured, but {optimizer}")

        self.param_groups_func = param_groups_func
        is_set_by_user = (len(self.local_optimizers) == self.num_models)
        if not is_set_by_user:
            self.local_optimizers = [optimizer]
            for idx in range(1, self.num_models):
                if self.param_groups_func:
                    params = self.param_groups_func(self.original_local_models[idx])
                else:
                    params = self.original_local_models[idx].parameters()
                args = self._get_required_args_value(optimizer)
                # https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
                cls = type(optimizer.__class__.__name__, (optimizer.__class__,), dict(optimizer.__dict__))
                copied_optimizer = cls(params, lr=optimizer.defaults['lr'], *args)
                copied_optimizer.load_state_dict(optimizer.state_dict())
                self.local_optimizers.append(copied_optimizer)
        for optimizer in self.local_optimizers:
            optimizer.zero_grad()

    def _set_local_schedulers(self, scheduler=None):
        # LR scheduler is optional
        if scheduler is not None:
            is_set_by_user = (len(self.local_schedulers) == self.num_models)
            if not is_set_by_user:
                self.local_schedulers = [scheduler]

    @property
    def main_stream(self):
        return self.model_streams[0]

    @property
    def eval_model(self):
        if type(self.local_models[0]) == torch.nn.parallel.IndependentIdenticalDataParallel:
            return self.local_models[0].module
        else:
            return self.local_models[0]

    @property
    def main_model(self):
        return self.local_models[0]

    @property
    def main_optimizer(self):
        return self.local_optimizers[0]

    @property
    def main_scheduler(self):
        return self.local_schedulers[0] if self.local_schedulers is not None else None

    @property
    def num_local_models(self):
        return self.num_models

    @property
    def epoch(self):
        return self.epoch_iterator.epoch

    @epoch.setter
    def epoch(self, epoch):
        self.epoch_iterator.epoch = epoch

    def remaining_epochs(self, final_epochs):
        self.epoch_iterator.final_epochs = final_epochs
        try:
            for epoch in self.epoch_iterator.__iter__():
                yield epoch
        finally:
            self.print_final_results()

    def print_final_results(self):
        self.trainer_print(f'Total epoch time (sec): {self.total_epoch_time}')
        self.trainer_print(f'Total epoch time: {datetime.timedelta(seconds=self.total_epoch_time)}')

    def set_model_train(self):
        for local_model in self.local_models:
            local_model.train()

    def _create_model_streams(self):
        for _ in range(self.num_models):
            self.model_streams.append(torch.cuda.Stream())

    def _create_stream_for_optimizer(self):
        self.optimizer_stream = torch.cuda.Stream()

    @contextmanager
    def accum_processing(self):
        if dist.is_initialized() and self.local_accum_step == -1:
            self.prev_require_forward_param_sync = self.main_model.require_forward_param_sync
            def _forward_model(model, stream):
                with torch.cuda.stream(stream):
                    if model.require_forward_param_sync:
                        model._sync_params()
                        model.require_forward_param_sync = False

            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_forward_model,
                                        args=(self.local_models[idx], self.model_streams[idx],))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        try:
            yield
        finally:
            self.local_accum_step += 1
            if dist.is_initialized() and self.local_accum_step == self.accum_step:
                for model in self.local_models:
                    model.require_forward_param_sync = self.prev_require_forward_param_sync

    def _compute_forward_and_loss(self, model, criterion, input, target):
        if self.output_as_loss:
            model_to_inspect = model.module if type(model) == torch.nn.parallel.IndependentIdenticalDataParallel else model
            args_of_model = inspect.getfullargspec(getattr(model_to_inspect, 'forward')).args
            if 'self' in args_of_model:
                args_of_model.remove('self')
            num_args = len(args_of_model)
            if num_args > 2:
                loss = criterion(model(*input, target))
            else:
                loss = criterion(model(input, target))
        else:
            if isinstance(input, (tuple, list)):
                output = model(*input)
            else:
                output = model(input)
            loss = criterion(output, target)
        return loss

    def _prepare_hooks_for_local_models(self, hook):
        def dummy_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result(bucket.get_tensors())
            return fut

        for model_idx in range(self.num_models):
            if model_idx == 0:
                self.hooks.append(hook)
            else:
                self.hooks.append(dummy_hook)

    def _check_overlap_with_ddp(self):
        if not self.prepared_for_ddp:
            raise ValueError(
                "DDP instance must be prepared with self.weight_sync_method = overlap")

        for param_indices in self.ddp_bucket_indices:
            if param_indices != sorted(param_indices):
                raise RuntimeError(
                    "Parameter indices in each bucket must be sorted with self.weight_sync_method = overlap")

        if self.hooks is None:
            raise ValueError(
                "hooks must be prepared with self.weight_sync_method = overlap")
        elif not isinstance(self.hooks, (list, tuple)):
            raise ValueError(
                f"Argument hooks must be list or tuple type: {type(self.hooks)}")
        elif len(self.hooks) != self.num_models:
            raise ValueError(f"Number of hooks: {len(self.hooks)} "
                             f"must be equal to "
                             f"number of local models : {self.num_models}")
        for hook in self.hooks:
            if not callable(hook):
                raise TypeError("hook must be callable.")

        if self.optimizer_stream is None:
            raise ValueError(
                "optimizer_stream must be assigned with self.weight_sync_method = overlap")

    def trainer_print(self, message, status='info'):
        print_msg = f'[{status.upper()}][{self.__class__.__name__}] {message}'
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(print_msg)
        else:
            print(print_msg)

    @contextmanager
    def measure_epoch_time(self):
        try:
            start_time = time.time()
            yield
        finally:
            self.elapsed_time = int(time.time() - start_time)
            self.trainer_print(f'Epoch time: {self.elapsed_time}')
            self.total_epoch_time += self.elapsed_time

    @contextmanager
    def record_epoch_data(self):
        try:
            yield
        finally:
            self.trainer_print(f'record at epoch: {self.epoch} | iterations: {self.sync_step} | loss: {self.losses[0]:.3f}')

    def state_dict(self):
        trainer_state_dict = {}
        if self.local_schedulers:
            scheduler_state = self.main_scheduler.state_dict()
        else:
            scheduler_state = None
        trainer_state_dict.update({
            'epoch': self.epoch,
            'total_epoch_time': self.total_epoch_time,
            'step': self.sync_step,
            'model' : self.main_model.module.state_dict(),
            'optimizer'  : self.main_optimizer.state_dict(),
            'scheduler'  : scheduler_state,
        })

        # NOTE: To confirm the consistency of the last saved configuration
        trainer_state_dict['global_batch_size'] = self.global_batch_size
        trainer_state_dict['local_batch_size'] = self.local_batch_size

        assert len(self.all_num_models_in_process_group) == len(self.all_accum_step_in_process_group)
        iidp_config_map_in_cluster = {}
        for rank, (num_models, accum_step) in enumerate(
                zip(self.all_num_models_in_process_group, self.all_accum_step_in_process_group)):
            iidp_config_map_in_cluster[rank] = (num_models, accum_step)
        trainer_state_dict['iidp_config_map_in_cluster'] = iidp_config_map_in_cluster

        return trainer_state_dict

    def load_state_dict(self, state_dict, restrict_saved_config=False):
        for local_model in self.local_models:
            local_model.module.load_state_dict(state_dict['model'])
        for local_optimizer in self.local_optimizers:
            local_optimizer.load_state_dict(state_dict['optimizer'])
        for local_scheduler in self.local_schedulers:
            local_scheduler.load_state_dict(state_dict['scheduler'])
        self.epoch = state_dict['epoch']
        self.total_epoch_time = state_dict['total_epoch_time']
        self.sync_step = state_dict['step']

        if restrict_saved_config:
            # [CHECK 1] Confirm the resume resource setup is equal to the previous saved one
            print(f'[load_state_dict] The saved IIDP config on cluster setup: {state_dict["iidp_config_map_in_cluster"]}')
            all_ranks = list(state_dict['iidp_config_map_in_cluster'].keys())
            if len(all_ranks) != dist.get_world_size():
                raise ValueError(
                    f'[load_state_dict] Current number of GPUs: {dist.get_world_size()} '
                    f'is not equal to the saved number of GPUs: {len(all_ranks)}')
            global_batch_size_in_current_cluster = 0
            for _, (num_models, accum_step) in state_dict['iidp_config_map_in_cluster'].items():
                global_batch_size_in_current_cluster += int(state_dict['local_batch_size'] * (num_models*(accum_step+1)))
            if global_batch_size_in_current_cluster != state_dict['global_batch_size']:
                raise ValueError(
                    f'[load_state_dict] The saved global batch size: {state_dict["global_batch_size"]} '
                    f'is not equal to the global batch size on the cluster : {global_batch_size_in_current_cluster}')

            # [CHECK 2] Confirm the resume IIDP configuration is equal to the previous saved one
            assert self.global_batch_size == state_dict['global_batch_size'], \
                f"self.global_batch_size: {self.global_batch_size} | state_dict['global_batch_size']: {state_dict['global_batch_size']}"
            assert self.local_batch_size == state_dict['local_batch_size'], \
                f"self.local_batch_size: {self.local_batch_size} | state_dict['local_batch_size']: {state_dict['local_batch_size']}"
            assert self.num_models == state_dict['iidp_config_map_in_cluster'][dist.get_rank()][0], \
                f"self.num_models: {self.num_models} | saved num_models: {state_dict['iidp_config_map_in_cluster'][dist.get_rank()][0]}"
            assert self.accum_step == state_dict['iidp_config_map_in_cluster'][dist.get_rank()][1], \
                f"self.accum_step: {self.accum_step} | saved accum_step: {state_dict['iidp_config_map_in_cluster'][dist.get_rank()][1]}"