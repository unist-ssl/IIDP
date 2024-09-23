import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import torch

import threading

import iidp
from iidp.utils.global_vars import CHECKPOINT_FILE_NAME, WEIGHT_SYNC_METHOD_SELECTION_THRESHOLD
from iidp.train.train_helper import TrainerHelper, calculate_bucket_size_distribution, \
                                     select_weight_sync_method


class IIDPTrainer(TrainerHelper):
    def __init__(self, gpu, local_batch_size, num_models, accum_step, weight_sync_method='recommend'):
        super().__init__(gpu, local_batch_size, num_models, accum_step, weight_sync_method)

    def prepare_stream_parallel(self, model, criterion, **kwargs):
        gradient_as_bucket_view = kwargs.get('gradient_as_bucket_view') or False
        find_unused_parameters = kwargs.get('find_unused_parameters') or False
        self._create_model_streams()
        self._set_original_local_models(model)
        self._set_criterion(criterion)

        for idx, original_model in enumerate(self.original_local_models):
            # Assign buckets for DDP to model's stream to synchronize copy in all-reduce
            with torch.cuda.stream(self.model_streams[idx]):
                local_ddp_module = torch.nn.parallel.IndependentIdenticalDataParallel(
                                    original_model, device_ids=[self.gpu], output_device=[self.gpu],
                                    find_unused_parameters=find_unused_parameters,
                                    gradient_as_bucket_view=gradient_as_bucket_view,
                                    model_index=idx, num_local_models=self.num_models,
                                    total_num_models=self.total_num_decoupled_workers,
                                    sync_buffer_barrier=self._sync_buffer_barrier)
                self.local_models.append(local_ddp_module)
        # It is used for overlapping optimizer with torch.nn.parallel.IndependentIdenticalDataParallel
        self.ddp_bucket_indices = self.main_model.bucket_indices
        self.prepared_for_ddp = True

        if self.weight_sync_method == 'recommend':
            self._recommend_weight_sync_method()

    def prepare_weight_sync_method(self, optimizer, scheduler=None, param_groups_func=None):
        self._set_local_optimizers(optimizer, param_groups_func)
        self._set_local_schedulers(scheduler)
        if self.weight_sync_method == 'overlap':
            if self.prepared_for_ddp:
                self._prepare_overlap_optimizer_with_ddp()
            else:
                raise RuntimeError("[ERROR] Without DDP, overlap optimizer cannot work")

    def _recommend_weight_sync_method(self):
        bukcet_capacity = self.main_model.bucket_bytes_cap / (1024 * 1024) # MB
        bucket_size_distribution = calculate_bucket_size_distribution(self.main_model)

        weight_sync_method = select_weight_sync_method(bucket_size_distribution, bukcet_capacity)

        self.weight_sync_method = weight_sync_method

    def parallel_compute(self, scatter_input, scatter_target, accum_step=-1):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")
        if self.is_accum_mode and self.prepared_for_ddp:
            self.seq_parallel_compute(scatter_input, scatter_target, accum_step)
            return

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion, optimizer):
            with torch.cuda.stream(stream):
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                with lock:
                    self.losses[index] = loss
                if not self.is_accum_mode:
                    optimizer.zero_grad()
                loss.backward(model_index=index)

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx],
                                            self.criterion, self.local_optimizers[idx]))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx],
                          self.criterion, self.local_optimizers[idx])

    def seq_parallel_compute(self, scatter_input, scatter_target, accum_step=-1):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")
        if self.max_accum_step <= 0:
            raise RuntimeError('If self.max_accum_step <= 0, seq_parallel_compute() must not be called')

        lock = threading.Lock()
        def _local_accum_worker(index, model, stream, input, target, criterion):
            with torch.cuda.stream(stream):
                with model.no_sync():
                    loss = self._compute_forward_and_loss(model, criterion, input, target)
                    with lock:
                        self.losses[index] = loss
                    loss.backward(model_index=index)

        def _local_sync_worker(index, model, stream, input, target, criterion):
            with torch.cuda.stream(stream):
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                with lock:
                    self.losses[index] = loss
                self.sync_accum_barrier.wait()
                loss.backward(model_index=index)

        if self.num_models > 1:
            if accum_step < self.accum_step - 1:
                threads = []
                for idx in range(self.num_models):
                    threads.append(threading.Thread(target=_local_accum_worker,
                                            args=(idx, self.local_models[idx], self.model_streams[idx],
                                                scatter_input[idx], scatter_target[idx],
                                                self.criterion))
                                )
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            else:
                threads = []
                for idx in range(self.num_models):
                    threads.append(threading.Thread(target=_local_sync_worker,
                                            args=(idx, self.local_models[idx], self.model_streams[idx],
                                                scatter_input[idx], scatter_target[idx],
                                                self.criterion))
                                )
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
        else:
            idx = 0
            if accum_step < self.accum_step - 1:
                _local_accum_worker(idx, self.local_models[idx], self.model_streams[idx],
                                    scatter_input[idx], scatter_target[idx],
                                    self.criterion)
            else:
                _local_sync_worker(idx, self.local_models[idx], self.model_streams[idx],
                                    scatter_input[idx], scatter_target[idx],
                                    self.criterion)

    def compute(self, data):
        if self.is_accum_mode and self.prepared_for_ddp:
            parallel_input = [data[i][0] for i in range(self.num_models)]
            parallel_target = [data[i][1] for i in range(self.num_models)]
            with self.accum_processing():
                self.parallel_compute(parallel_input, parallel_target, self.local_accum_step)
        else:
            parallel_input = [data[i][0] for i in range(self.num_models)]
            parallel_target = [data[i][1] for i in range(self.num_models)]
            self.parallel_compute(parallel_input, parallel_target)

    def parallel_forward(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion):
            with torch.cuda.stream(stream):
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                with lock:
                    self.losses[index] = loss

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx], self.criterion))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx], self.criterion)

    def parallel_backward(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion, optimizer):
            with torch.cuda.stream(stream):
                loss = self.losses[index]
                if not self.is_accum_mode:
                    optimizer.zero_grad()
                loss.backward(model_index=index)

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx],
                                            self.criterion, self.local_optimizers[idx]))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx],
                          self.criterion, self.local_optimizers[idx])

    def _create_optimizer_hook(self, hook):
        def hook_rebuild_bucket_setup():
            if self.main_model._has_rebuilt_buckets and not self.is_rebuilt_ddp_bucket_indices:
                self.ddp_bucket_indices = self.main_model.bucket_indices
                calculate_bucket_size_distribution(self.main_model)
                self.is_rebuilt_ddp_bucket_indices = True

        def hook_with_optimizer_step(state, bucket):
            future_work = hook(state, bucket)
            hook_rebuild_bucket_setup()
            def optimizer_step(fut: torch.futures.Future):
                bucket_index = bucket.get_index()
                param_indices = self.ddp_bucket_indices[bucket_index]
                nccl_stream = torch.cuda.current_stream()
                self.optimizer_stream.wait_stream(nccl_stream)
                with torch.cuda.stream(self.optimizer_stream):
                    gradients = bucket.get_gradients()
                    for index, grad in zip(param_indices, gradients):
                        grad.index = index
                    self._optimizer_step(gradients, param_indices)
                return bucket.get_tensors()

            return future_work.then(optimizer_step)
        return hook_with_optimizer_step

    def _prepare_overlap_optimizer_with_ddp(self):
        hook = iidp.ddp_comm_hooks.iidp_allreduce_hook
        self._create_stream_for_optimizer()
        self._prepare_hooks_for_local_models(hook)
        self._check_overlap_with_ddp()
        for i, (local_model, hook) in enumerate(zip(self.local_models, self.hooks)):
            if i == 0:
                state = iidp.ddp_comm_hooks.IIDPState(None, self.total_num_decoupled_workers)
                hook = self._create_optimizer_hook(hook)
            else:
                state = None
            local_model.register_comm_hook(state=state, hook=hook)

    def _optimizer_step(self, gradients, param_indices):
        if not self.weight_sync_method == 'overlap':
            raise RuntimeError("This function must be called if weight_sync_method is overlap")

        self.main_optimizer.step(gradients)
        # Partial weight copy
        partial_src_params_to_copy = [self.main_model.ddp_register_params[i] for i in param_indices]
        for idx in range(1, self.num_models):
            partial_dst_params_to_copy = [self.local_models[idx].ddp_register_params[i] for i in param_indices]
            for src_param, dst_param in \
                    zip(partial_src_params_to_copy, partial_dst_params_to_copy):
                dst_param.data.copy_(src_param.data)

    def is_sync_step(self):
        if self.is_accum_mode and self.local_accum_step < self.accum_step:
            return False
        else:
            return True

    def step(self):
        if self.is_accum_mode and self.local_accum_step < self.accum_step:
            # NOTE: Synchronize multi-stream before next computation
            # to avoid RuntimeError: CUDA error: device-side assert triggered
            torch.cuda.synchronize()
            return False
        if self.weight_sync_method == 'overlap':
            if self.is_accum_mode:
                with torch.cuda.stream(self.optimizer_stream):
                    self.main_optimizer.zero_grad()
                for idx in range(1, self.num_models):
                    stream = self.model_streams[idx]
                    optimizer = self.local_optimizers[idx]
                    with torch.cuda.stream(stream):
                        optimizer.zero_grad()
            torch.cuda.synchronize()

        elif self.weight_sync_method == 'sequential':
            with torch.cuda.stream(self.main_stream):
                self.main_optimizer.step()
                if self.is_accum_mode:
                    self.main_optimizer.zero_grad()
            torch.cuda.synchronize()
            for idx in range(1, self.num_models):
                stream = self.model_streams[idx]
                optimizer = self.local_optimizers[idx]
                with torch.cuda.stream(stream):
                    for src_param, dst_param in \
                            zip(self.main_model.parameters(), self.local_models[idx].parameters()):
                        dst_param.data.copy_(src_param.data)
                    if self.is_accum_mode:
                        optimizer.zero_grad()
            torch.cuda.synchronize()

        else:
            raise RuntimeError(f'Not support weight_sync_method: {self.weight_sync_method}')
        self.sync_step += 1
        self.local_accum_step = -1
        return True

    def scheduler_step(self):
        if self.local_schedulers:
            if self.weight_sync_method == 'overlap':
                with torch.cuda.stream(self.optimizer_stream):
                    self.main_scheduler.step()
            elif self.weight_sync_method == 'sequential':
                with torch.cuda.stream(self.main_stream):
                    self.main_scheduler.step()
            else:
                raise RuntimeError(f'Not support weight_sync_method: {self.weight_sync_method}')

    def save(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path): # checkpoint_path is directory
            checkpoint_dir = checkpoint_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILE_NAME)
        else: # checkpoint_path is file
            checkpoint_dir = os.path.dirname(checkpoint_path)

        trainer_state_dict = self.state_dict()

        self.trainer_print(f'Save trainer state to {checkpoint_path}')
        torch.save(trainer_state_dict, checkpoint_path)

    def load(self, checkpoint_path, restrict_saved_config=False):
        if not os.path.exists(checkpoint_path):
            raise FileExistsError(f'Checkpoint path: {checkpoint_path} does not exist')
        if not os.path.isfile(checkpoint_path): # checkpoint_path is directory
            checkpoint_dir = checkpoint_path
            checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILE_NAME)
        else: # checkpoint_path is file
            checkpoint_dir = os.path.dirname(checkpoint_path)

        self.trainer_print(f'Load trainer state from {checkpoint_path}')
        loc = 'cuda:{}'.format(self.gpu) if type(self.gpu) == int else self.gpu
        state_dict = torch.load(checkpoint_path, map_location=loc)
        self.load_state_dict(state_dict, restrict_saved_config)