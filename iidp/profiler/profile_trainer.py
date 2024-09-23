import threading

import torch

from iidp.train.train_helper import TrainerHelper


class IIDPSingleGPUProfileTrainer(TrainerHelper):
    def __init__(self, gpu, local_batch_size, num_models, accum_step, weight_sync_method='sequential'):
        if weight_sync_method != 'sequential':
            raise ValueError(f'[ERROR][{self.__class__.__name__}] '
                             f'Not support other weight sync method except sequential')
        super().__init__(gpu, local_batch_size, num_models, accum_step, weight_sync_method)

    def prepare_stream_parallel(self, model, criterion):
        self._create_model_streams()
        self._set_original_local_models(model)
        self._set_criterion(criterion)
        self.local_models = self.original_local_models

    def prepare_weight_sync_method(self, optimizer, scheduler=None, param_groups_func=None):
        self._set_local_optimizers(optimizer, param_groups_func)
        self._set_local_schedulers(scheduler)

    def profile_parallel_compute(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        # reference: https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread
        class ParallelComputeThread(threading.Thread):
            def run(self):
                self._exc = None
                try:
                    super().run()
                except Exception as e:
                    self._exc = e

            def join(self, timeout=None):
                super().join(timeout=timeout)
                if self._exc:
                    raise self._exc

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion, optimizer):
            if index == 0:
                fwd_start.record()
            with torch.cuda.stream(stream):
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                with lock:
                    self.losses[index] = loss
                if not self.is_accum_mode:
                    optimizer.zero_grad()
                if index == self.num_models-1:
                    fwd_end.record()
                if index == 0:
                    bwd_start.record()
                loss.backward(model_index=index)

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(ParallelComputeThread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx],
                                            self.criterion, self.local_optimizers[idx]))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                try:
                    thread.join()
                except RuntimeError as e:
                    raise RuntimeError(e)
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx],
                          self.criterion, self.local_optimizers[idx])
        if idx == self.num_models-1:
            bwd_end.record()
        torch.cuda.synchronize()
        return fwd_start.elapsed_time(fwd_end), bwd_start.elapsed_time(bwd_end)

    def profile_step(self):
        update_start = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.cuda.stream(self.main_stream):
            update_start.record()
            self.main_optimizer.step()
            update_end.record()
        torch.cuda.synchronize()
        if self.num_models == 1:
            copy_start.record()
        for idx in range(1, self.num_models):
            stream = self.model_streams[idx]
            with torch.cuda.stream(stream):
                if idx == 1:
                    copy_start.record()
                for src_param, dst_param in \
                        zip(self.main_model.parameters(), self.local_models[idx].parameters()):
                    dst_param.data.copy_(src_param.data)
        if self.num_models == 1 or idx == self.num_models-1:
            copy_end.record()
        torch.cuda.synchronize()
        return update_start.elapsed_time(update_end), copy_start.elapsed_time(copy_end)

    def profile_parallel_forward(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion):
            with torch.cuda.stream(stream):
                if index == 0:
                    fwd_start.record()
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
        if idx == self.num_models-1:
            fwd_end.record()
        torch.cuda.synchronize()
        return fwd_start.elapsed_time(fwd_end)

    def profile_parallel_backward(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion, optimizer):
            with torch.cuda.stream(stream):
                if index == 0:
                    bwd_start.record()
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
        if idx == self.num_models-1:
            bwd_end.record()
        torch.cuda.synchronize()
        return bwd_start.elapsed_time(bwd_end)