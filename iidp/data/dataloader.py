import torch
import torch.distributed as dist

from iidp.data.sampler import ImbalancedSampler
from iidp.train import GLOBAL_TRAINER_STATE, LOCAL_TRAINER_STATE


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, batch_fn=None, loading_once=None, **kwargs):
        if kwargs.get("batch_sampler") is not None:
            sampler = getattr(kwargs.get("batch_sampler"), 'sampler', None)
        elif kwargs.get("sampler") is not None:
            sampler = kwargs.get("sampler")
        else:
            sampler = None
        if sampler is None or type(sampler) is torch.utils.data.distributed.DistributedSampler:
            if dist.is_initialized():
                imblanced_sampler = ImbalancedSampler(
                    dataset, partition_size=GLOBAL_TRAINER_STATE.partition_size)
                if kwargs.get("batch_sampler") is not None:
                    kwargs.get("batch_sampler").sampler = imblanced_sampler
                if kwargs.get("sampler") is not None:
                    kwargs["sampler"] = imblanced_sampler
        if (kwargs.get("num_workers") is not None and kwargs.get("num_workers") > 0) or \
            (kwargs.get("persistent_workers") is not None and kwargs.get("persistent_workers") is True):
            persistent_workers = True
        else:
            persistent_workers = False
        super().__init__(dataset, batch_size, persistent_workers=persistent_workers, **kwargs)
        self.initial_dataloader_length = super().__len__() // (LOCAL_TRAINER_STATE.accum_step+1) # Equal dataloader length among all ranks
        if batch_fn is None:
            raise ValueError(f'Argument "batch_fn" must be configured by user, but: {batch_fn}')
        if loading_once is None:
            raise ValueError(f'Argument "loading_once" must be configured by user, but: {loading_once}')
        self.batch_fn = batch_fn
        self.loading_once = loading_once
        self.global_batch_size = GLOBAL_TRAINER_STATE.global_batch_size
        self.total_local_num_models = LOCAL_TRAINER_STATE.num_models
        self.accum_step = LOCAL_TRAINER_STATE.accum_step
        self.data_index = -1
        self.done = False
        self.epoch = 0

    def __iter__(self):
        self.data_index = 0
        self.step_index = 0
        self.done = False
        num_yielded = 0

        if self.loading_once is True:
            while not self.done:
                print(f'[INFO][iidp.data.DataLoader] Initial loading.. it might take time..')
                for idx, batch in enumerate(super().__iter__()):
                    chunked_batch = self.batch_fn(batch, self.total_local_num_models, self.loading_once)
                    yield chunked_batch
                    num_yielded += 1
                    if num_yielded % (self.accum_step+1) == 0:
                        self.data_index += self.global_batch_size
                        self.step_index += 1
                    if self.data_index >= len(self.dataset):
                        self.done = True
                        break
        else:
            # NOTE: Since self._index_sampler.batch_size is changed to local batch size,
            # len(super().__iter__()) is also modified.
            self._index_sampler.batch_size = LOCAL_TRAINER_STATE.local_batch_size
            local_batch_data = []
            while not self.done:
                print(f'[INFO][iidp.data.DataLoader] Initial loading.. it might take time..')
                for idx, batch in enumerate(super().__iter__()):
                    if len(local_batch_data) < self.total_local_num_models:
                        local_batch_data.append(batch)

                    if len(local_batch_data) == self.total_local_num_models:
                        #assert len(local_batch_data) == total_local_num_models
                        chunked_batch = self.batch_fn(local_batch_data, self.total_local_num_models, self.loading_once)
                        yield chunked_batch
                        num_yielded += 1
                        local_batch_data = []
                        if num_yielded % (self.accum_step+1) == 0:
                            self.data_index += self.global_batch_size
                            self.step_index += 1
                    if self.data_index >= len(self.dataset):
                        self.done = True
                        break

        if self.done is False:
            raise RuntimeError(f'[ERROR][iidp.data.DataLoader] Flag done is not True even iterator is finished')
        self.epoch += 1

    def __len__(self):
        return self.initial_dataloader_length

    def state_dict(self):
        return {
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        if hasattr(self._index_sampler.sampler, 'epoch'):
            self._index_sampler.sampler.epoch = self.epoch