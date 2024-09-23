import torch
import torchvision

from iidp.profiler import IIDPCustomProfilerHelper


class ImageNetProfiler(IIDPCustomProfilerHelper):
    def __init__(self, lbs, num_models, model_name):
        super().__init__(lbs, num_models)

        self.model_name = model_name
        self.model = torchvision.models.__dict__[self.model_name]().to(self.gpu)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.gpu)

        if self.model_name.startswith('inception'):
            self.image_size = (3, 299, 299)
        else:
            self.image_size = (3, 224, 224)

        self.prepare()

        if self.model_name == 'resnet50':
            self.auxiliary_profile_data = {'bn_sync_time': 6.29}
        else: # TODO
            self.auxiliary_profile_data = {'bn_sync_time': 0}

    def set_optimizer(self):
        model = self.trainer.main_model
        self.optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                         momentum=0.9,
                                         weight_decay=0.004)

    def run(self):
        print(f'====> Run IIDP profiler with the number of VSWs: {self.num_models}')
        self.trainer.set_model_train()
        input_shape = [self.trainer.batch_size_per_gpu, *self.image_size]
        for i in range(self.warmup_step+self.num_minibatches):
            dummy_images = torch.randn(*input_shape)
            dummy_targets = torch.empty(self.trainer.batch_size_per_gpu, dtype=torch.int64).random_(1000)
            with self.record_cuda_time():
                images = dummy_images.cuda(self.gpu, non_blocking=True)
                target = dummy_targets.cuda(self.gpu, non_blocking=True)

                scatter_images = torch.chunk(images, self.trainer.num_local_models)
                scatter_targets = torch.chunk(target, self.trainer.num_local_models)
            data_time = self.cuda_time
            fwd_time, bwd_time = self.trainer.profile_parallel_compute(scatter_images, scatter_targets)
            update_time, copy_time = self.trainer.profile_step()

            if i >= self.warmup_step:
                total_time = data_time + fwd_time + bwd_time + update_time + copy_time
                self.profile_data.update(data_time, fwd_time, bwd_time, update_time, copy_time, total_time)
                if i % 10 == 0:
                    print(f'[step {i}] {self.profile_data}')

        print(self.profile_data)
