import torch
import torchvision

from iidp.profiler import DDPHelper, CUDAEventTimer


class ImageNetProfiler(DDPHelper):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = torchvision.models.__dict__[self.model_name]().to(self.gpu)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.gpu)
        self.lbs = 32 # Not important value

        if self.model_name.startswith('inception'):
            self.image_size = (3, 299, 299)
        else:
            self.image_size = (3, 224, 224)

    def _get_ddp_bucket_indices(self):
        self.ddp_module = torch.nn.parallel.IndependentIdenticalDataParallel(
            self.model, device_ids=[self.gpu], output_device=[self.gpu],
            model_index=0, num_local_models=1, total_num_models=1)
        self.ddp_module.train()
        input_shape = [self.lbs, *self.image_size]
        print(f'[DDPHelper] step: {self.step}')
        for step in range(self.step):
            dummy_images = torch.randn(*input_shape).cuda(self.gpu, non_blocking=True)
            dummy_targets = torch.empty(self.lbs, dtype=torch.int64).random_(1000).cuda(self.gpu, non_blocking=True)
            is_verbose = (step >= 1)
            with CUDAEventTimer('[Profile info] DDP forward (+BN sync) time', verbose=is_verbose) as timer:
                output = self.ddp_module(dummy_images)
            loss = self.criterion(output, dummy_targets)
            loss.backward()

    def run(self):
        self.get_bucket_size_distribution()