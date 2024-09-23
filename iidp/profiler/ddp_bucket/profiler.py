import os
import json

import torch
import torch.distributed as dist

from iidp.utils.json_utils import read_json


class DDPBucketProfiler(object):
    def __init__(self, profiler_instance, profile_dir=None):
        torch.cuda.empty_cache()
        if not dist.is_initialized():
            torch.cuda.set_device(0)
            dist.init_process_group(
                backend='nccl', init_method='tcp://127.0.0.1:22222', world_size=1, rank=0)

        if profiler_instance is None:
            raise ValueError('Argument profiler_instance must be configured.')
        self.profiler_instance = profiler_instance
        self.model_name = self.profiler_instance.model_name

        self.profile_dir = profile_dir

        self.profile_data = {
            'model': self.model_name,
            'bucket_size_distribution': []
        }

    def run(self):
        self.profiler_instance.run()
        self.profile_data['bucket_size_distribution'] = self.profiler_instance.bucket_size_distribution
        if dist.get_rank() == 0:
            if self.profile_dir:
                self.record_profile_data()

    def record_profile_data(self):
        os.makedirs(self.profile_dir, exist_ok=True)
        json_file = os.path.join(
            self.profile_dir,
            f'{self.model_name}_bucket_size_profile.json'
        )
        try:
            with open(json_file, 'w') as jf:
                json_str = json.dumps(self.profile_data)
                jf.write(json_str)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

        # Test to confirm write json object to file
        json_data = read_json(json_file)
        print(json_data)