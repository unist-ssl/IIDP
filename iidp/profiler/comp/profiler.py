import os
import json
import socket
import math
import time

import torch

from iidp.profiler.comp.profile_data_schema import CompProfileData, CompProfileJSONData
from iidp.utils.json_utils import read_json
from iidp.config.config_utils import sorted_listdir
from iidp.profiler.memory.profile_utils import get_max_num_models_for_static_lbs


SLEEP_TIME = 10


class ComputationProfiler(object):
    def __init__(self, profiler_instance, profile_dir=None):
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()

        if profiler_instance is None:
            raise ValueError('Argument profiler_instance must be configured.')
        self.profiler_instance = profiler_instance
        if not hasattr(self.profiler_instance, 'model_name') or \
            not hasattr(self.profiler_instance, 'lbs') or \
                not hasattr(self.profiler_instance, 'num_models') or \
                    not hasattr(self.profiler_instance, 'profile_data'):
            raise AttributeError(f'[{self.__class__.__name__}] {self.profiler_instance.__dict__}')
        if not isinstance(self.profiler_instance.profile_data, CompProfileData):
            raise TypeError(f'[{self.__class__.__name__}] '
                            f'Type of self.profiler_instance.profile_data must be CompProfileData, '
                            'but {type(self.profiler_instance.profile_data)}')
        if self.profiler_instance.model_name is None or self.profiler_instance.lbs is None or \
            self.profiler_instance.num_models is None:
                raise ValueError(f'[{self.__class__.__name__}] '
                                 f'model_name: {self.profiler_instance.model_name} | '
                                 f'lbs: {self.profiler_instance.lbs} | '
                                 f'num_models: {self.profiler_instance.num_models}')
        self.model_name = self.profiler_instance.model_name
        self.lbs = self.profiler_instance.lbs
        self.num_models = self.profiler_instance.num_models
        self.accum_step = 0
        self.weight_sync_method = 'sequential'

        self.hostname = socket.gethostname()
        self.gpu_type = torch.cuda.get_device_name()
        self.profile_dir = profile_dir

        self.profile_json_data = CompProfileJSONData(
            self.model_name, self.gpu_type, self.lbs, self.num_models)

    def run(self):
        self.profiler_instance.run()
        self.profile_json_data.update(self.profiler_instance.profile_data,
                                      self.profiler_instance.auxiliary_profile_data)
        if self.profile_dir:
            self.record_profile_data()

    def record_profile_data(self):
        self.profile_dir = os.path.join(
                self.profile_dir, str(self.lbs), self.hostname)
        os.makedirs(self.profile_dir, exist_ok=True)
        json_file = os.path.join(
            self.profile_dir,
            f'{self.hostname}_{self.model_name}_{self.lbs}_{self.num_models}_comp_profile.json'
        )
        try:
            with open(json_file, 'w') as jf:
                json_str = json.dumps(self.profile_json_data.dict)
                jf.write(json_str)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

        # Test to confirm write json object to file
        json_data = read_json(json_file)
        print(json_data)


class ComputationProfilerDriver(object):
    def __init__(self, memory_profile_dir, user_defined_cmd, gpu_resuse_pause_time=300):
        if not os.path.exists(memory_profile_dir):
            raise ValueError(f'Argument memory_profile_dir = {memory_profile_dir} must exist')
        if not isinstance(gpu_resuse_pause_time, int):
            raise TypeError(f'Argument gpu_resuse_pause_time type must be integer, but {type(gpu_resuse_pause_time)}')
        if gpu_resuse_pause_time < 0:
            raise TypeError(f'Argument gpu_resuse_pause_time value must >= 0, but {gpu_resuse_pause_time}')
        self.memory_profile_dir = memory_profile_dir
        self.user_defined_cmd = user_defined_cmd
        self.gpu_resuse_pause_time = gpu_resuse_pause_time

    def run(self):
        max_gpus_in_server = torch.cuda.device_count()
        gpu_id = 0
        for lbs in sorted_listdir(self.memory_profile_dir):
            max_num_models = get_max_num_models_for_static_lbs(self.memory_profile_dir, lbs)
            for num_models in self._get_list_of_profile_num_models(max_num_models):
                args = {
                    'gpu_id': int(gpu_id),
                    'local_batch_size': int(lbs),
                    'num_models': int(num_models)
                }
                command = self.user_defined_cmd % args
                print(f'[INFO] command: {command}')
                os.system(command)
                if SLEEP_TIME > 0:
                    print(f'[INFO] sleep {SLEEP_TIME} sec ..')
                    time.sleep(SLEEP_TIME)
                gpu_id += 1
                if gpu_id == max_gpus_in_server:
                    if self.gpu_resuse_pause_time > 0:
                        print(f'[INFO] sleep {self.gpu_resuse_pause_time} sec for GPU reuse ..')
                        time.sleep(self.gpu_resuse_pause_time)
                gpu_id = gpu_id % max_gpus_in_server

    def _get_list_of_profile_num_models(self, max_num_models):
        if max_num_models <= 0:
            raise ValueError(f'[ERROR] Argument max_num_models must be > 0, but {max_num_models}')
        if max_num_models == 1:
            return [max_num_models]
        mid_num_models = 0
        if max_num_models > 3:
            mid_num_models = math.floor(max_num_models/2)
        if mid_num_models != 0:
            return [1, mid_num_models, max_num_models]
        else:
            return [1, max_num_models]