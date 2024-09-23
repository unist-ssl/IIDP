import os
import json
import socket
import time
import math

import asyncio
from contextlib import contextmanager

import torch

from iidp.utils.json_utils import read_json
from iidp.utils.global_vars import MAX_MEM_PROFILE_FILE_NAME

from iidp.profiler.memory.profile_data_schema import MemoryProfileJSONData
from iidp.profiler.memory.profile_helper import nvidia_smi_memory_monitoring, async_run_command


TIMEOUT = 60
MEM_UTIL_THRESHOLD = 100
MIN_NUM_MODELS_TO_GET_MEM_UTIL = 1
SLEEP_TIME = 3
# NOTE: Define MAX_ENGINE_POOL = 10 in [torch/csrc/autogard/python_engine.cpp]
MAX_NUM_MODELS_FOR_PYTORCH_AUTOGRAD_ENGINE = 10


class BaseMultiGPUMemoryProfiler(object):
    def __init__(self, profile_dir, user_defined_cmd,
                 timeout=TIMEOUT, mem_util_threshold=MEM_UTIL_THRESHOLD):
        self.profile_dir = profile_dir

        self.hostname = socket.gethostname()
        self.gpu_type = torch.cuda.get_device_name()
        self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory

        self.max_mem_profile_json_data = MemoryProfileJSONData(
            self.gpu_type, self.total_gpu_memory
        )

        self.timeout = timeout
        self.user_defined_cmd = user_defined_cmd
        self.mem_util_threshold = mem_util_threshold

        self.max_num_models_on_hardware = min(
            os.cpu_count() // torch.cuda.device_count(), MAX_NUM_MODELS_FOR_PYTORCH_AUTOGRAD_ENGINE)
        self.log(f'Max number of models that this GPU server can run: ' \
                 f'{self.max_num_models_on_hardware} | ' \
                 f'CPU count: {os.cpu_count()} | GPU count: {torch.cuda.device_count()}')
        self.max_num_models = 1
        self.max_num_models_by_prev_lbs = 0

        self.loop = None

    def log(self, message, status='info'):
        print_msg = f'[{status.upper()}][{self.__class__.__name__}] {message}'
        print(print_msg)

    def record_max_mem_profile_data(self, lbs):
        lbs = str(lbs)
        profile_dir = os.path.join(
                self.profile_dir, lbs, self.hostname)
        os.makedirs(profile_dir, exist_ok=True)
        json_file = os.path.join(
            profile_dir,
            MAX_MEM_PROFILE_FILE_NAME
        )
        try:
            with open(json_file, 'w') as jf:
                json_str = json.dumps(self.max_mem_profile_json_data.dict)
                jf.write(json_str)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

        # Test to confirm write json object to file
        json_data = read_json(json_file)
        self.log(f'Record data: {json_data} to {json_file}')

    @contextmanager
    def execute_handler(self):
        try:
            yield
        finally:
            pass

    def _check_terminate_condition(self, num_models):
        terminate_condition = False
        if num_models >= self.max_num_models_on_hardware:
            self.log('Terminate to reach max number of VSWs to max constaint on hardware')
            terminate_condition = True
        if self.max_num_models_by_prev_lbs > 0 and \
                num_models >= self.max_num_models_by_prev_lbs:
            self.log('Terminate to reach max number of VSWs with previous local batch size')
            terminate_condition = True
        return terminate_condition

    def _execute(self):
        lbs_str = str(self.local_batch_size)
        while True:
            try:
                accum_step = 0
                args = {
                    'local_batch_size': int(self.local_batch_size),
                    'num_models': self.max_num_models,
                    'accum_step': accum_step,
                }
                command = self.user_defined_cmd % args
                log_str = f'Start to profile number of VSWs ({lbs_str}): {self.max_num_models}'
                row_str = '*' * (len(log_str) + 1)
                self.log(row_str)
                self.log(log_str)
                self.log(row_str)
                with nvidia_smi_memory_monitoring(
                        self.max_num_models, self.mem_util_threshold,
                        min_num_models_to_run=MIN_NUM_MODELS_TO_GET_MEM_UTIL) as memory_monitor:
                    self.loop = asyncio.get_event_loop()
                    self.loop.run_until_complete(async_run_command(command, self.timeout))
                max_mem_util_by_monitor = memory_monitor.max_mem_util
                self.log('Success to execute command')
                if SLEEP_TIME > 0:
                    self.log(f'Sleep {SLEEP_TIME} sec ..')
                    time.sleep(SLEEP_TIME)
            except RuntimeError as e: # OOM happen
                self.log(e)
                kill_python_cmd = "kill -9 `ps | grep python | grep -v {0} | grep -v defunct | awk -F' ' '{{print $1}}'`".format(os.getpid())
                os.system(kill_python_cmd)
                kill_nvidiasmi_query_cmd = \
                    f"kill -9 `ps -ef | grep -v grep | grep \"nvidia-smi --query\" | awk '{{print $2}}' `"
                os.system(kill_nvidiasmi_query_cmd)
                if SLEEP_TIME > 0:
                    self.log(f'Sleep {SLEEP_TIME} sec ..')
                    time.sleep(SLEEP_TIME)
                self.max_num_models -= 1
                break

            if self._check_terminate_condition(self.max_num_models):
                break

            next_num_models_infered_by_util = 0
            if self.max_num_models > MIN_NUM_MODELS_TO_GET_MEM_UTIL:
                next_num_models_infered_by_util = math.floor(
                    self.max_num_models * (self.mem_util_threshold / max_mem_util_by_monitor))
            if next_num_models_infered_by_util > self.max_num_models + 1 and \
                    not self._check_terminate_condition(next_num_models_infered_by_util):
                self.max_num_models = next_num_models_infered_by_util
            else:
                self.max_num_models += 1

    def run(self):
        raise NotImplementedError


class StaticLocalBatchSizeMultiGPUMemoryProfiler(BaseMultiGPUMemoryProfiler):
    def __init__(self, profile_dir, user_defined_cmd, local_batch_size,
                 timeout=TIMEOUT, mem_util_threshold=MEM_UTIL_THRESHOLD):
        super().__init__(profile_dir, user_defined_cmd, timeout, mem_util_threshold)
        if local_batch_size is None:
            raise ValueError('Argument local_batch_size must be configured.')
        if not isinstance(local_batch_size, int):
            raise ValueError(
                f'Argument local_batch_size must be integer type, '
                f'but {type(local_batch_size)}')

        self.local_batch_size = str(local_batch_size)

    def run(self):
        with self.execute_handler():
            self._execute()
            # Get max num models
            log_str = f'Profiled max number of VSWs ({self.local_batch_size}): {self.max_num_models}'
            row_str = '=' * (len(log_str) + 1)
            self.log(row_str)
            self.log(log_str)
            self.log(row_str)
            if self.max_num_models >= 1:
                self.max_mem_profile_json_data.update(
                    {'lbs': self.local_batch_size, 'max_num_models': self.max_num_models}
                )
                self.record_max_mem_profile_data(self.local_batch_size)

        self.loop.close()


class DynamicLocalBatchSizeMultiGPUMemoryProfiler(BaseMultiGPUMemoryProfiler):
    def __init__(self, profile_dir, user_defined_cmd, min_lbs, search_lbs_fn, max_lbs=None,
                 timeout=TIMEOUT, mem_util_threshold=MEM_UTIL_THRESHOLD):
        super().__init__(profile_dir, user_defined_cmd, timeout, mem_util_threshold)
        if min_lbs is None:
            raise ValueError('Argument min_lbs must be configured.')
        if not isinstance(min_lbs, int):
            raise ValueError(f'Argument min_lbs must be integer type, but {type(min_lbs)}')
        if search_lbs_fn is None:
            raise ValueError(
                f'Argumnet search_lbs_fn must be configured')

        self.min_batch_size = min_lbs
        self.max_batch_size = max_lbs
        self.search_lbs_fn = search_lbs_fn

        self.local_batch_size = self.min_batch_size

    def run(self):
        with self.execute_handler():
            while True:
                if self.max_batch_size is not None and self.local_batch_size > self.max_batch_size:
                    break
                self._execute()
                # Get max num models
                log_str = f'Profiled max number of VSWs ({self.local_batch_size}): {self.max_num_models}'
                row_str = '=' * (len(log_str) + 1)
                self.log(row_str)
                self.log(log_str)
                self.log(row_str)
                if self.max_num_models >= 1:
                    self.max_mem_profile_json_data.update(
                        {'lbs': self.local_batch_size, 'max_num_models': self.max_num_models}
                    )
                    self.record_max_mem_profile_data(self.local_batch_size)
                else:
                    break

                # Traverse next local batch size by search_lbs_fn()
                self.local_batch_size = self.search_lbs_fn(self.local_batch_size)
                self.max_num_models_by_prev_lbs = self.max_num_models
                self.max_num_models = 1