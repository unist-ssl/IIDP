import os
import subprocess
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import asyncio
from asyncio.subprocess import PIPE, STDOUT


# reference: Second solution in https://stackoverflow.com/questions/10756383/timeout-on-subprocess-readline-in-python
async def async_run_command(command, timeout=60):
    command = command.split(' ')
    command = list(filter(lambda a: a != '', command))
    program = command[0]
    args = command[1:]
    print(f'[INFO] command: {" ".join(command)}')
    print(f'[INFO] It might take time .. please wait ..')
    proc = await asyncio.create_subprocess_exec(program, *args, stdout=PIPE, stderr=STDOUT)
    while True:
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            pass
        else:
            if not line: # EOF
                break
            else:
                print(line.decode('utf-8').replace('\n',''))
                log_str = line.decode('utf-8')
                if 'out of memory' in log_str or 'RuntimeError' in log_str:
                    proc.kill()
                    raise RuntimeError('CUDA out of memory error')
                continue
        proc.kill() # Timeout or some criterion is not satisfied
        raise RuntimeError('TimeoutExpired - CUDA out of memory error')
    return await proc.wait() # Wait for the child process to exit


class nvidia_smi_memory_monitoring(object):
    def __init__(self, num_models, mem_util_threshold, min_num_models_to_run=1):
        self.num_models = num_models
        self.mem_util_threshold = mem_util_threshold
        self.min_num_models_to_run = min_num_models_to_run
        self.max_mem_util, self.max_mem_used = 0, 0

        self.proc = None
        if self.num_models > self.min_num_models_to_run:
            nvidiasmi_query_cmd = "nvidia-smi --query-gpu=memory.total,memory.used --format=csv -lms 100 &"
            # NOTE: shell=True for background command
            self.proc = subprocess.Popen(nvidiasmi_query_cmd,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         shell=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        is_timeout_error, is_mem_util_over = False, False
        if self.num_models > self.min_num_models_to_run:
            kill_nvidiasmi_query_cmd = \
                f"kill -9 `ps -ef | grep -v grep | grep \"nvidia-smi --query\" | awk '{{print $2}}' `"
            os.system(kill_nvidiasmi_query_cmd)
            # NOTE: Safe to check max mem usage by TimeoutExpired error
            if exc_type:
                if 'TimeoutExpired' in str(exc_val):
                    is_timeout_error = True
                    print(f'[INFO][{self.__class__.__name__}] '
                          f'Safe to check max mem usage by TimeoutExpired error: {exc_val}')
                else:
                    return
            memory_value_parsing_count = 0
            self.max_mem_util, self.max_mem_used = 0, 0
            for stdout in self.proc.stdout.readlines():
                log_str = stdout.decode('utf-8').replace('\n','')
                if 'memory' not in log_str and 'MiB' in log_str:
                    if len(log_str.split()) != 4: # NOTE: To avoid some stdout that has only total memory size
                        continue
                    try:
                        mem_total, mem_used = float(log_str.split()[0]), float(log_str.split()[2])
                    except:
                        print(f'[ERROR][{self.__class__.__name__}] '
                              f'log_str: {log_str}\n '
                              f'log_str.split(): {log_str.split()} | '
                              f'log_str.split(): {log_str.split()}')
                        exit(1)
                    if memory_value_parsing_count == 0 and mem_used != 0:
                        #assert mem_used == 0, f"The first parsed memory used must be zero, but {mem_used}"
                        print(f'[WARNING][{self.__class__.__name__}] '
                              f'The first parsed memory used must be zero, but {mem_used} | '
                              f'Log: {log_str}')
                    memory_value_parsing_count+=1
                    mem_util = (mem_used / mem_total) * 100
                    if mem_util >= self.mem_util_threshold:
                        is_mem_util_over = True
                        self.max_mem_util = mem_util
                        self.max_mem_used = mem_used
                        self.proc.kill()
                        self.proc = None
                        raise RuntimeError(
                            f'[{self.__class__.__name__}] CUDA out of memory error - '
                            f'Memory util: {mem_util:.2f}% > threshold: 99% | '
                            f'{mem_used}MiB / {mem_total}MiB')
                    else:
                        if mem_util > self.max_mem_util:
                            self.max_mem_util = mem_util
                            self.max_mem_used = mem_used

            if is_timeout_error is True and is_mem_util_over is False:
                print(f'[WARNING][{self.__class__.__name__}] '
                      f'TimeoutExpired error might not caused by Out of Memory error | '
                      f'Max memory usage: {self.max_mem_used}MiB / {mem_total}MiB')
                # NOTE: Timeout expired error should be handled.
                # If not, error `address already in use` will happen at the next procedure.
                return

            print(f'[INFO][{self.__class__.__name__}] '
                  f'Max memory usage util: {self.max_mem_util:.2f}% | '
                  f'{self.max_mem_used}MiB / {mem_total}MiB')
            return self
