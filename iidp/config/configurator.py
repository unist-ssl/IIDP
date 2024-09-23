import os

import torch

from iidp.utils.distributed import print_one_rank
from iidp.utils.json_utils import read_json
from iidp.config.model.throughput.throughput_model import ThroughputModel
from iidp.cluster.resource import ResourceInfo, GlobalResourceInfo
from iidp.utils.global_vars import MAX_MEM_PROFILE_FILE_NAME
from iidp.config.config_utils import print_table, sorted_listdir


class IIDPConfig(object):
    def __init__(self, lbs, num_models, accum_step, weight_sync_method):
        self.lbs = lbs
        self.num_models = num_models
        self.accum_step = accum_step
        self.weight_sync_method = weight_sync_method


class IIDPConfigurator(object):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                 memory_profile_dir, local_config, global_server_info,
                 max_global_batch_size, is_dynamic_local_batch_size=False, gpu=None,
                 num_gpu_alloc_unit=None):
        if not isinstance(max_global_batch_size, int) or max_global_batch_size < 0:
            raise TypeError(
                f'Argument ```max_global_batch_size``` must be positive integer, '
                f'but {max_global_batch_size} and type: {type(max_global_batch_size)}')
        self.max_global_batch_size = max_global_batch_size
        self.global_server_info = global_server_info
        self.is_dynamic_local_batch_size = is_dynamic_local_batch_size
        self.gpu = gpu
        self.total_num_gpus = self.global_server_info.total_num_gpus

        self.static_lbs = local_config.lbs if self.is_dynamic_local_batch_size is False else -1
        self.num_gpu_alloc_unit = num_gpu_alloc_unit or torch.cuda.device_count()

        all_server_names = []
        for server_info in self.global_server_info:
            all_server_names.append(server_info.name)
        self.configurators = {}

        if len(sorted_listdir(comp_profile_dir)) != len(sorted_listdir(memory_profile_dir)):
            raise ValueError(
                f'[ERROR][{self.__class__.__name__}] '
                f'Computation and memory profile data for range of local batch size '
                f'must be equal, but comp: {sorted_listdir(comp_profile_dir)} | mem: {sorted_listdir(memory_profile_dir)}')
        # Create memory profile info (type: dict)
        memory_profile_info = {}
        for lbs in sorted_listdir(memory_profile_dir):
            if not lbs in memory_profile_info.keys():
                memory_profile_info[lbs] = {}
            static_lbs_mem_profile_dir = os.path.join(memory_profile_dir, lbs)
            static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, lbs)
            # Check the same server profile data in computation and memory profile dir
            if os.listdir(static_lbs_comp_profile_dir) != os.listdir(static_lbs_mem_profile_dir):
                raise ValueError(
                    f'[ERROR] For static LBS of {lbs}, server profile data is not consistent among comp and memory profile dir!\n'
                    f'comp profile dir - {static_lbs_comp_profile_dir} : {os.listdir(static_lbs_comp_profile_dir)}\n'
                    f'memory profile dir - {static_lbs_mem_profile_dir} : {os.listdir(static_lbs_mem_profile_dir)}')
            for server_name in os.listdir(static_lbs_mem_profile_dir):
                max_memory_profile_file = os.path.join(
                    static_lbs_mem_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
                memory_profile_json_data = read_json(max_memory_profile_file)
                memory_profile_info[lbs][memory_profile_json_data['gpu_type']] = memory_profile_json_data['max_num_models']

        # Instantiate configurator for static local batch size
        for lbs in sorted_listdir(comp_profile_dir):
            local_batch_size = int(lbs)
            if self.is_dynamic_local_batch_size is False and local_batch_size != local_config.lbs:
                continue
            static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, lbs)
            # Check current local batch size can be supported by current global servers
            if not set(all_server_names).issubset(set(os.listdir(static_lbs_comp_profile_dir))):
                print_one_rank(
                    f'[{self.__class__.__name__}] local_batch_size: {local_batch_size} '
                    f'is not supported by current cluster: {self.global_server_info} '
                    f'==> skip it for IIDP configuration'
                )
                continue
            if max_global_batch_size//local_batch_size < self.total_num_gpus:
                print_one_rank(
                    f'[{self.__class__.__name__}] local_batch_size: {local_batch_size} '
                    f'is not satisfied with current total number of GPUs: {self.total_num_gpus} '
                    f'==> skip it for IIDP configuration'
                )
                continue
            static_lbs_memory_profile_info = memory_profile_info[lbs]
            max_num_workers = max_global_batch_size//local_batch_size+1
            self.configurators[local_batch_size] = IIDPStaticLocalBatchSizeConfigurator(
                static_lbs_comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                static_lbs_memory_profile_info, local_batch_size,
                local_config.weight_sync_method, global_server_info, max_num_workers,
                self.num_gpu_alloc_unit
            )
        if local_config.lbs not in self.configurators.keys():
            raise ValueError(
                f'No such profile computation data for local batch size: '
                f'{local_config.lbs}, but existing data: {self.configurators.keys()}'
            )


class IIDPStaticLocalBatchSizeConfigurator(object):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                 memory_profile_info, local_batch_size, weight_sync_method,
                 global_server_info, max_num_workers, num_gpu_alloc_unit=None):
        self.comp_profile_dir = comp_profile_dir
        self.comm_profile_dir = comm_profile_dir
        self.bucket_profile_dir = bucket_profile_dir
        self.global_server_info = global_server_info
        self.total_num_gpus = self.global_server_info.total_num_gpus
        self.weight_sync_method = weight_sync_method
        self.local_batch_size = local_batch_size
        self.throughput_models = {}
        self.all_max_num_local_models_in_process_group = memory_profile_info
        self.max_num_workers = max_num_workers
        self.num_gpu_alloc_unit = num_gpu_alloc_unit or torch.cuda.device_count()

        self._build_throughput_model()

        self._init_dp_solver()

    def _build_throughput_model(self):
        """
        Assumption: Profile data of all servers must be placed on 'comp_profile_dir/{server name}'
        """
        for server_info in self.global_server_info:
            local_comp_profile_dir = os.path.join(self.comp_profile_dir, server_info.name)
            if server_info.name not in self.throughput_models.keys():
                self.throughput_models[server_info.name] = \
                    ThroughputModel(local_comp_profile_dir, self.comm_profile_dir, self.bucket_profile_dir)

    def _init_dp_solver(self):
        self.dp_solver = DynamicProgrammingSolver(
            self.local_batch_size,
            self.weight_sync_method,
            self.throughput_models,
            self.all_max_num_local_models_in_process_group,
            self.global_server_info,
            self.max_num_workers,
            self.num_gpu_alloc_unit
        )

    def estimate_time(self, server_name, num_models, accum_step,
                      resource_info: ResourceInfo, global_resource_info: GlobalResourceInfo):
        """Estimate local server iteration time"""
        iter_time, _ = self.throughput_models[server_name].evaluate(
            num_models, accum_step, self.weight_sync_method, resource_info, global_resource_info)
        return iter_time, _

    def estimate_throughput(self, total_num_decoupled_workers: int, iter_time: float):
        global_batch_size = self.local_batch_size * total_num_decoupled_workers
        thp = global_batch_size / iter_time
        return thp

    def solve_dynamic_programming(self, total_num_workers):
        throughput, iter_time, total_num_workers_by_solver,  new_config_set = -1, -1, -1, {}
        try:
            throughput, iter_time, total_num_workers_by_solver, new_config_set = self.dp_solver.solve(total_num_workers)
        finally:
            return [throughput, iter_time, total_num_workers_by_solver, new_config_set]


class DynamicProgrammingSolver(object):
    def __init__(self, local_batch_size, weight_sync_method, throughput_models,
                 all_max_num_models_info, global_server_info, max_num_workers, num_gpu_alloc_unit=None):
        self.local_batch_size = local_batch_size
        self.weight_sync_method = weight_sync_method
        self.throughput_models = throughput_models
        # NOTE: all_max_num_local_models_in_process_group = {'device name (str)': max number of VSWs (int)}
        self.all_max_num_local_models_in_process_group = all_max_num_models_info
        self.global_server_info = global_server_info
        self.total_num_gpus = self.global_server_info.global_resource_info.total_num_gpus
        self.max_num_workers = max_num_workers

        # NOTE: Important assumption: All servers contain the same number of GPUs
        self.num_gpu_alloc_unit = num_gpu_alloc_unit or torch.cuda.device_count()
        assert self.num_gpu_alloc_unit > 0, \
            f"[ERROR][{self.__class__.__name__}] Number of GPU allocation unit must be > 0, but {self.num_gpu_alloc_unit}"

        self.A = self.create_table(max_num_workers)

    def _split_rank_by_num_gpu_alloc_unit(self, arr):
        if len(arr) % self.num_gpu_alloc_unit != 0:
            raise ValueError(
                f'The number of GPUs in server: {len(arr)} is not divisible '
                f'by GPU allocation unit: {self.num_gpu_alloc_unit}')
        if len(arr) == self.num_gpu_alloc_unit:
            return [arr]
        arrays = []
        while len(arr) > self.num_gpu_alloc_unit:
            pice = arr[:self.num_gpu_alloc_unit]
            arrays.append(pice)
            arr   = arr[self.num_gpu_alloc_unit:]
        arrays.append(arr)
        return arrays

    def _generate_config_name(self, server_name, ranks, num_models, accum_step):
        """e.g, server1:0,VSW:3,GA:1 -> server1: ranks: [0, 1] (VSW, GA) = (3,1)"""
        return server_name+':'+str(ranks[0])+','+'VSW:'+str(num_models)+','+'GA:'+str(accum_step)

    def generate_config_map(self, config_set: list):
        config_map = {} # {rank: (num_models, accum_step)}
        for config_name in config_set:
            region_str, num_models_str, accum_step_str = config_name.split(',')
            head_rank = self.convert_config_str_to_int(region_str)
            num_models = self.convert_config_str_to_int(num_models_str)
            accum_step = self.convert_config_str_to_int(accum_step_str)
            for i in range(self.num_gpu_alloc_unit):
                config_map[head_rank+i] = (num_models, accum_step)
        return config_map

    def estimate_throughput(self, total_num_decoupled_workers: int, iter_time: float):
        global_batch_size = self.local_batch_size * total_num_decoupled_workers
        thp = global_batch_size / iter_time
        return thp

    def get_pruned_table_by_hash(self, A):
        """
            Table element: [iter time, number of workers(= number of VSWs * GA), config_name]
            NOTE: Important assumption - sort by iteration time in increasing order
            Prune by Hashing => The first unique config has the fastest iter time
        """

        pruned_A = []
        pruned_A_hashmap = {}
        for A_elem in A:
            _, num_worker, config_name = A_elem
            if num_worker == 1:
                pruned_A.append(A_elem)
            else:
                config_region = config_name.split(',')[0]
                hash_key = str(num_worker) + config_region
                if hash_key not in pruned_A_hashmap:
                    pruned_A_hashmap[hash_key] = A_elem
        pruned_A.extend(pruned_A_hashmap.values())

        return pruned_A

    def create_table(self, max_num_workers):
        """Table: [iter time, number of workers(= number of VSWs * GA), config_name]"""
        if max_num_workers < self.total_num_gpus:
            raise ValueError(f"Argument max_num_workers: {max_num_workers} < self.total_num_gpus: {self.total_num_gpus}")

        A = []
        MAX_GA_STEPS = 1000
        for server_info in self.global_server_info:
            server_name = server_info.name
            gpu_type = server_info.resource_info.device_name
            split_ranks = self._split_rank_by_num_gpu_alloc_unit(server_info.ranks)
            for i, ranks in enumerate(split_ranks):
                try:
                    max_num_models = self.all_max_num_local_models_in_process_group[gpu_type]
                except Exception as e:
                    print(f'self.all_max_num_local_models_in_process_group: {self.all_max_num_local_models_in_process_group}')
                    print(f'self.global_server_info: {self.global_server_info}')
                    raise e

                if max_num_workers > 0:
                    min_running_workers = self.total_num_gpus - self.num_gpu_alloc_unit
                    pruned_max_num_models = max(min((max_num_workers-min_running_workers)//self.num_gpu_alloc_unit, max_num_models), 1)
                    assert pruned_max_num_models >= 1
                    max_num_models = pruned_max_num_models
                for num_models in range(1, max_num_models+1):
                    if max_num_workers > 0:
                        pruned_max_accum_step = min(MAX_GA_STEPS, max((max_num_workers-min_running_workers)//self.num_gpu_alloc_unit//num_models, 0))
                        assert pruned_max_accum_step >= 0, f"{pruned_max_accum_step} | {num_models}"
                        max_accum_step = pruned_max_accum_step
                    else:
                        max_accum_step = MAX_GA_STEPS

                    for accum_step in range(0, max_accum_step+1):
                        iter_time, _ = self.throughput_models[server_name].evaluate(
                            num_models, accum_step, self.weight_sync_method,
                            server_info.resource_info, self.global_server_info.global_resource_info
                        )
                        config_name = self._generate_config_name(server_name, ranks, num_models, accum_step)
                        A.append([iter_time, num_models * (accum_step+1), config_name])

        # NOTE: As at least one worker must exists on every GPUs, one worker must be put ahead of table
        A.sort(key=lambda x: x[1])
        A_with_one_worker = A[:self.total_num_gpus//self.num_gpu_alloc_unit]
        A_over_one_worker = A[self.total_num_gpus//self.num_gpu_alloc_unit:]
        # NOTE: Important assumption - sort by iteration time in increasing order
        A_with_one_worker.sort(key=lambda x: x[0])
        A_over_one_worker.sort(key=lambda x: x[0])
        A = A_with_one_worker + A_over_one_worker
        return self.get_pruned_table_by_hash(A)

    def convert_config_str_to_int(self, config_str):
        return int(config_str.split(':')[-1])

    def _combine_same_region_config(self, prev_candidate_config_set: list, curr_config_name: str):
        if not isinstance(prev_candidate_config_set, list):
            raise TypeError(f"prev_candidate_config_set must be list type, but {type(prev_candidate_config_set)}")

        curr_region, curr_num_models_str, curr_accum_step_str = curr_config_name.split(',')
        curr_num_models = self.convert_config_str_to_int(curr_num_models_str)
        curr_accum_step = self.convert_config_str_to_int(curr_accum_step_str)
        new_config_set = []
        for config_name in prev_candidate_config_set:
            prev_region = config_name.split(',')[0]
            if prev_region != curr_region:
                new_config_set.append(config_name)
        new_curr_config_name = ','.join([curr_region, 'VSW:' +str(curr_num_models), 'GA:'+str(curr_accum_step)])
        new_config_set.append(new_curr_config_name)
        new_config_set.sort()
        return new_config_set

    def _get_curr_config_set(self, prev_candidate_config_set: list, curr_config_name: str) -> list:
        return self._combine_same_region_config(prev_candidate_config_set, curr_config_name)

    def _get_curr_num_workers(self, config_set: list):
        total_num_decoupled_workers = 0
        for config_name in config_set:
            _, curr_num_models_str, curr_accum_step_str = config_name.split(',')
            curr_num_models = self.convert_config_str_to_int(curr_num_models_str)
            curr_accum_step = self.convert_config_str_to_int(curr_accum_step_str)
            total_num_decoupled_workers += (curr_num_models*(curr_accum_step+1))
        return total_num_decoupled_workers

    def solve(self, total_num_workers):
        """
        NOTE: [Important]
        A[i][0] = iter time, A[i][1] = number of (virtual) workers, A[i][2] = config name on one allocation
        [Dynamic Programming] Table for DP
        each element has [iter_time, config_name]
        col: Candidate ranks to be assigned new virtual workers
        row: Candidate number of virtual workers to be assigned
        Assume self.num_gpu_alloc_unit = 2 and the number of GPUs in each server = 4
        -----------------------------------------------
                            |                real idx (number of virtual workers)                |
        ____________________|                               0                                    | 1(2) | 2(4) | 3(6) ..
        server1:0,VSW:1,GA:1| [throughput, iter time, number of workers, config set: List]   ..  |
        server1:2,VSW:1,GA:1|                                                                    |
        server2:4,VSW:1,GA:1|
        server2:6,VSW:1,GA:1|
        -----------------------------------------------
        'config_name' is a unit of 2 GPUs assignment
            e.g, config_name = server1:0,VSW:3,GA:1 ==> server1:[0,1] -> (VSW, GA) = (3,1)
        DP element: [throughput, iter time, number of workers, config set]
        DP[i][j][0] = throughput, DP[i][j][1] = iter_time, DP[i][j][2] = number of (virtual) workers, DP[i][j][3] = [config_name, ..]
        """
        if len(self.A) == 0:
            print('[INFO] No table for Dynamic Programming')
            return
        dp_row = total_num_workers//self.num_gpu_alloc_unit+1 # Purpose of +1 is that the row index of DP table indicates the number of current workers
        dp_col = len(self.A)

        # DP element: [throughput, iter time, number of workers, config set]
        dp_elem = [0, 0, 0, []]
        DP = [[dp_elem for _ in range(dp_row)] for _ in range(dp_col)]
        # Initialize table for DP
        for j in range(1, dp_row):
            if self.A[0][1] <= j:
                iter_time = self.A[0][0]
                num_workers = self.A[0][1]
                config_name = [self.A[0][2]]
                thp = self.estimate_throughput(num_workers*self.num_gpu_alloc_unit, iter_time)
                DP[0][j] = [thp, iter_time, num_workers, config_name]

        # [ Main algorithm ]
        # NOTE: Assumption: A - sorted by iteration time in increasing order (reference: create_table())
        # previous configuration with prev_max_workers has optimal sub-structure => DP[i-1][prev_max_workers]
        prev_max_workers = 1
        for i in range(1, dp_col): # i: Candidate configuration of GPU allocation
            if self.A[i][1] > dp_row: # Number of workers in a new configuration (self.A[i][1]) is over than the required total number of workers (dp_row)
                #break
                continue
            # Update current DP table to previous optimal configuration (<=prev_max_workers)
            if i == 1:
                DP[i][prev_max_workers] = DP[i-1][prev_max_workers]
                prev_max_workers+=1
            for j in range(1, prev_max_workers):
                DP[i][j] = DP[i-1][j]
            # Traverse right direction (toward increasing number of workers)
            for j in range(prev_max_workers, dp_row): # j: Candidate number of (virtual) workers
                curr_config_thp, prev_config_thp = 0, 0
                # [ Main logic for DP ] - combine previous set with a new configuration and compute objective value (throughput)
                curr_config_set = self._get_curr_config_set(DP[i-1][j][3], self.A[i][2])
                curr_max_iter_time = max(self.A[i][0], DP[i-1][j][1])
                curr_num_workers = self._get_curr_num_workers(curr_config_set)
                curr_config_thp = self.estimate_throughput(curr_num_workers*self.num_gpu_alloc_unit, curr_max_iter_time)
                prev_config_thp = DP[i-1][j][0]
                # [ Main logic for DP ] - compare objective value (throughput) in previous optimal sub-problem
                if (curr_config_thp > prev_config_thp and curr_num_workers < j) or curr_num_workers == j:
                    DP[i][j] = [curr_config_thp, curr_max_iter_time, curr_num_workers, curr_config_set]
                    if curr_num_workers == j:
                        prev_max_workers = DP[i][j][2] + 1
                        for k in range(j+1, dp_row):
                            DP[i][k] = DP[i][j]
                        break
                else:
                    DP[i][j] = DP[i-1][j]

        solution = None
        for s in range(dp_col, 0, -1):
            is_total_num_workers_required = (DP[s-1][dp_row-1][2] == dp_row-1)
            is_num_gpu_allocation_required = (len(DP[s-1][dp_row-1][-1]) == self.total_num_gpus//self.num_gpu_alloc_unit)
            if is_total_num_workers_required and is_num_gpu_allocation_required:
                solution = DP[s-1][dp_row-1]
                break
        if solution is None:
            raise AssertionError(f'[ERROR] DP Solution for total_num_workers: {total_num_workers} does not exist')
        solution[2] = solution[2] * self.num_gpu_alloc_unit
        return solution
