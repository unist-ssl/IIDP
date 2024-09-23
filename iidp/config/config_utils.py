import torch

from iidp.utils.distributed import print_one_rank
from iidp.utils.json_utils import read_json

import os
import socket
import math

from iidp.utils.global_vars import REQUIRED_CONFIG_JSON_KEYS, REQUIRED_CONFIG_FILES, \
                                    DDP_DEFAULT_BUCKET_CAPACITY
from iidp.cluster.server import GlobalServerInfo
from iidp.train.train_helper import select_weight_sync_method


# Similar logic in __init__ of IIDPConfigurator in [iidp/config/configurator.py]
def get_possible_batch_size_across_cluster(comp_profile_dir, global_server_info):
    all_server_names = []
    if type(global_server_info) == list:
        all_server_names = global_server_info
    elif type(global_server_info) == GlobalServerInfo:
        for server_info in global_server_info:
            all_server_names.append(server_info.name)
    else:
        raise TypeError(f'Not support type of arugment global_server_info: {type(global_server_info)}')

    min_possible_lbs = math.inf
    for lbs in os.listdir(comp_profile_dir):
        local_batch_size = int(lbs)
        static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, lbs)
        # Check current local batch size can be supported by current global servers
        if not set(all_server_names).issubset(set(os.listdir(static_lbs_comp_profile_dir))):
            continue
        if local_batch_size < min_possible_lbs:
            min_possible_lbs = local_batch_size
    if type(min_possible_lbs) != int:
        raise ValueError(
            f'[ERROR][get_possible_batch_size_across_cluster()] '
            f'Not exists any possible min local batch size across cluster: '
            f'{",".join(all_server_names)}.\n'
            f'It might cause since no profile data for local batch size exists on some server.\n'
            f'Please check profile data directory: ```{comp_profile_dir}```')
    return min_possible_lbs


def recommend_weight_sync_method_by_bucket_profile_data(bucket_profile_dir):
    assert len(os.listdir(bucket_profile_dir)) == 1, \
        f"[ERROR] A unique bucket size distribution json file must exist, " \
        f"but there exists {os.listdir(bucket_profile_dir)}"

    bucket_profile_file_name = sorted(os.listdir(bucket_profile_dir))[-1]
    bucket_profile_file_path = os.path.join(bucket_profile_dir, bucket_profile_file_name)
    json_data = read_json(bucket_profile_file_path)
    bucket_size_distribution = json_data['bucket_size_distribution']
    bukcet_capacity = DDP_DEFAULT_BUCKET_CAPACITY / (1024 * 1024) # MB

    weight_sync_method = select_weight_sync_method(bucket_size_distribution, bukcet_capacity)

    return weight_sync_method


def print_table(table, len):
    for i in range(0, len):
        print_one_rank(str(table[i]))


def sorted_listdir(dir_path):
    return sorted(os.listdir(dir_path), key=lambda x: int(x))


def check_server_profile_data_exists(profile_dir, curr_server=None):
    if curr_server is None:
        curr_server = socket.gethostname()
    # At least one profile data exists for current server
    is_server_profile_data_exist = False
    for lbs in os.listdir(profile_dir):
        static_lbs_profile_dir = os.path.join(profile_dir, lbs)
        for server_name in os.listdir(static_lbs_profile_dir):
            if server_name == curr_server:
                is_server_profile_data_exist = True
    return is_server_profile_data_exist


def check_user_config_is_valid(config_file):
    config_params = read_json(config_file)
    print_one_rank(f'[INFO] configuration parameters: {config_params}')

    # [CHECK 1] Required JSON keys
    is_config_json_has_required_keys = set(REQUIRED_CONFIG_JSON_KEYS).issubset(set(config_params.keys()))
    if is_config_json_has_required_keys is False:
        missing_keys = ','.join(
            list(filter(lambda elem: elem not in list(config_params.keys()), REQUIRED_CONFIG_JSON_KEYS))
        )
        raise ValueError(
            f'[FAIL] Configuration JSON \"{config_file}\" misses the required keys: ```{missing_keys}``` '
            f'among required keys: {REQUIRED_CONFIG_JSON_KEYS}')
    else:
        print_one_rank(f'[PASS] Configuration JSON \"{config_file}\" has all requied JSON keys: {REQUIRED_CONFIG_JSON_KEYS}')

    # [CHECK 2] Required profile dir exists
    for key in REQUIRED_CONFIG_FILES:
        if os.path.exists(config_params[key]) is False:
            raise ValueError(
                f'[FAIL] File \"{config_params[key]}\" must exist')
        else:
            print_one_rank(f'[PASS] \"{config_params[key]}\" exists')

    # [CHECK 3] Structure of Comp and mem profile dir
    comp_profile_dir = config_params['comp_profile_dir']
    memory_profile_dir = config_params['memory_profile_dir']
    if comp_profile_dir == memory_profile_dir:
        raise ValueError(
            f'[FAIL] Path of Computation and memory profile dir must be different, but same path: {comp_profile_dir}')
    # Check the range of local batch sizes
    if len(os.listdir(comp_profile_dir)) != len(os.listdir(memory_profile_dir)):
        raise ValueError(
            f'[FAIL] Computation and memory profile data for range of local batch size '
            f'must be equal, but comp: {sorted_listdir(comp_profile_dir)} | mem: {sorted_listdir(memory_profile_dir)}')

    # Check the same server profile data in computation and memory profile dir
    for lbs in sorted_listdir(memory_profile_dir):
        static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, lbs)
        static_lbs_mem_profile_dir = os.path.join(memory_profile_dir, lbs)
        if os.listdir(static_lbs_comp_profile_dir) != os.listdir(static_lbs_mem_profile_dir):
            raise ValueError(
                f'[FAIL] For static LBS of {lbs}, server profile data is not consistent among comp and memory profile dir!\n'
                f'comp profile dir - {static_lbs_comp_profile_dir} : {os.listdir(static_lbs_comp_profile_dir)}\n'
                f'memory profile dir - {static_lbs_mem_profile_dir} : {os.listdir(static_lbs_mem_profile_dir)}')

    # Check at least one profile data exists on current server. If not, the current server cannot be registered in available_servers
    if socket.gethostname() in config_params['available_servers']:
        if check_server_profile_data_exists(comp_profile_dir) is False:
            raise ValueError(f'[FAIL] No such computation profile data for {socket.gethostname()} '
                            f'in {comp_profile_dir}')
        if check_server_profile_data_exists(memory_profile_dir) is False:
            raise ValueError(f'[FAIL] No such memory profile data for {socket.gethostname()} '
                            f'in {memory_profile_dir}')
    for available_server in config_params['available_servers']:
        if check_server_profile_data_exists(comp_profile_dir, available_server) is False:
            raise ValueError(f'[FAIL] No such computation profile data for available server: ```{available_server}``` '
                            f'in {comp_profile_dir} => Registered available_servers: {config_params["available_servers"]}')
        if check_server_profile_data_exists(memory_profile_dir, available_server) is False:
            raise ValueError(f'[FAIL] No such memory profile data for available server: ```{available_server}``` '
                            f'in {memory_profile_dir} => Registered available_servers: {config_params["available_servers"]}')
    print_one_rank(f'[PASS] \"{comp_profile_dir}\" and \"{memory_profile_dir}\" has the right structure for configuration solver')

    # [CHECK 4] GPU cluster info JSON data
    gpu_cluster_info_file = config_params['gpu_cluster_info']
    gpu_cluster_info = read_json(gpu_cluster_info_file)
    if socket.gethostname() in config_params['available_servers']:
        if socket.gethostname() not in gpu_cluster_info.keys():
            raise ValueError(
                f'Current server: {socket.gethostname()} is not registered '
                f'in gpu cluster info json file: {gpu_cluster_info_file}'
            )
        if torch.cuda.get_device_name() != gpu_cluster_info[socket.gethostname()]['type']:
            raise ValueError(
                f'Registerd GPU type in server {socket.gethostname()} in {gpu_cluster_info_file} '
                f'```{gpu_cluster_info[socket.gethostname()]["type"]}``` is not equal to '
                f'real GPU type in server: ```{torch.cuda.get_device_name()}```'
            )
    for available_server in config_params['available_servers']:
        if available_server not in gpu_cluster_info.keys():
            raise ValueError(
                f'Current server: {available_server} is not registered '
                f'in gpu cluster info json file: {gpu_cluster_info_file}'
            )
    print_one_rank(f'[PASS] \"{gpu_cluster_info_file}\" registers the right GPU hardware for configuration solver')
