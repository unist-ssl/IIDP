import os
import socket

from iidp.utils.json_utils import read_json
from iidp.utils.global_vars import MAX_MEM_PROFILE_FILE_NAME


def get_max_profile_json_data(profile_dir, lbs):
    memory_profile_json_data, max_memory_profile_file = None, None
    static_lbs_profile_dir = os.path.join(profile_dir, lbs)
    for server_name in os.listdir(static_lbs_profile_dir):
        if socket.gethostname() != server_name:
            continue
        max_memory_profile_file = os.path.join(
            static_lbs_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
        memory_profile_json_data = read_json(max_memory_profile_file)
    if max_memory_profile_file is None:
        current_server_max_memory_profile_file = os.path.join(
            static_lbs_profile_dir, socket.gethostname(), MAX_MEM_PROFILE_FILE_NAME)
        raise ValueError(f'No such memory profile dir: {current_server_max_memory_profile_file}')
    if memory_profile_json_data is None:
        raise ValueError('return value memory_profile_json_data is None')
    return memory_profile_json_data


def get_max_num_models_for_static_lbs(profile_dir, lbs):
    memory_profile_json_data = get_max_profile_json_data(profile_dir, lbs)
    profiled_max_num_models = memory_profile_json_data['max_num_models']
    return profiled_max_num_models
