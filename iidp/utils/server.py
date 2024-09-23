from iidp.cluster.server import GlobalServerInfo, ServerInfo


def build_global_cluster_by_config_file(available_servers: list, gpu_cluster_info: dict, verbose=False):
    cluster_str_list = []
    for available_server in available_servers:
        num_gpus_in_server = gpu_cluster_info[available_server]['number']
        cluster_str_list.append(f'{available_server}:{num_gpus_in_server}')
    cluster = ','.join(cluster_str_list)
    if verbose:
        log_str = f'[INFO][iidp/utils][build_global_cluster_by_config_file] Global cluster: {cluster}'
        row_str = '=' * (len(log_str) + 1)
        print(row_str)
        print(log_str)
        print(row_str)
    return cluster


def build_mock_server_info(cluster: str, gpu_cluster_info: dict, verbose=False):
    if type(cluster) != str:
        raise TypeError(f'[ERROR] Argument cluster must be string type, '
                        f'but type: {type(cluster)} | cluster: {cluster}')
    if type(gpu_cluster_info) != dict:
        raise TypeError(f'[ERROR] Argument gpu_cluster_info must be dictionary type, '
                        f'but type: {type(gpu_cluster_info)} | gpu_cluster_info: {gpu_cluster_info}')
    mock_global_server_group = {}
    server_groups = cluster.split(',')
    last_rank = 0
    total_num_gpus = 0
    for server_group in server_groups:
        hostname, num_gpus_in_server = server_group.split(':')
        ranks = [last_rank + rank for rank in range(int(num_gpus_in_server))]
        last_rank = ranks[-1] + 1
        mock_global_server_group[hostname] = ranks
        total_num_gpus+=int(num_gpus_in_server)
    if verbose:
        print(f'[INFO] Server group: {mock_global_server_group}')
    mock_global_server_info = GlobalServerInfo()
    for name, ranks in mock_global_server_group.items():
        mock_global_server_info.add(ServerInfo(name, ranks, gpu_cluster_info[name]))
    if verbose:
        print(f'[INFO] Global Server Info: {mock_global_server_info}')
    return mock_global_server_info, mock_global_server_group