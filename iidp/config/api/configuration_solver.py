import argparse

from iidp.config.configurator import IIDPConfig, IIDPConfigurator
from iidp.config.config_utils import get_possible_batch_size_across_cluster, \
                                     recommend_weight_sync_method_by_bucket_profile_data, \
                                     check_user_config_is_valid
from iidp.cluster.cluster_manager import IIDPClusterManager
from iidp.utils.json_utils import read_json
from iidp.utils.server import build_mock_server_info, build_global_cluster_by_config_file
from iidp.utils.global_vars import REGISTERED_WEIGHT_SYNC_METHODS


parser = argparse.ArgumentParser(description='Configuration Solver API')
parser.add_argument('--config-file', '-c', type=str, required=True,
                    help='Configuration file path (json)')
parser.add_argument('--global-batch-size', '-gbs', default=None, type=int, required=True,
                    help='Global batch size')

# Optional
parser.add_argument('--local-batch-size', '-lbs', default=None, type=int,
                    help='Local batch size to be preserved')
parser.add_argument('--weight-sync-method', '-wsm', type=str, default='recommend',
                    choices=REGISTERED_WEIGHT_SYNC_METHODS,
                    help='Weight synchronization method in IIDP')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Verbose to dynamic programming procedure')


def convert_rank_in_config_set_to_allocated_num_gpus(config_set: list, num_gpu_alloc_unit: int) -> list:
    """Current: ['server1:0,VSW:3,GA:1', ..] -> Return: ['server1:2GPU,VSW:3,GA:1', ..]"""
    new_config_set = []
    if config_set == []:
        return config_set
    for config in config_set:
        server_info_str, vsw_info_str, ga_info_str = config.split(',')
        hostname, rank = server_info_str.split(':')
        server_placement_str = hostname + ':' + str(num_gpu_alloc_unit) + 'GPU'
        new_config_set.append(','.join([server_placement_str, vsw_info_str, ga_info_str]))
    return new_config_set


def main():
    args = parser.parse_args()

    check_user_config_is_valid(args.config_file)

    config_params = read_json(args.config_file)

    enable_adjust_lbs = (args.local_batch_size is None)
    if args.verbose:
        if enable_adjust_lbs is True:
            print(f'=====================================================')
            print(f'[INFO] Dynamic local batch size')
            print(f'=====================================================')
        else:
            print(f'=====================================================')
            print(f'[INFO] Static local batch size')
            print(f'=====================================================')

    gpu_cluster_info = read_json(config_params["gpu_cluster_info"])

    # Build Mock global and available server info
    args.cluster = build_global_cluster_by_config_file(
            config_params["available_servers"], gpu_cluster_info)
    mock_available_server_info, mock_global_server_group = build_mock_server_info(
            args.cluster, gpu_cluster_info, args.verbose)

    mock_available_server_name_list = mock_global_server_group.keys()
    mock_cluster_manager = IIDPClusterManager(config_params["gpu_cluster_info"])
    mock_cluster_manager.global_server_info = mock_available_server_info

    # Get local batch size
    if args.local_batch_size is None:
        try:
            args.local_batch_size = get_possible_batch_size_across_cluster(
                    config_params["comp_profile_dir"], list(mock_available_server_name_list))
        except Exception as e:
            print(e)
            print(f'[ERROR] Not support such local batch size: {args.local_batch_size}')
            exit(1)

    # Build IIDPConfigurator
    if args.weight_sync_method == 'recommend':
        args.weight_sync_method = recommend_weight_sync_method_by_bucket_profile_data(config_params["bucket_profile_dir"])
    if args.verbose:
        print(f'=====================================================')
        print(f'[INFO] Weight synchronization method: {args.weight_sync_method}')
        print(f'=====================================================')
    local_config = IIDPConfig(args.local_batch_size, 1, 1, args.weight_sync_method)
    iidp_configurator = IIDPConfigurator(
        config_params["comp_profile_dir"],
        config_params["comm_profile_dir"],
        config_params["bucket_profile_dir"],
        config_params["memory_profile_dir"],
        local_config,
        mock_cluster_manager.global_server_info,
        args.global_batch_size,
        enable_adjust_lbs
    )
    # Search space of GPU configuration
    best_throughput = 0
    best_result = []
    best_lbs = 0
    if args.verbose:
        print('[INFO] ********************** Search start! **********************\n')
    for local_batch_size, configurator in iidp_configurator.configurators.items():
        result = []
        total_num_workers = int(args.global_batch_size / local_batch_size)
        if args.verbose:
            print(f'[INFO] GBS: {args.global_batch_size} | LBS: {local_batch_size} | M: {total_num_workers}')
        if args.global_batch_size != (local_batch_size * total_num_workers):
            if args.verbose:
                print(f'[WARNING] GBS: {args.global_batch_size} != LBS: {local_batch_size} * M: {total_num_workers} ==> skip!')
            continue
        if total_num_workers >= configurator.total_num_gpus:
            result = configurator.solve_dynamic_programming(total_num_workers)
            result[-1] = convert_rank_in_config_set_to_allocated_num_gpus(result[-1], configurator.dp_solver.num_gpu_alloc_unit)
            if result[2] != total_num_workers:
                if args.verbose:
                    print(f"[WARNING] resulting number of total workers: {result[2]}, but required one: {total_num_workers} ==> skip!")
                continue
        else:
            print(f'[INFO] GBS: {args.global_batch_size} | LBS: {local_batch_size} | '
                    f'total_num_workers: {total_num_workers} < configurator.total_num_gpus: {configurator.total_num_gpus}')
        if len(result)> 0 and result[0] > best_throughput:
            best_throughput = result[0]
            best_result = result
            best_lbs = local_batch_size

    if len(best_result) == 0:
        print(f'\n=====================================================')
        print(f'[INFO] No solution - GBS: {args.global_batch_size}')
        print(f'=====================================================\n')
    else:
        log_str = f'[INFO] Solution - GBS: {args.global_batch_size} | LBS: {best_lbs} | ' \
                  f'weight sync method: {args.weight_sync_method} | config: {best_result[-1]}'
        row_str = '=' * (len(log_str) + 1)
        print('\n' + row_str)
        print(log_str)
        print(row_str)


if __name__ == '__main__':
    main()
