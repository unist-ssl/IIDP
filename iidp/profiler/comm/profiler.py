import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


parser = argparse.ArgumentParser(description='Allreduce communication Profiler')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true')
parser.add_argument('--bucket-cap-mb', default=25, type=int, help='Bucket capacity (MB)')
parser.add_argument('--network-bandwidths', type=float, nargs='+', default=[15750000000, 7000000000],
                    help="Available network bandwidth in bytes/sec")
parser.add_argument('--profile-dir', type=str, default='comm_profile_data',
                    help='Directory of profile data file.')


def convert_sec_to_ms(val):
    return val * 1000


def main():
    args = parser.parse_args()
    args.use_inter_node = args.world_size > 1
    args.intra_bandwidth = args.network_bandwidths[0]
    args.inter_bandwidth = args.network_bandwidths[1]

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        print("Use GPU: {}".format(args.gpu))

    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
        print(f'rank: {args.rank}')
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    comm_time = 0
    total_comm_time_list = []
    bucket_cap = args.bucket_cap_mb * 1024 * 1024
    allreduce_tensor = torch.ones(int(bucket_cap/4)).to(args.gpu)
    # warm-up
    send_start = time.time()
    req = dist.all_reduce(allreduce_tensor, async_op=True)
    req.wait()
    print('warmup-allreduce time (s)', time.time()-send_start)
    torch.cuda.synchronize()

    warmup_step = 1
    step = 1000
    for i in range(warmup_step+step):
        allreduce_tensor = torch.ones(int(bucket_cap/4)).to(args.gpu)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        req = dist.all_reduce(allreduce_tensor, async_op=True)
        req.wait()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup_step:
            comm_time = start.elapsed_time(end)
            total_comm_time_list.append(start.elapsed_time(end))
        if args.rank == 0 and i % 100 == 0:
            print(f'======> all-reduce time at step: {i}: {comm_time}')
    avg_comm_time = sum(total_comm_time_list) / step
    max_comm_time = max(total_comm_time_list)

    if args.rank == 0:
        param_size = allreduce_tensor.nelement()*allreduce_tensor.element_size()
        print('Param size (Byte):', param_size)
        bandwidth = args.inter_bandwidth if args.use_inter_node else args.intra_bandwidth
        print('Bandwidth (Byte/sec):', bandwidth)
        TOTAL_NUM_GPUS = args.world_size
        allreduce_step = 4 *((TOTAL_NUM_GPUS-1)/TOTAL_NUM_GPUS)
        print('All-reduce step:', allreduce_step)
        print('Real avg. communication time (ms):', avg_comm_time)
        print('Real max communication time (ms):', max_comm_time)
        peak_comm_time = convert_sec_to_ms((param_size * allreduce_step / bandwidth))
        print('Theoretical communication time (ms):', peak_comm_time)

        # Record profile data
        os.makedirs(args.profile_dir, exist_ok=True)
        file_name = 'inter_comm_profile_data.txt' if args.use_inter_node else 'intra_comm_profile_data.txt'
        profile_file_path = os.path.join(args.profile_dir, file_name)
        with open(profile_file_path, 'a') as f:
            line = ','.join([str(peak_comm_time), str(avg_comm_time)])
            f.write(line + '\n')


if __name__ == '__main__':
    main()
