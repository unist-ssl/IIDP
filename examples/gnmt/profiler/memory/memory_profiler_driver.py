import argparse
import os

from iidp.profiler import StaticLocalBatchSizeMultiGPUMemoryProfiler, \
                            DynamicLocalBatchSizeMultiGPUMemoryProfiler


parser = argparse.ArgumentParser()
parser.add_argument('--profile-dir', type=str, default=None, required=True,
                    help='Directory of profile data file.')
parser.add_argument('--local-batch-size', '-lbs', default=None, type=int,
                    help='Local batch size for profiling')
parser.add_argument('--min-batch-size', default=None, type=int,
                    help='Min local batch size for profiling')
parser.add_argument('--max-batch-size', default=None, type=int,
                    help='Max local batch size for profiling')


def main():
    args = parser.parse_args()

    if args.local_batch_size is None and args.min_batch_size is None:
        raise ValueError(f'One of argument --local-batch-size or --min-batch-size must be configured')
    if args.local_batch_size is not None and args.min_batch_size is not None:
        raise ValueError(f'Not support both arguments --local-batch-size and --min-batch-size')

    is_static_lbs = (args.local_batch_size is not None)
    is_dynamic_lbs = (args.min_batch_size is not None)

    if is_dynamic_lbs:
        if os.path.exists(args.profile_dir):
            raise ValueError(f'--profile-dir {args.profile_dir} must not exist')

        def search_lbs_fn(lbs):
            return lbs + args.min_batch_size

    memory_util_threshold = 99
    MAX_PROFILE_ITER = 20
    data_dir = os.path.join(os.getenv('IIDP_DATA_STORAGE'), 'pytorch_wmt16_en_de')
    CMD_TEMPLATE = f"""python train.py \
        --dist-url tcp://localhost:24543 \
        --dist-backend nccl \
        --num-minibatches {MAX_PROFILE_ITER} \
        --rank 0 \
        --world-size 1 \
        --weight-sync-method recommend \
        -lbs %(local_batch_size)d \
        --num-models %(num_models)d \
        --accum-step %(accum_step)d \
        --dataset-dir {data_dir}"""

    if is_static_lbs:
        mem_profiler = StaticLocalBatchSizeMultiGPUMemoryProfiler(
            args.profile_dir, CMD_TEMPLATE,
            args.local_batch_size, mem_util_threshold=memory_util_threshold)
    elif is_dynamic_lbs:
        mem_profiler = DynamicLocalBatchSizeMultiGPUMemoryProfiler(
            args.profile_dir, CMD_TEMPLATE, args.min_batch_size,
            search_lbs_fn, args.max_batch_size, mem_util_threshold=memory_util_threshold)
    mem_profiler.run()


if __name__ == '__main__':
    main()