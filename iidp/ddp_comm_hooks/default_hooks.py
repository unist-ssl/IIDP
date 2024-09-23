import torch
import torch.distributed as dist


def _allreduce_sum_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    fut = dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future()

    def get_value(fut):
        return [fut.value()[0]]

    return fut.then(get_value)


def allreduce_sum_hook(
    process_group: dist.ProcessGroup, bucket: dist._GradBucket
) -> torch.futures.Future:
    return _allreduce_sum_fut(process_group, bucket.get_tensors()[0])