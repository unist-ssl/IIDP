import torch
import torch.distributed as dist


def print_rank(message, status='info'):
    if not isinstance(message, str):
        message = str(message)
    print(f'[{status.upper()}] rank: {dist.get_rank()} | ' + message)


def print_one_rank(message, status='info', rank=0):
    if not isinstance(message, str):
        message = str(message)
    if dist.is_initialized():
        if dist.get_rank() == rank:
            print(f'[{status.upper()}] rank: {dist.get_rank()} | ' + message)
    else:
        print(f'[{status.upper()}] | ' + message)


def get_allgather_value(value, gpu=None):
    if dist.is_initialized():
        total_num_gpus = dist.get_world_size()
        device = gpu if gpu is not None else dist.get_rank()
        tensor_list = [
            torch.tensor([0], dtype=torch.float32).to(device) for _ in range(total_num_gpus)
        ]
        tensor = torch.tensor([value], dtype=torch.float32).to(device)
        dist.all_gather(tensor_list, tensor)
        if isinstance(value, int):
            ret_val = [int(tensor.item()) for tensor in tensor_list]
        else:
            ret_val = [tensor.item() for tensor in tensor_list]
        # Save GPU memory
        num_tensors = len(tensor_list)
        for _ in range(num_tensors):
            tensor_list[-1].cpu()
            del tensor_list[-1]
        return ret_val
    else:
        return [value]