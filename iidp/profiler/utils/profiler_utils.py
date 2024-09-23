import torch
import torch.distributed as dist

from contextlib import ContextDecorator


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class CUDAEventTimer(ContextDecorator):
    def __init__(self, msg, verbose=True):
        self.msg = msg
        self.verbose = verbose
        self.elapsed = 0

    def __enter__(self):
        self.event_start = torch.cuda.Event(enable_timing=True)
        self.event_end = torch.cuda.Event(enable_timing=True)
        self.event_start.record()
        return self

    def __exit__(self, type, value, traceback):
        self.event_end.record()
        torch.cuda.synchronize()
        self.elapsed = self.event_start.elapsed_time(self.event_end)
        if self.verbose:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    print(f'{self.msg} takes {self.elapsed:.3f} ms ({self.elapsed/1000:.3f} sec)')
            else:
                print(f'{self.msg} takes {self.elapsed:.3f} ms ({self.elapsed/1000:.3f} sec)')