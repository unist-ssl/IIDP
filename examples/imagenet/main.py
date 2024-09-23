import argparse
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, MultiStepLR, _LRScheduler

import iidp
from iidp.optim import ShardSGD


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='data', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--num-minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--no-validate', dest='no_validate', action='store_true',
                    help="No validation")
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--checkpoint-path', type=str,
                    default=None,
                    help='checkpoint path')
parser.add_argument('--lr-scaling', action='store_true',
                    help="LR linear scaling rule")

# IIDP
parser.add_argument('--local-batch-size', '-lbs', default=32, type=int,
                    help='Local batch size to be preserved')
parser.add_argument('--accum-step', type=int, default=0, help='Gradient accumulation step')
parser.add_argument('--num-models', type=int, default=1, help='Number of VSWs')
parser.add_argument('--weight-sync-method', type=str, default='recommend',
                    choices=['recommend', 'overlap', 'sequential'],
                    help='Weight synchronization method in IIDP')

parser.add_argument('--synthetic-dataset', action='store_true',
                    help="Use synthetic dataset")


class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = torch.randn(*input_size)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length


best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.num_minibatches is not None:
        torch.cuda.empty_cache()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.dist_url = 'tcp://127.0.0.1:22222'
        print(f'[INFO] single-GPU | args.rank: {args.rank}')
        print(f'[INFO] single-GPU | args.world_size: {args.world_size}')
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print('Local Batch Size:', args.local_batch_size)

    trainer = iidp.IIDPTrainer(
        args.gpu, args.local_batch_size, args.num_models, args.accum_step, args.weight_sync_method)

    args.batch_size = trainer.batch_size_per_gpu
    print('Local Batch Size per GPU:', args.batch_size)
    args.global_batch_size = trainer.global_batch_size
    print('Global Batch Size:', args.global_batch_size)

    # Create model
    if args.arch.startswith('inception'):
        model = models.inception_v3(transform_input=True, aux_logits=False)
    else:
        model = models.__dict__[args.arch]()
    model = model.to(args.gpu)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Prepare stream parallelism
    trainer.prepare_stream_parallel(model, criterion)

    cudnn.benchmark = False

    # Data loading code
    if not args.synthetic_dataset:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch.startswith('inception'):
        if args.synthetic_dataset:
            train_dataset = SyntheticDataset((3, 299, 299), 1281167)
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    normalize,
                ]))
    else:
        if args.synthetic_dataset:
            train_dataset = SyntheticDataset((3, 224, 224), 1281167)
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    def chunk_func(batch, num_chunks, loading_once):
        if loading_once is True:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda(args.gpu)
            targets = targets.cuda(args.gpu)
            # NOTE: torch.chunk() may return smaller number of chunks
            chunked_inputs = torch.tensor_split(inputs, num_chunks)
            chunked_targets = torch.tensor_split(targets, num_chunks)
            parallel_local_data = []
            for chunked_input, chunked_target in zip(chunked_inputs, chunked_targets):
                if chunked_input.numel() == 0 or chunked_target.numel() == 0:
                    print(f'[WARNING] empty input or target: {chunked_input} | {chunked_target}')
                    print(f'[WARNING] inputs: {inputs.size()} | num_chunks: {num_chunks}')
                    return []
                parallel_local_data.append([chunked_input, chunked_target])
            return parallel_local_data
        else:
            parallel_local_data = []
            for (images, target) in batch:
                assert images.size()[0] == trainer.local_batch_size, \
                    f"Input size must be equal to local batch size, but {images.size()[0]} != {trainer.local_batch_size}"
                images = images.cuda(args.gpu)
                target = target.cuda(args.gpu)
                parallel_local_data.append([images, target])
        return parallel_local_data

    train_loader = iidp.data.DataLoader(
        train_dataset, batch_size=args.batch_size, batch_fn=chunk_func, loading_once=True,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    if not args.no_validate and not args.synthetic_dataset:
        validate_batch_size = 32
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=validate_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # Create optimizer
    if trainer.weight_sync_method == 'overlap':
        optimizer = ShardSGD(trainer.main_model.parameters(), args.lr,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(trainer.main_model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    gradual_warmup_scheduler = None
    if (args.arch == 'resnet50' and args.global_batch_size > 256 and
        args.lr_scaling and not args.num_minibatches):
        print('LR linear scaling rule is applied '
            'when training ResNet-50 with the global batch size > 256')

        gradual_warmup_scheduler = GradualWarmupStepLR(
                optimizer, warmup_epochs=5, global_batch_size=args.global_batch_size,
                iterations_per_epoch=len(train_loader))

    trainer.prepare_weight_sync_method(optimizer, gradual_warmup_scheduler)

    model = trainer.eval_model
    optimizer = trainer.main_optimizer
    scheduler = get_lr_scheduler(optimizer, args.arch)

    # optionally resume from a checkpoint
    if args.resume:
        trainer.load(args.resume)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.num_minibatches is not None:
        print('Number of mini-batches:', args.num_minibatches)
        args.epochs = 1
        print('Start epoch, epochs:', trainer.epoch, args.epochs)

    for epoch in trainer.remaining_epochs(args.epochs):
        if args.distributed and train_sampler:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        with trainer.measure_epoch_time(), trainer.record_epoch_data():
            train(train_loader, trainer, epoch, args)
            scheduler.step()

        if not args.no_validate and not args.synthetic_dataset:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

        if args.checkpoint_path and args.epochs > 1:
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and \
                     args.rank % ngpus_per_node == 0):
                trainer.save(args.checkpoint_path)


def train(train_loader, trainer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.5f')
    comp_time = AverageMeter('Comp', ':6.5f')
    update_time = AverageMeter('Update', ':6.5f')
    losses = AverageMeter('Loss', ':.4e')
    if args.num_minibatches is not None:
        num_batches = args.num_minibatches
    else:
        num_batches = len(train_loader)
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, comp_time, update_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    trainer.set_model_train()

    end = time.time()
    is_sync_step = False
    warmup_step = 10 * (trainer.accum_step+1)
    for i, data in enumerate(train_loader):
        if args.num_minibatches is not None and trainer.sync_step > args.num_minibatches:
            break
        is_record_step = ((i+1) % (trainer.accum_step+1) == 0)
        # measure data loading time
        if i >= warmup_step and is_record_step:
            data_time.update(time.time() - end)
            start = time.time()

        trainer.compute(data)

        # Record loss
        losses.update(trainer.losses[0].detach(), trainer.local_batch_size)

        if i >= warmup_step and is_record_step:
            comp_time.update(time.time() - start)
            start = time.time()

        # Update parameters
        is_sync_step = trainer.step()

        if is_sync_step and \
                trainer.local_schedulers and epoch < trainer.local_schedulers[0].warmup_epochs:
            trainer.scheduler_step()

        if i >= warmup_step and is_record_step:
            update_time.update(time.time() - start)

        # measure elapsed time
        if i >= warmup_step and is_record_step:
            batch_time.update(time.time() - end)
        if is_record_step:
            end = time.time()

        if is_record_step and ((train_loader.step_index+1) % args.print_freq == 0):
            # As step starts from 0, printing step+1 is right
            progress.display(train_loader.step_index+1)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.detach(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class GradualWarmupStepLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, global_batch_size, iterations_per_epoch,
                 last_epoch=-1, verbose=False):
        self.base_lr = 0.1
        self.warmup_epochs = warmup_epochs
        self.global_batch_size = global_batch_size
        self.iterations_per_epoch = iterations_per_epoch

        self.scaled_lr = self.base_lr * (global_batch_size/256)
        self.warmup_steps = self.warmup_epochs * self.iterations_per_epoch
        self.increment_per_step = float((self.scaled_lr - self.base_lr) / self.warmup_steps)
        super(GradualWarmupStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            lr = [base_lr + (self.increment_per_step*self.last_epoch) for base_lr in self.base_lrs]
        else:
            raise IndexError(f'self.last_epoch: {self.last_epoch} > self.warmup_steps: {self.warmup_steps}')
        return lr


def get_lr_scheduler(optimizer, arch):
    if 'resnet' in arch:
        scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    return scheduler


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
