import argparse

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from iidp.profiler import ComputationProfiler
from comp_profiler import ImageNetProfiler


parser = argparse.ArgumentParser(description='Computation Profiler')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--local-batch-size', '-lbs', default=None, type=int, required=True,
                    help='Local batch size to be preserved')
parser.add_argument('--num-models', type=int, default=1, help='Number of VSWs')
parser.add_argument('--profile-dir', type=str, default=None,
                    help='Directory of profile data file.')


def main():
    args = parser.parse_args()

    profiler_instance = ImageNetProfiler(args.local_batch_size, args.num_models, args.arch)
    comp_profiler = ComputationProfiler(profiler_instance, args.profile_dir)
    comp_profiler.run()


if __name__ == '__main__':
    main()