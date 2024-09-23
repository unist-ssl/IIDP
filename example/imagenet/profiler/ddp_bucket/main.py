import argparse

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from ddp_bucket_profiler import ImageNetProfiler
from iidp.profiler import DDPBucketProfiler


parser = argparse.ArgumentParser(description='DDP Bucket Profiler')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--profile-dir', type=str, default=None,
                    help='Directory of profile data file.')


def main():
    args = parser.parse_args()

    profiler_instance = ImageNetProfiler(args.arch)
    ddp_bucket_profiler = DDPBucketProfiler(
            profiler_instance, args.profile_dir)
    ddp_bucket_profiler.run()


if __name__ == '__main__':
    main()