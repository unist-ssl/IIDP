from iidp.train.train_helper import calculate_bucket_size_distribution


class DDPHelper(object):
    def __init__(self):
        """
        The below member variables are required for DDPBucketProfiler
            ```model_name```
        """
        self.gpu = 0
        self.model_name = None
        self.ddp_module = None
        self.lbs = 1
        self.model = None
        self.criterion = None
        self.step = 2
        self.bucket_size_distribution = []

    def _get_ddp_bucket_indices(self):
        raise NotImplementedError

    def get_bucket_size_distribution(self):
        self._get_ddp_bucket_indices()
        if self.ddp_module is None:
            raise TypeError(
                f'[ERROR][{self.__class__.__name__}] Member variable ddp_module is None')

        self.bucket_size_distribution = calculate_bucket_size_distribution(self.ddp_module)
        print(f'[Profile info] bucket_size_distribution (backward order): {self.bucket_size_distribution}')

    def run(self):
        raise NotImplementedError