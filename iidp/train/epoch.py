class EpochIterator(object):
    def __init__(self):
        self.epoch_idx = -1
        self.final_epochs = -1

    @property
    def epoch(self):
        if self.final_epochs == -1:
            return 0
        else:
            return self.epoch_idx

    @epoch.setter
    def epoch(self, epoch):
        self.epoch_idx = epoch

    def __iter__(self):
        return self

    def __next__(self):
        if self.final_epochs == -1:
            raise ValueError(
                f'[ERROR][iidp.train.epoch.EpochIterator] final_epochs must be > 0 in EpochIterator().__next__()')
        if self.epoch_idx < self.final_epochs-1:
            self.epoch_idx += 1
            return self.epoch_idx
        else:
            self.epoch_idx = -1
            raise StopIteration

    def __len__(self):
        return self.final_epochs