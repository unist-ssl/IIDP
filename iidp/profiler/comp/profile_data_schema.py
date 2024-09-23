from iidp.profiler.profile_data_schema import ProfileDataSchema
from iidp.profiler.utils.profiler_utils import AverageMeter


class CompProfileData(ProfileDataSchema):
    def __init__(self):
        super().__init__()
        # NOTE: If new member variable is added, CompProfileJSONData().update() should be updated
        self.avg_total_time = AverageMeter('Total', ':6.3f')
        self.avg_data_time = AverageMeter('Data', ':6.3f')
        self.avg_fwd_time = AverageMeter('Fwd', ':6.3f')
        self.avg_bwd_time = AverageMeter('Bwd', ':6.3f')
        self.avg_update_time = AverageMeter('Update', ':6.3f')
        self.avg_copy_time = AverageMeter('Copy', ':6.3f')

    def __str__(self):
        return f'[Profile time (ms)] {self.avg_data_time} | {self.avg_fwd_time} | {self.avg_bwd_time} | {self.avg_update_time} | {self.avg_copy_time} | {self.avg_total_time}'

    def update(self, data_time, fwd_time, bwd_time, update_time, copy_time, total_time):
        self.avg_data_time.update(data_time)
        self.avg_fwd_time.update(fwd_time)
        self.avg_bwd_time.update(bwd_time)
        self.avg_update_time.update(update_time)
        self.avg_copy_time.update(copy_time)
        self.avg_total_time.update(total_time)


class CompProfileJSONData(ProfileDataSchema):
    def __init__(self, model_name, gpu_type, lbs, num_models):
        super().__init__()
        self.dict = {
            'model': model_name,
            'gpu_type': gpu_type,
            'lbs': lbs,
            'num_models': num_models
        }

    def update(self, runtime_profile_data, auxiliary_profile_data={}):
        self.dict.update({
            'total_time': runtime_profile_data.avg_total_time.avg,
            'data_time': runtime_profile_data.avg_data_time.avg,
            'fwd_time': runtime_profile_data.avg_fwd_time.avg,
            'bwd_time': runtime_profile_data.avg_bwd_time.avg,
            'update_time': runtime_profile_data.avg_update_time.avg,
            'copy_time': runtime_profile_data.avg_copy_time.avg,
        })
        if len(auxiliary_profile_data) > 0:
            self.dict.update(auxiliary_profile_data)