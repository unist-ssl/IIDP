from iidp.profiler import ProfileDataSchema


class MemoryProfileJSONData(ProfileDataSchema):
    def __init__(self, gpu_type, total_memory):
        super().__init__()
        self.dict = {
            'gpu_type': gpu_type,
            'total_memory': total_memory
        }

    def update(self, runtime_profile_data):
        self.dict.update(runtime_profile_data)