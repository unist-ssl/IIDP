import os

from iidp.utils.json_utils import read_json
from iidp.utils.distributed import print_one_rank
from iidp.config.model.comm.allreduce_model import AllreduceModel
from iidp.config.model.comp.comp_model import VSWThroughputModel, VSWRealThroughputModel
from iidp.cluster.resource import ResourceInfo, GlobalResourceInfo


class ThroughputModel(object):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir, verbose=False):
        self.comp_profile_dir = comp_profile_dir
        self.comm_profile_dir = comm_profile_dir
        self.bucket_profile_dir = bucket_profile_dir
        self.verbose = verbose

        self.all_comp_data = []

        self._get_all_comp_profile_data()
        self.get_constant()
        self._build_allreduce_model()
        self._build_comp_model()

    def _get_all_comp_profile_data(self):
        for comp_profile_file in os.listdir(self.comp_profile_dir):
            comp_profile_file_path = os.path.join(self.comp_profile_dir, comp_profile_file)
            json_data = read_json(comp_profile_file_path)
            self.all_comp_data.append(json_data)
        # Sort data by number of models in increasing order
        try:
            self.all_comp_data.sort(key= lambda x:x['num_models'])
        except Exception as e:
            print(f'[ERROR] All computation profile data: {self.all_comp_data}')
            print(f'[ERROR] Check computation profile dir path: {self.comp_profile_dir}')
            raise e

    def get_constant(self):
        # comp profile data
        json_data = self.all_comp_data[-1] # Data with max number of models
        self.max_num_models = json_data['num_models']
        self.lbs = json_data['lbs']
        self.gpu_type = json_data['gpu_type']
        self.update_time = json_data['update_time'] / 1000 # ms -> s
        if self.max_num_models > 1:
            self.copy_time_per_model = (json_data['copy_time'] / (self.max_num_models-1)) / 1000 # ms -> s
        else:
            self.copy_time_per_model = 0
        self.fwd_ratio = json_data['fwd_time'] / (json_data['fwd_time'] + json_data['bwd_time'])
        self.bwd_ratio = 1 - self.fwd_ratio

        self.bn_sync_time = 0
        try:
            self.bn_sync_time = json_data['bn_sync_time'] / 1000 # ms -> s
        except:
            pass

        # bucket profile data
        bucket_profile_file_name = sorted(os.listdir(self.bucket_profile_dir))[-1]
        bucket_profile_file_path = os.path.join(self.bucket_profile_dir, bucket_profile_file_name)
        json_data = read_json(bucket_profile_file_path)
        self.bucket_size_distribution = json_data['bucket_size_distribution']

    def _build_allreduce_model(self):
        self.intra_allreduce_model, self.inter_allreduce_model = AllreduceModel(), AllreduceModel()
        for comm_profile_file in os.listdir(self.comm_profile_dir):
            comm_profile_file_path = os.path.join(self.comm_profile_dir, comm_profile_file)
            #print_one_rank(f'All-reduce profile data file path: {comm_profile_file_path}')
            x_data, y_data = [], []
            with open(comm_profile_file_path, 'r') as f:
                for line in f.readlines():
                    x_data.append(float(line.split(',')[0]))
                    y_data.append(float(line.split(',')[1]))
            if 'intra' in comm_profile_file:
                self.intra_allreduce_model.train(x_data, y_data)
            elif 'inter' in comm_profile_file:
                self.inter_allreduce_model.train(x_data, y_data)
            else:
                raise ValueError(
                    f'allreduce profile filename must inclue inter or intra term: {comm_profile_file}')

    def _build_comp_model(self):
        if self.max_num_models < 3:
            return self._build_real_comp_model()
        self.vsw_thp_model = VSWThroughputModel()
        x_data, y_data = [], []
        self.init_thp = 0
        self.data_times = {}
        for json_data in self.all_comp_data:
            num_models = json_data['num_models']
            self.data_times[num_models] = json_data['data_time'] / 1000 # ms -> s
            fwd_time = json_data['fwd_time']
            bwd_time = json_data['bwd_time']
            fwd_bwd_time = (fwd_time + bwd_time) / 1000 # convert to ms
            if num_models == 1:
                self.init_thp = round((self.lbs * num_models) / fwd_bwd_time, 2)
            else:
                assert self.init_thp != 0, f"[ERROR] self.init_thp must be > 0 if num_models > 1 - file order maybe not sorted"
            thp = (self.lbs * num_models) / fwd_bwd_time
            norm_thp = round(thp / self.init_thp, 3)
            x_data.append(num_models)
            y_data.append(norm_thp)
        self.vsw_thp_model.train(x_data, y_data)

    def _build_real_comp_model(self):
        self.vsw_thp_model = VSWRealThroughputModel()
        x_data, y_data = [], []
        self.init_thp = 0
        self.data_times = {}
        for json_data in self.all_comp_data:
            num_models = json_data['num_models']
            self.data_times[num_models] = json_data['data_time'] / 1000 # ms -> s
            fwd_time = json_data['fwd_time']
            bwd_time = json_data['bwd_time']
            fwd_bwd_time = (fwd_time + bwd_time) / 1000 # convert to ms
            if num_models == 1:
                self.init_thp = round((self.lbs * num_models) / fwd_bwd_time, 2)
            thp = (self.lbs * num_models) / fwd_bwd_time
            norm_thp = round(thp / self.init_thp, 3)
            x_data.append(num_models)
            y_data.append(norm_thp)
        self.vsw_thp_model.train(x_data, y_data)

    def evaluate_fwd_bwd_time(self, num_models):
        predicted_norm_thp = self.vsw_thp_model.evaluate(num_models)
        predicted_fwd_bwd_time = round((self.lbs * num_models) / (predicted_norm_thp * self.init_thp), 4)
        return predicted_fwd_bwd_time

    def calculate_ideal_allreduce_time(self, bucket_size_byte, total_num_gpus, bandwidth):
        allreduce_step = 4*(total_num_gpus-1)
        network_volume = bucket_size_byte/total_num_gpus * allreduce_step
        return network_volume / bandwidth * 1000 # sec to ms

    def evaluate(self, num_models, accum_step, weight_sync_method,
                 resource_info: ResourceInfo, global_resource_info: GlobalResourceInfo):
        """
        Args:
            num_models (int): Number of VSWs
            accum_step (int): GA steps
            weight_sync_method (str): 'overlap', 'sequential'
            resource_info (ResourceInfo): Resource Information of intra-server aspect

        Returns:
            iter_time (float): predicted iteration time within server level
            predicted_thp (float): predicted throughput within server level
        """
        predicted_thp = 0
        fwd_time = self.evaluate_fwd_bwd_time(num_models) * self.fwd_ratio
        bwd_time = self.evaluate_fwd_bwd_time(num_models) * self.bwd_ratio
        if self.verbose:
            print_one_rank(f'====== [{self.__class__.evaluate.__qualname__}] ======')
            print_one_rank(f'Predicted fwd time: {fwd_time:.3f} | bwd time: {bwd_time:.3f}')
        all_bucket_allreduce_time = []
        if global_resource_info.total_num_servers > 1:
            bandwidth = resource_info.inter_network_bandwidth
            allreduce_model = self.inter_allreduce_model
        else:
            bandwidth = resource_info.intra_network_bandwidth
            allreduce_model = self.intra_allreduce_model
        for bucket_size in self.bucket_size_distribution:
            bucket_size_byte = bucket_size * 1024 * 1024
            ideal_allreduce_time = self.calculate_ideal_allreduce_time(
                bucket_size_byte, global_resource_info.total_num_gpus, bandwidth)
            predicted_allreduce_time = allreduce_model.evaluate(ideal_allreduce_time) / 1000 # ms to sec
            all_bucket_allreduce_time.append(predicted_allreduce_time)
        try:
            data_time = self.data_times[num_models]
        except:
            data_time = self.data_times[1] * num_models
        if weight_sync_method == 'sequential':
            iter_time = \
                data_time + self.bn_sync_time * num_models + \
                (data_time + fwd_time + bwd_time) * accum_step + fwd_time + \
                max(bwd_time, sum(all_bucket_allreduce_time[:-1])) + \
                all_bucket_allreduce_time[-1] + \
                self.update_time + (self.copy_time_per_model * (num_models-1))

            gbs = self.lbs * num_models * (accum_step+1) * resource_info.num_gpus_in_server
            predicted_thp = round(gbs / iter_time, 2)
            if self.verbose:
                print_one_rank(f'iter time = {iter_time:.4f}')
                print_one_rank(f'GBS = {gbs}')
                print_one_rank(f'Predicted throughput of GBS: {gbs} = {predicted_thp:.4f}')

                # In-depth analysis
                print_one_rank(f'========================== Breakdown ========================== \n' \
                    f'1) Data: {data_time + data_time*accum_step} \n' \
                    f'2) BN sync: {self.bn_sync_time} \n' \
                    f'3) GA + fwd: {(fwd_time + bwd_time) * accum_step + fwd_time:.4f} \n' \
                    f'4) overlap bwd + allreduce: {max(bwd_time, sum(all_bucket_allreduce_time[:-1])):.4f} \n' \
                    f'\t4-1) bwd: {bwd_time} 4-2) allreduce: {sum(all_bucket_allreduce_time[:-1]):.4f} \n' \
                    f'5) allreduce of last bucket: {all_bucket_allreduce_time[-1]:.4f} \n' \
                    f'6) update: {self.update_time:.4f} \n'
                    f'7) copy: {self.copy_time_per_model * (num_models-1):.4f} \n' \
                    f'===============================================================')

        elif weight_sync_method == 'overlap':
            last_bucket_size_ratio = self.bucket_size_distribution[-1] / sum(self.bucket_size_distribution)
            if self.verbose:
                print_one_rank(f'[INFO] last_bucket_size_ratio: {last_bucket_size_ratio}')
            iter_time = \
                data_time + self.bn_sync_time * num_models + \
                (data_time + fwd_time + bwd_time) * accum_step + fwd_time + \
                max(bwd_time, sum(all_bucket_allreduce_time[:-1])) + \
                all_bucket_allreduce_time[-1] + \
                (self.update_time + (self.copy_time_per_model * (num_models-1))) * last_bucket_size_ratio

            gbs = self.lbs * num_models * (accum_step+1) * resource_info.num_gpus_in_server
            predicted_thp = round(gbs / iter_time, 2)
            if self.verbose:
                print_one_rank(f'iter time = {iter_time:.4f}')
                print_one_rank(f'GBS = {gbs}')
                print_one_rank(f'Predicted throughput of GBS: {gbs} = {predicted_thp:.4f}')

                # In-depth analysis
                print_one_rank(f'========================== Breakdown ========================== \n' \
                    f'1) Data: {data_time + data_time*accum_step} \n' \
                    f'2) BN sync: {self.bn_sync_time} \n' \
                    f'3) GA + fwd: {(fwd_time + bwd_time) * accum_step + fwd_time:.4f} \n' \
                    f'4) overlap bwd + allreduce: {max(bwd_time, sum(all_bucket_allreduce_time[:-1])):.4f} \n' \
                    f'\t4-1) bwd: {bwd_time} 4-2) allreduce: {sum(all_bucket_allreduce_time[:-1]):.4f} \n' \
                    f'5) allreduce of last bucket: {all_bucket_allreduce_time[-1]:.4f} \n' \
                    f'6) update: {self.update_time*last_bucket_size_ratio:.4f} \n'
                    f'7) copy: {(self.copy_time_per_model * (num_models-1))*last_bucket_size_ratio:.4f} \n' \
                    f'===============================================================')
        else:
            raise ValueError(f'Not support such weight sync method: {weight_sync_method}')

        return iter_time, predicted_thp
