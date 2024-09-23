from iidp.utils.json_utils import read_json
from iidp.cluster.server import GlobalServerInfo


class IIDPClusterManager(object):
    def __init__(self, gpu_cluster_info_file, gpu=None):
        self.gpu_cluster_info = read_json(gpu_cluster_info_file)
        self.gpu = gpu
        self.global_server_info = GlobalServerInfo()