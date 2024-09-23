class ResourceInfo(object):
    def __init__(self, name, num_gpus_in_server, max_num_gpus_in_server, intra_network_bandwidth, inter_network_bandwidth, tfplos):
        self.device_name = name
        self.num_gpus_in_server = num_gpus_in_server
        self.max_num_gpus_in_server = max_num_gpus_in_server
        self.intra_network_bandwidth = intra_network_bandwidth # byte/sec
        self.inter_network_bandwidth = inter_network_bandwidth # byte/sec
        self.tfplos = tfplos

    def __repr__(self):
        return str(self.__dict__)


class GlobalResourceInfo(object):
    def __init__(self):
        self.total_num_servers = 0
        self.total_num_gpus = 0

    def __call__(self, total_num_servers, total_num_gpus):
        self.total_num_servers = total_num_servers
        self.total_num_gpus = total_num_gpus
        return self

    def __repr__(self):
        return str(self.__dict__)
