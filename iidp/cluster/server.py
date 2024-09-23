from iidp.cluster.resource import ResourceInfo, GlobalResourceInfo


class ServerInfo(object):
    def __init__(self, name, ranks, resource_info):
        self.name = name
        self.ranks = ranks # List of global ranks
        self.resource_info = ResourceInfo(
            name=resource_info['type'],
            num_gpus_in_server=len(ranks),
            max_num_gpus_in_server=resource_info['number'],
            intra_network_bandwidth=resource_info['intra_network_bandwidth'],
            inter_network_bandwidth=resource_info['inter_network_bandwidth'],
            tfplos=resource_info['tfplos'])

    def __repr__(self):
        return f'[ServerInfo] name: {self.name} | ranks: {self.ranks} | ' \
               f'self.resource_info: {self.resource_info.__repr__()}'


class GlobalServerInfo(object):
    def __init__(self):
        self.local_servers = []
        self.local_server_idx = 0
        self.resource_info = GlobalResourceInfo()
        self.rank_to_server_map = {}
        self.name_to_server_map = {}
        self.registered_server_name = []

    @property
    def global_resource_info(self):
        return self.resource_info(self.total_num_servers, self.total_num_gpus)

    @property
    def total_num_servers(self):
        return len(self.local_servers)

    @property
    def total_num_gpus(self):
        total_num_gpus_in_current_local_servers = 0
        for local_server in self.local_servers:
            total_num_gpus_in_current_local_servers += local_server.resource_info.num_gpus_in_server
        return total_num_gpus_in_current_local_servers

    def add(self, local_server: ServerInfo):
        # Handle if name of server is already registered
        if local_server.name in self.registered_server_name:
            if local_server.ranks != self.name_to_server_map[local_server.name].ranks:
                self._merge_local_servers(local_server.name, local_server.ranks)
                return
            else: # Skip add() as it is the same local server info
                return
        self.local_servers.append(local_server)
        self.registered_server_name.append(local_server.name)

        self.name_to_server_map[local_server.name] = local_server
        for rank in local_server.ranks:
            self.rank_to_server_map[rank] = local_server

    def _merge_local_servers(self, name, ranks):
        self.name_to_server_map[name].ranks.extend(
            rank for rank in ranks \
                if rank not in self.name_to_server_map[name].ranks
        )
        self.name_to_server_map[name].resource_info.num_gpus_in_server = len(self.name_to_server_map[name].ranks)
        for rank in ranks:
            self.rank_to_server_map[rank] = self.name_to_server_map[name]

    def __repr__(self):
        repr_str = f'[GlobalServerInfo] total_num_gpus: {self.total_num_gpus} | ' \
                   f'total_num_servers: {self.total_num_servers} | ' \
                   f'registered_server_name: {self.registered_server_name} | ' \
                   f'[LocalServerInfos]: '
        for local_server in self.local_servers:
            repr_str += (local_server.__repr__() + ' ')
        return repr_str

    def __iter__(self):
        return self

    def __next__(self):
        if self.local_server_idx < self.total_num_servers:
            local_server = self.local_servers[self.local_server_idx]
            self.local_server_idx += 1
            return local_server
        else:
            self.local_server_idx = 0
            raise StopIteration

    def __eq__(self, other):
        if isinstance(other, GlobalServerInfo):
            result = self.total_num_servers == other.total_num_servers and \
                    self.total_num_gpus == other.total_num_gpus and \
                    self.registered_server_name.sort() == other.registered_server_name.sort()
            if result == True:
                # sort by server name
                sorted_self_local_servers = sorted(self.local_servers, key=lambda x: x.name.lower())
                sorted_other_local_servers = sorted(other.local_servers, key=lambda x: x.name.lower())
                for self_server_info, other_server_info in zip(sorted_self_local_servers, sorted_other_local_servers):
                    self_num_gpus_in_server = self_server_info.resource_info.num_gpus_in_server
                    other_num_gpus_in_server = other_server_info.resource_info.num_gpus_in_server
                    if self_server_info.name != other_server_info.name or self_num_gpus_in_server != other_num_gpus_in_server:
                        result = False
                        break
            return result
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
