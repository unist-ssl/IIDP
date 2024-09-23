MAX_MEM_PROFILE_FILE_NAME = 'max_memory_profile_info.json'

REGISTERED_WEIGHT_SYNC_METHODS = [
    'recommend',
    'overlap',
    'sequential',
]

DDP_DEFAULT_BUCKET_CAPACITY = 25 * 1024 * 1024

WEIGHT_SYNC_METHOD_SELECTION_THRESHOLD = 1.5

CHECKPOINT_FILE_NAME = 'trainer_state_checkpoint.pth'

REQUIRED_CONFIG_JSON_KEYS = [
    'memory_profile_dir',
    'comp_profile_dir',
    'comm_profile_dir',
    'bucket_profile_dir',
    'gpu_cluster_info',
    'available_servers',
]

REQUIRED_CONFIG_FILES = [
    'memory_profile_dir',
    'comp_profile_dir',
    'comm_profile_dir',
    'bucket_profile_dir',
    'gpu_cluster_info'
]