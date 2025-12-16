import os

import yaml


def load_config(config_path):
    """Load configuration from YAML file"""
    print('The path of the configuration file is '+str(config_path))
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup paths based on GCM_STORE environment variable
    gcm_store = os.getenv('GCM_STORE', 'lustre')
    if gcm_store in config['paths']:
        paths = config['paths'][gcm_store]
        config['paths'] = paths
    else:
        raise ValueError('Unknown entry for <gcm_store> !')

    return config


def load_config_argo(config_path):
    """Load configuration from YAML file"""
    print('The path of the configuration file is '+str(config_path))
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup paths based on GCM_STORE environment variable
    gcm_store = os.getenv('GCM_STORE', 'lustre')
    if gcm_store in config['paths']:
        paths = config['paths'][gcm_store]
        # Handle special cases for argo environment
        if gcm_store == 'argo':
            data_dir = os.getenv("DATA_DIR", "")
            paths['home'] = data_dir
            paths['path_gcm_base'] = data_dir + paths['path_gcm_base']
            paths['path_gcm_base_derived'] = data_dir + paths['path_gcm_base_derived']
            paths['path_gcm_base_masked'] = data_dir + paths['path_gcm_base_masked']
            paths['dir_forecast'] = data_dir + paths['dir_forecast']
        config['paths'] = paths
    else:
        raise ValueError('Unknown entry for <gcm_store> !')

    return config
