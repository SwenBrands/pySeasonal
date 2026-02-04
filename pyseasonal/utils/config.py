import os

import yaml


def load_config(config_path):
    """Load configuration from YAML file"""
    print('The path of the configuration file is '+str(config_path))
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup paths with HOME
    home = os.getenv('HOME_DIR', config['paths'].get('home', ''))

    for ky, vl in config['paths'].items():
        if ky == 'home':
            continue
        config['paths'][ky] = os.path.join(home, vl.lstrip('/'))

    return config
