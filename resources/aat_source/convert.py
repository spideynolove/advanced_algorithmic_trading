import subprocess
import sys
from pathlib import Path
import os
import json


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def convert_files_in_directory(directory_path, p2j_path):
    python_files = Path(directory_path).glob('*.py')
    for python_file in python_files:
        conversion_command = [p2j_path, str(python_file)]
        # print(conversion_command)
        subprocess.run(conversion_command, check=True)
        print(f"Converted {python_file}")
        python_file.unlink()
        print(f"Deleted {python_file}")


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'config.json')
    config = load_config(config_path)
    p2j_path = config['p2j_path']

    folder_name = sys.argv[1]
    base_directory = os.path.dirname(os.path.realpath(__file__))
    directory_path = os.path.join(base_directory, folder_name)
    convert_files_in_directory(directory_path, p2j_path)