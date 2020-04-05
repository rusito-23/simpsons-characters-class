import os
import sys


def create_output_folder(output_path):
    if not os.path.isdir(output_path):
        sys.exit('Output folder path non existing')
    versions = [int(x.strip('v'))
                for x in os.listdir(output_path)
                if os.path.isdir(os.path.join(output_path, x))]
    nv = (max(versions) + 1) if versions else 1
    os.mkdir(os.path.join(output_path, f'v{nv}'))
    os.mkdir(os.path.join(output_path, f'v{nv}', 'chcks'))
    return os.path.join(output_path, f'v{nv}')
