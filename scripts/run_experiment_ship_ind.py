import argparse
import os
import pathlib
import subprocess
from typing import Dict


def load_environment(environment_path: pathlib.Path) -> Dict[str, str]:
    env = os.environ.copy()
    with environment_path.open(mode='r') as f:
        for line in f:
            var_name, var_value = line.strip().split('=')
            env[var_name] = var_value
    return env


def main():
    parser = argparse.ArgumentParser('Run experiments for the 4-DOF ship in-distribution dataset.')
    parser.add_argument('device-idx')
    args = parser.parse_args()

    device_idx = int(args.device_idx)

    main_path = pathlib.Path(__file__).parent.parent.absolute()
    reportin = reportout = main_path.joinpath('configuration').joinpath('progress-ship.json')
    environment_path = main_path.joinpath('environment').joinpath('ship-ind.env')

    environment = load_environment(environment_path)

    if not reportin.exists():
        action = 'NEW'
    else:
        action = 'CONTINUE'

    subprocess.call([
        'deepsysid',
        'session',
        '--enable-cuda',
        f'--device-idx={device_idx}',
        f'--reportin={reportin}',
        reportout,
        action
    ], env=environment)

    action = 'TEST_BEST'
    subprocess.call([
        'deepsysid',
        'session',
        '--enable-cuda',
        f'--device-idx={device_idx}',
        f'--reportin={reportin}',
        reportout,
        action
    ], env=environment)


if __name__ == '__main__':
    main()
