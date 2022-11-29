import argparse
import pathlib
import subprocess

from utils_pbrl import load_environment


def main():
    parser = argparse.ArgumentParser('Run experiments for the 4-DOF ship in-distribution dataset.')
    parser.add_argument('device-idx')
    args = parser.parse_args()

    device_idx = int(args.device_idx)

    main_path = pathlib.Path(__file__).parent.parent.absolute()
    reportin_path = reportout_path = main_path.joinpath('configuration').joinpath('progress-ship.json')
    environment_path = main_path.joinpath('environment').joinpath('ship-ind.env')

    environment = load_environment(environment_path)

    if not reportin_path.exists():
        action = 'NEW'
    else:
        action = 'CONTINUE'

    subprocess.call([
        'deepsysid',
        'session',
        '--enable-cuda',
        f'--device-idx={device_idx}',
        f'--reportin={reportin_path}',
        reportout_path,
        action
    ], env=environment)

    action = 'TEST_BEST'
    subprocess.call([
        'deepsysid',
        'session',
        '--enable-cuda',
        f'--device-idx={device_idx}',
        f'--reportin={reportin_path}',
        reportout_path,
        action
    ], env=environment)


if __name__ == '__main__':
    main()
