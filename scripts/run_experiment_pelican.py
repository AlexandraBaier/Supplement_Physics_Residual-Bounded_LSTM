import argparse
import pathlib
import subprocess

from pbrl_utils import load_environment


def main():
    parser = argparse.ArgumentParser('Run experiments for the Pelican dataset.')
    parser.add_argument('device')
    args = parser.parse_args()

    device_idx = int(args.device)

    main_path = pathlib.Path(__file__).parent.parent.absolute()
    reportin_path = reportout_path = main_path.joinpath('configuration').joinpath('progress-pelican.json')
    environment_path = main_path.joinpath('environment').joinpath('pelican.env')

    environment = load_environment(environment_path)

    if not reportin_path.exists():
        action = 'NEW'
    else:
        action = 'CONTINUE'

    if reportin_path.exists():
        subprocess.call([
            'deepsysid',
            'session',
            '--enable-cuda',
            f'--device-idx={device_idx}',
            f'--reportin={reportin_path}',
            reportout_path,
            action
        ], env=environment)
    else:
        subprocess.call([
            'deepsysid',
            'session',
            '--enable-cuda',
            f'--device-idx={device_idx}',
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
