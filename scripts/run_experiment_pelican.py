import argparse
import pathlib

from pbrl.utils import load_environment, run_full_gridsearch_session


def main():
    parser = argparse.ArgumentParser('Run experiments for the Pelican dataset.')
    parser.add_argument('device')
    args = parser.parse_args()

    device_idx = int(args.device)

    main_path = pathlib.Path(__file__).parent.parent.absolute()
    report_path = main_path.joinpath('configuration').joinpath('progress-pelican.json')
    environment_path = main_path.joinpath('environment').joinpath('pelican.env')

    environment = load_environment(environment_path)

    run_full_gridsearch_session(
        report_path=report_path,
        device_idx=device_idx,
        environment=environment
    )


if __name__ == '__main__':
    main()
