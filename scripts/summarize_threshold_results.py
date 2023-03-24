import pathlib
from typing import Set, List

import h5py
import numpy as np
import pandas as pd
from deepsysid.pipeline.configuration import ExperimentGridSearchTemplate, ExperimentConfiguration
from deepsysid.pipeline.data_io import build_result_file_name
from deepsysid.pipeline.metrics import NormalizedRootMeanSquaredErrorMetric, BaseMetricConfig
from deepsysid.pipeline.gridsearch import ExperimentSessionReport


def get_best_models(report_path: pathlib.Path) -> Set[str]:
    report = ExperimentSessionReport.parse_file(report_path)

    if report.best_per_class is None or report.best_per_base_name is None:
        raise ValueError(
            'Did not run "deepsysid session" with action=TEST_BEST yet. '
            'Best performing models have not been identified yet.'
        )

    best_models = set(report.best_per_class.values()).union(report.best_per_base_name.values())

    return best_models


def summarize_prediction_scores(
    configuration: ExperimentConfiguration,
    models: Set[str],
    result_directory: pathlib.Path,
    thresholds: List[float]
) -> pd.DataFrame:
    rows = []
    for model in models:
        result_file_name = build_result_file_name(
            mode='test',
            window_size=configuration.window_size,
            horizon_size=configuration.horizon_size,
            extension='hdf5'
        )

        row = [model]
        with h5py.File(
                result_directory.joinpath(model, result_file_name)
        ) as f:
            if 'thresholds' not in f['additional']['bounded_residual']['metadata'].keys():
                print(f'no thresholds for model {model}')
                continue

            for required_threshold in thresholds:
                pred_states = []
                true_states = []

                for idx, threshold in enumerate(f['additional']['bounded_residual']['metadata']['thresholds'][:]):
                    if threshold != required_threshold:
                        continue

                    outputs = f['additional']['bounded_residual'][str(idx)]['outputs']
                    pred_states.append(outputs['pred_state'][:])
                    true_states.append(outputs['true_state'][:])

                if len(pred_states) == 0:
                    print(f'failed to find threshold={required_threshold}')
                    continue

                nrmse_metric = NormalizedRootMeanSquaredErrorMetric(BaseMetricConfig(
                    state_names=configuration.state_names, sample_time=configuration.time_delta
                ))

                nrmse, _ = nrmse_metric.measure(true_states, pred_states)
                nrmse = np.mean(nrmse)
                row.append(nrmse)

        rows.append(row)

    df = pd.DataFrame(
        data=rows,
        columns=['model'] + [f'T={threshold}' for threshold in thresholds]
    )
    return df


def summarize_experiment(
    report_path: pathlib.Path,
    configuration_path: pathlib.Path,
    result_directory: pathlib.Path,
    thresholds: List[float]
) -> None:
    best_models = get_best_models(report_path)

    configuration = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_file(configuration_path)
    )
    n_runs = configuration.session.total_runs_for_best_models

    prediction_scores = summarize_prediction_scores(
        configuration,
        best_models,
        result_directory=result_directory,
        thresholds=thresholds
    )
    prediction_scores['run'] = 0
    for run_idx in range(1, n_runs):
        additional_prediction_scores = summarize_prediction_scores(
            configuration,
            best_models,
            result_directory=result_directory.joinpath(f'repeat-{run_idx}'),
            thresholds=thresholds
        )
        additional_prediction_scores['run'] = run_idx
        prediction_scores = pd.concat((prediction_scores, additional_prediction_scores))

    # https://stackoverflow.com/a/53522680
    stats = prediction_scores\
        .groupby(['model'])[[f'T={threshold}' for threshold in thresholds]]\
        .agg(['mean', 'count', 'std'])
    for threshold in thresholds:
        count = stats[(f'T={threshold}', 'count')]
        std = stats[(f'T={threshold}', 'std')]
        stats[(f'T={threshold}', 'ci95-width')] = 1.96 * std / np.sqrt(count)

    prediction_scores.to_csv(
        result_directory.joinpath('summary-threshold.csv'),
    )
    stats.to_csv(
        result_directory.joinpath('summary-threshold-ci.csv'),
    )


def main():
    main_path = pathlib.Path(__file__).parent.parent.absolute()

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-ship.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('ship.json'),
        result_directory=main_path.joinpath('results').joinpath('ship-ind'),
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 10.0]
    )

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-ship.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('ship.json'),
        result_directory=main_path.joinpath('results').joinpath('ship-ood'),
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 10.0]
    )

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-pelican.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('pelican.json'),
        result_directory=main_path.joinpath('results').joinpath('pelican'),
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 10.0]
    )


if __name__ == '__main__':
    main()
