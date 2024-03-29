import pathlib
from typing import Set, List

import h5py
import numpy as np
import pandas as pd
from deepsysid.pipeline.configuration import ExperimentGridSearchTemplate, ExperimentConfiguration
from deepsysid.pipeline.data_io import build_score_file_name, build_explanation_result_file_name
from deepsysid.pipeline.evaluation import ReadableEvaluationScores
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
    horizons: List[int]
) -> pd.DataFrame:
    rows = []
    for model in models:
        score_file_name = build_score_file_name(
            mode='test',
            window_size=configuration.window_size,
            horizon_size=configuration.horizon_size,
            extension='json'
        )
        scores = ReadableEvaluationScores.parse_file(
            result_directory.joinpath(model).joinpath(score_file_name)
        )
        row = [model]
        for horizon in horizons:
            nrmse_multi = np.mean(
                scores.scores_per_horizon[horizon]['nrmse']
            )
            row.append(nrmse_multi)
        rows.append(row)

    df = pd.DataFrame(
        data=rows,
        columns=['model'] + [f'H={horizon}' for horizon in horizons]
    )
    return df


def summarize_experiment(
    report_path: pathlib.Path,
    configuration_path: pathlib.Path,
    result_directory: pathlib.Path,
    horizons: List[int]
) -> None:
    best_models = get_best_models(report_path)

    configuration = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_file(configuration_path)
    )
    n_runs = configuration.session.total_runs_for_best_models

    prediction_scores = summarize_prediction_scores(
        configuration,
        best_models,
        result_directory,
        horizons
    )
    prediction_scores['run'] = 0
    for run_idx in range(1, n_runs):
        additional_prediction_scores = summarize_prediction_scores(
            configuration,
            best_models,
            result_directory=result_directory.joinpath(f'repeat-{run_idx}'),
            horizons=horizons
        )
        additional_prediction_scores['run'] = run_idx
        prediction_scores = pd.concat((prediction_scores, additional_prediction_scores))

    # https://stackoverflow.com/a/53522680
    stats = prediction_scores\
        .groupby(['model'])[[f'H={horizon}' for horizon in horizons]]\
        .agg(['mean', 'count', 'std'])
    for horizon in horizons:
        mean = stats[(f'H={horizon}', 'mean')]
        count = stats[(f'H={horizon}', 'count')]
        std = stats[(f'H={horizon}', 'std')]
        stats[(f'H={horizon}', 'ci95-width')] = 1.96 * std / np.sqrt(count)
        stats[(f'H={horizon}', 'ci95-lo')] = mean - stats[(f'H={horizon}', 'ci95-width')]
        stats[(f'H={horizon}', 'ci95-hi')] = mean + stats[(f'H={horizon}', 'ci95-width')]

    prediction_scores.to_csv(
        result_directory.joinpath('summary-prediction.csv'),
    )
    stats.to_csv(
        result_directory.joinpath('summary-prediction-ci.csv'),
    )


def main():
    main_path = pathlib.Path(__file__).parent.parent.absolute()

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-ship.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('ship.json'),
        result_directory=main_path.joinpath('results').joinpath('ship-ind'),
        horizons=[1, 450, 900]
    )

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-ship.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('ship.json'),
        result_directory=main_path.joinpath('results').joinpath('ship-ood'),
        horizons=[1, 450, 900]
    )

    summarize_experiment(
        report_path=main_path.joinpath('configuration').joinpath('progress-pelican.json'),
        configuration_path=main_path.joinpath('configuration').joinpath('pelican.json'),
        result_directory=main_path.joinpath('results').joinpath('pelican'),
        horizons=[1, 50, 100]
    )


if __name__ == '__main__':
    main()
