"""Evaluation metrics and result export for the main experiment."""

import numpy as np
import pandas as pd

from utils import loss


METRIC_FUNCTIONS = {
    'MAE': loss.mae_torch,
    'MSE': loss.mse_torch,
    'RMSE': loss.rmse_torch,
    'MAPE': loss.mape_torch,
}


def collect(y_pred, y_true, config):
    """Compute the configured metrics for every forecast horizon."""
    metrics = [metric for metric in config.metrics if metric in METRIC_FUNCTIONS]
    results = {
        f'{metric}@{step}': []
        for step in range(1, config.output_window + 1)
        for metric in metrics
    }

    for step in range(1, config.output_window + 1):
        prediction = y_pred[:, step - 1]
        target = y_true[:, step - 1]

        for metric in metrics:
            metric_function = METRIC_FUNCTIONS.get(metric)
            if metric_function is None:
                continue
            value = metric_function(prediction, target, 0)

            results[f'{metric}@{step}'].append(float(value))

    return results


def save_result(intermediate_result, config, mode):
    """Write one row per forecast horizon to the configured result file."""
    metrics = [metric for metric in config.metrics if metric in METRIC_FUNCTIONS]
    data = {metric: [] for metric in metrics}
    for step in range(1, config.output_window + 1):
        for metric in metrics:
            data[metric].append(
                np.mean(intermediate_result[f'{metric}@{step}'])
            )

    dataframe = pd.DataFrame(
        data,
        index=range(1, config.output_window + 1),
    )
    path = config.result_path.replace('.csv', f'_{mode}.csv')
    dataframe.to_csv(path, index=False)
    return dataframe
