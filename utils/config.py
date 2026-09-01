"""Command-line configuration for the main New York experiment."""

import argparse

import torch


DEFAULT_OUTPUT_WINDOW = 1
DEFAULT_METRICS = [
    'MAE',
    'MSE',
    'RMSE',
    'MAPE',
]


def my_config(gpu_id=0):
    """Parse the settings required by ``main.py``."""
    device = (
        torch.device(f'cuda:{gpu_id}')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    parser = argparse.ArgumentParser(
        description='Train and evaluate CoM-STGNN on prepared New York data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment settings.
    parser.add_argument('--training', type=int, default=1,
                        help='whether to train the model (1/0)')
    parser.add_argument('--input_window', type=int, default=6,
                        help='number of historical time steps')
    parser.add_argument('--output_window', type=int,
                        default=DEFAULT_OUTPUT_WINDOW,
                        help='number of forecast time steps')
    parser.add_argument('--output_dim', type=int, default=2,
                        help='number of output channels')

    # Prepared taxi data.
    parser.add_argument('--x_train_taxi', default='data/x_train_taxi.npy',
                        help='taxi training inputs')
    parser.add_argument('--y_train_taxi', default='data/y_train_taxi.npy',
                        help='taxi training targets')
    parser.add_argument('--x_val_taxi', default='data/x_val_taxi.npy',
                        help='taxi validation inputs')
    parser.add_argument('--y_val_taxi', default='data/y_val_taxi.npy',
                        help='taxi validation targets')
    parser.add_argument('--x_test_taxi', default='data/x_test_taxi.npy',
                        help='taxi test inputs')
    parser.add_argument('--y_test_taxi', default='data/y_test_taxi.npy',
                        help='taxi test targets')

    # Prepared Citi Bike data.
    parser.add_argument('--x_train_bike', default='data/x_train_bike.npy',
                        help='bike training inputs')
    parser.add_argument('--y_train_bike', default='data/y_train_bike.npy',
                        help='bike training targets')
    parser.add_argument('--x_val_bike', default='data/x_val_bike.npy',
                        help='bike validation inputs')
    parser.add_argument('--y_val_bike', default='data/y_val_bike.npy',
                        help='bike validation targets')
    parser.add_argument('--x_test_bike', default='data/x_test_bike.npy',
                        help='bike test inputs')
    parser.add_argument('--y_test_bike', default='data/y_test_bike.npy',
                        help='bike test targets')

    # Model settings.
    parser.add_argument('--device', default=device, help='computation device')
    parser.add_argument('--num_of_vertices', type=int, default=4761,
                        help='number of graph nodes')
    parser.add_argument('--Ks', type=int, default=3,
                        help='number of Chebyshev graph kernels')
    parser.add_argument('--Kt', type=int, default=3,
                        help='temporal convolution kernel size')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout setting')

    # Optimization and evaluation settings.
    parser.add_argument('--scaler_type', choices=('normal', 'standard',
                                                  'minmax01', 'minmax11',
                                                  'log', 'none'),
                        default='log', help='normalization method')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Adam learning rate')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='training batch size')
    parser.add_argument('--metrics', nargs='+', choices=DEFAULT_METRICS,
                        default=DEFAULT_METRICS.copy(),
                        help='metrics written to the evaluation results')

    return parser.parse_args()
