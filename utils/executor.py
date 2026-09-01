"""Training, evaluation, logging, and checkpoint utilities."""

import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import evaluator


def train(model, train_dataloader, eval_dataloader, config):
    """Train the model and save the best validation checkpoint."""
    print(get_local_time(), 'Train starting')
    best_loss = float('inf')
    patience = 0

    for epoch in range(config.epochs):
        if patience > 10:
            break

        train_loss = _train_epoch(model, train_dataloader, config)
        validation_loss = _valid_epoch(model, eval_dataloader, config)

        if validation_loss < best_loss:
            best_loss = validation_loss
            patience = 0
            torch.save(model.state_dict(), config.checkpoint_path)
        else:
            patience += 1

        learning_rate = config.optimizer.state_dict()['param_groups'][0]['lr']
        loss_info = (
            f'Epochs: {epoch}/{config.epochs}  '
            f'Train loss: {train_loss:.6f}  '
            f'Val loss: {validation_loss:.6f}  '
            f'Learning rate:{learning_rate:.6f}'
        )
        logging(loss_info, config)
        config.writer.add_scalar('Train loss', train_loss, epoch)
        config.writer.add_scalar('Val loss', validation_loss, epoch)
        config.writer.add_scalar('Learning rate', learning_rate, epoch)

        print(get_local_time(), loss_info)
        torch.cuda.empty_cache()
        config.scheduler.step()

    print(get_local_time(), 'Train finished!')


def evaluate(model, test_dataloader, config, show=None):
    """Evaluate the model, save metrics, and export predictions."""
    del show

    with torch.no_grad():
        model.eval()
        truths = []
        predictions = []
        losses = []

        for batch in test_dataloader:
            bike = batch[0].to(config.device)
            taxi = batch[1].to(config.device)
            bike_target = batch[2].to(config.device)
            taxi_target = batch[3].to(config.device)

            output, _ = model.predict(bike, taxi)
            batch_loss = model.calculate_loss(
                bike, taxi, bike_target, taxi_target
            )
            losses.append(batch_loss.item())

            bike_target = config.scaler_bike.inverse_transform(bike_target)
            taxi_target = config.scaler_taxi.inverse_transform(taxi_target)
            bike_prediction = config.scaler_bike.inverse_transform(output[..., 0:1])
            taxi_prediction = config.scaler_taxi.inverse_transform(output[..., 1:2])

            bike_target = torch.tensor(bike_target)
            taxi_target = torch.tensor(taxi_target)
            bike_prediction = torch.tensor(bike_prediction)
            taxi_prediction = torch.tensor(taxi_prediction)

            truths.append(torch.cat((bike_target, taxi_target), dim=3).numpy())
            predictions.append(
                torch.cat((bike_prediction, taxi_prediction), dim=3).numpy()
            )

        _, time_steps, nodes, channels = predictions[0].shape
        predictions = np.concatenate(predictions, axis=0)
        truths = np.concatenate(truths, axis=0)

        mode_names = ('bike', 'taxi')
        for channel in range(channels):
            channel_prediction = predictions[:, :, :, channel]
            channel_truth = truths[:, :, :, channel]
            channel_prediction = channel_prediction.reshape(-1, time_steps, nodes)
            channel_truth = channel_truth.reshape(-1, time_steps, nodes)

            result = evaluator.collect(
                torch.tensor(channel_prediction),
                torch.tensor(channel_truth),
                config,
            )
            mode = mode_names[channel] if channel < len(mode_names) else f'channel_{channel}'
            test_result = evaluator.save_result(result, config, mode)
            save_raw_result(
                channel_prediction,
                channel_truth,
                config.result_path.replace('.csv', ''),
                mode,
            )

        print('loss:', np.mean(losses))
        print(test_result)


def _train_epoch(model, train_dataloader, config):
    model.train()
    losses = []
    for batch in train_dataloader:
        config.optimizer.zero_grad()
        bike = batch[0].to(config.device)
        taxi = batch[1].to(config.device)
        bike_target = batch[2].to(config.device)
        taxi_target = batch[3].to(config.device)

        batch_loss = model.calculate_loss(
            bike, taxi, bike_target, taxi_target
        )
        losses.append(batch_loss.item())
        batch_loss.backward()
        config.optimizer.step()
    return np.mean(losses)


def _valid_epoch(model, eval_dataloader, config):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_dataloader:
            bike = batch[0].to(config.device)
            taxi = batch[1].to(config.device)
            bike_target = batch[2].to(config.device)
            taxi_target = batch[3].to(config.device)

            batch_loss = model.calculate_loss(
                bike, taxi, bike_target, taxi_target
            )
            losses.append(batch_loss.item())
    return np.mean(losses)


def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y %H:%M:%S')


def logging(log_info, config):
    with open(config.logging_path, 'a+', encoding='utf-8') as file:
        file.write(f'{get_local_time()}:{log_info}\n')


def init_path(config):
    """Create output directories and attach their paths to ``config``."""
    run_name = _time_suffix()
    config.run_name = run_name
    config.model_dir = os.path.join('save', 'model', run_name)
    config.checkpoint_path = os.path.join(
        config.model_dir, f'{run_name}_bike_taxi.pt'
    )
    config.logging_path = os.path.join(
        'save', 'logging', f'{run_name}_training.log'
    )
    config.result_path = os.path.join(
        'save', 'result', f'{run_name}.csv'
    )
    tensorboard_dir = os.path.join(
        'save', 'logging', 'tensorboard', run_name
    )

    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.logging_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.result_path), exist_ok=True)
    config.writer = SummaryWriter(tensorboard_dir)


def _time_suffix():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def save_raw_result(predict, truth, path_base, mode):
    """Save predictions and targets for every forecast horizon."""
    prediction_by_time = np.transpose(predict, (1, 0, 2))
    truth_by_time = np.transpose(truth, (1, 0, 2))

    for time_step, values in enumerate(prediction_by_time):
        path = f'{path_base}_{mode}_timestep_{time_step}_predict.csv'
        pd.DataFrame(values).to_csv(path)

    for time_step, values in enumerate(truth_by_time):
        path = f'{path_base}_{mode}_timestep_{time_step}_true.csv'
        pd.DataFrame(values).to_csv(path)
