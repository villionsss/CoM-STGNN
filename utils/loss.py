"""Metrics computed over valid target values."""

import torch


def mae_torch(preds, labels, null_val=0):
    labels[torch.abs(labels) < 1e-4] = 0
    mask = labels.ne(null_val).float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    error = torch.abs(preds - labels) * mask
    error = torch.where(torch.isnan(error), torch.zeros_like(error), error)
    return torch.mean(error)


def mape_torch(preds, labels, null_val=0, eps=1e-8):
    labels[torch.abs(labels) < 1e-4] = 0
    mask = labels.ne(null_val).float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    denominator = (torch.abs(labels) + torch.abs(preds)) / 2 + eps
    error = torch.abs(preds - labels) / denominator
    error = error * mask
    error = torch.where(torch.isnan(error), torch.zeros_like(error), error)
    return torch.mean(error)


def mse_torch(preds, labels, null_val=0):
    labels[torch.abs(labels) < 1e-4] = 0
    mask = labels.ne(null_val).float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    error = torch.square(preds - labels) * mask
    error = torch.where(torch.isnan(error), torch.zeros_like(error), error)
    return torch.mean(error)


def rmse_torch(preds, labels, null_val=0):
    return torch.sqrt(mse_torch(preds, labels, null_val))
