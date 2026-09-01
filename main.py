"""Main CoM-STGNN model and the prepared New York training workflow."""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from model import Layer_GCN, Layer_TGCN, Layer_attention
from utils.normalization import (
    LogScaler,
    MinMax01Scaler,
    MinMax11Scaler,
    NoneScaler,
    NormalScaler,
    StandardScaler,
)
from utils import executor
from utils import config as Config


def get_scalar(scaler_type, x_train, y_train):
    """Create the configured scaler from the training observations."""
    if scaler_type == "normal":
        scaler = NormalScaler(maxx=x_train.max())

    elif scaler_type == "standard":
        scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())

    elif scaler_type == "minmax01":
        scaler = MinMax01Scaler(
            maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))

    elif scaler_type == "minmax11":
        scaler = MinMax11Scaler(
            maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))

    elif scaler_type == "log":
        scaler = LogScaler()

    elif scaler_type == "none":
        scaler = NoneScaler()

    else:
        raise ValueError('Unknown scaler type.')
    return scaler


def scaler_data(config, x_train, y_train, x_val, y_val, x_test, y_test, scaler=None):
    """Normalize each split and pair each input sequence with its target."""
    output_dim = config.output_dim

    if scaler is None:
        scaler = get_scalar(config.scaler_type, x_train[..., :output_dim], y_train[..., :output_dim])

    x_train[..., :output_dim] = scaler.transform(x_train[..., :output_dim])
    y_train[..., :output_dim] = scaler.transform(y_train[..., :output_dim])
    x_val[..., :output_dim] = scaler.transform(x_val[..., :output_dim])
    y_val[..., :output_dim] = scaler.transform(y_val[..., :output_dim])
    x_test[..., :output_dim] = scaler.transform(x_test[..., :output_dim])
    y_test[..., :output_dim] = scaler.transform(y_test[..., :output_dim])

    train_data = list(zip(x_train, y_train))
    eval_data = list(zip(x_val, y_val))
    test_data = list(zip(x_test, y_test))

    return train_data, eval_data, test_data, scaler


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_pred, y_true, mask_value=0):
        """Compute MSE over valid (non-negative) target values."""
        mask = (y_true >= 0).float()
        mse = (y_pred - y_true) ** 2
        masked_mse = mse * mask
        loss = masked_mse.sum() / mask.sum()
        return loss


class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, view1, view2):
        view1 = torch.mean(view1, dim=-1)
        view2 = torch.mean(view2, dim=-1)
        cos_s = 1 - self.cos_sim(view1, view2)
        mean_s = torch.mean(cos_s)
        return mean_s


class my_model(nn.Module):
    def __init__(self, config):
        super(my_model, self).__init__()
        emd_dim = 16
        hide = 32
        out = 128
        self.out_dim = config.output_dim
        device = config.device

        self.aux_loss = AuxLoss()

        # Optional kernel regeneration. The source adjacency matrices are not
        # included in this public release; the prepared kernels below are used
        # by default. Uncomment this block only after supplying those matrices.
        #
        # od_neighbor = np.load('data/od_adjacency_matrix.npy')
        # od_flow_bike = np.load('data/bike_od_flowmatrix.npy')
        # od_flow_taxi = np.load('data/taxi_od_flowmatrix.npy')
        # od_builtupenv = np.load('data/builtupenv_matrix.npy')
        #
        # laplacian_mx = Layer_GCN.calculate_scaled_laplacian(od_neighbor)
        # sa_Lk = torch.as_tensor(
        #     Layer_GCN.calculate_cheb_poly(laplacian_mx, config.Ks),
        #     dtype=torch.float32,
        # ).to(config.device)
        # torch.save(sa_Lk.cpu(), 'data/sa_Lk.pt')
        #
        # laplacian_mx = Layer_GCN.calculate_scaled_laplacian(od_flow_bike)
        # od_bike_Lk = torch.as_tensor(
        #     Layer_GCN.calculate_cheb_poly(laplacian_mx, config.Ks),
        #     dtype=torch.float32,
        # ).to(config.device)
        # torch.save(od_bike_Lk.cpu(), 'data/od_bike_Lk.pt')
        #
        # laplacian_mx = Layer_GCN.calculate_scaled_laplacian(od_flow_taxi)
        # od_taxi_Lk = torch.as_tensor(
        #     Layer_GCN.calculate_cheb_poly(laplacian_mx, config.Ks),
        #     dtype=torch.float32,
        # ).to(config.device)
        # torch.save(od_taxi_Lk.cpu(), 'data/od_taxi_Lk.pt')
        #
        # laplacian_mx = Layer_GCN.calculate_scaled_laplacian(od_builtupenv)
        # od_builtupenv_Lk = torch.as_tensor(
        #     Layer_GCN.calculate_cheb_poly(laplacian_mx, config.Ks),
        #     dtype=torch.float32,
        # ).to(config.device)
        # torch.save(od_builtupenv_Lk.cpu(), 'data/od_builtupenv_Lk.pt')

        # Load the prepared graph kernels used by the six model branches.
        sa_Lk = torch.load('data/sa_Lk.pt', map_location=config.device)
        od_bike_Lk = torch.load(
            'data/od_bike_Lk.pt', map_location=config.device
        )
        od_taxi_Lk = torch.load(
            'data/od_taxi_Lk.pt', map_location=config.device
        )
        od_builtupenv_Lk = torch.load(
            'data/od_builtupenv_Lk.pt', map_location=config.device
        )

        self.output_window = config.output_window
        self.training = config.training
        self.loss = MaskedMSELoss()

        self.bike_sa = Layer_TGCN.TGCN_M(config, sa_Lk, emd_dim, hide, out)
        self.taxi_sa = Layer_TGCN.TGCN_M(config, sa_Lk, emd_dim, hide, out)
        print(executor.get_local_time(), 'Spatial branches initialized')

        self.bike_od = Layer_TGCN.TGCN_M(config, od_bike_Lk, emd_dim, hide, out)
        self.taxi_od = Layer_TGCN.TGCN_M(config, od_taxi_Lk, emd_dim, hide, out)
        print(executor.get_local_time(), 'OD branches initialized')

        self.bike_be = Layer_TGCN.TGCN_M(
            config, od_builtupenv_Lk, emd_dim, hide, out
        )
        self.taxi_be = Layer_TGCN.TGCN_M(
            config, od_builtupenv_Lk, emd_dim, hide, out
        )
        print(executor.get_local_time(), 'Built-environment branches initialized')


        self.attention_global = Layer_attention.SelfAttention(out, 2)
        self.attention_local = Layer_attention.SelfAttention(out, 2)
        self.attention_be = Layer_attention.SelfAttention(out, 2)

        print(executor.get_local_time(), 'Attention layers initialized')

        self.weight2 = nn.init.xavier_uniform_(nn.Parameter(
            torch.FloatTensor(1, config.num_of_vertices, out).to(device)
        ))
        self.weight1 = nn.init.xavier_uniform_(nn.Parameter(
            torch.FloatTensor(1, config.num_of_vertices, out).to(device)
        ))
        self.weight3 = nn.init.xavier_uniform_(nn.Parameter(
            torch.FloatTensor(1, config.num_of_vertices, out).to(device)
        ))

        self.out_fc = nn.Linear(out, config.output_window * self.out_dim)

    def forward(self, bike, taxi):
        """Return the forecast and the cross-stream alignment loss."""
        bike_sa_x = self.bike_sa(bike)
        taxi_sa_x = self.taxi_sa(taxi)

        bike_od_x = self.bike_od(bike)
        taxi_od_x = self.taxi_od(taxi)

        bike_be_x = self.bike_be(bike)
        taxi_be_x = self.taxi_be(taxi)

        B, N, _ = bike_sa_x.shape

        global_x_att = self.attention_global([bike_sa_x, taxi_sa_x])
        local_x_att = self.attention_local([bike_od_x, taxi_od_x])
        be_x_att = self.attention_be([bike_be_x, taxi_be_x])

        loss_aux = (
            self.aux_loss(bike_sa_x, taxi_sa_x)
            + self.aux_loss(bike_od_x, taxi_od_x)
            + self.aux_loss(bike_be_x, taxi_be_x)
        )

        x = (
            local_x_att * self.weight1
            + global_x_att * self.weight2
            + be_x_att * self.weight3
        )

        out_x = self.out_fc(x).view(B, N, self.output_window, self.out_dim)
        out_x1 = out_x.permute(0, 2, 1, 3)

        return out_x1, loss_aux

    def calculate_loss(self, bike, taxi, y_bike, y_taxi):
        y_predicted, loss_aux = self.predict(bike, taxi)
        if self.out_dim == 1:
            y_true = y_bike
        else:
            y_true = torch.cat((y_bike, y_taxi), dim=3)

        loss_mae = self.loss(y_predicted, y_true, mask_value=0)
        loss = loss_mae + loss_aux

        return loss

    def predict(self, bike, taxi):
        """Run the forward pass."""
        y_preds, loss_aux = self.forward(bike, taxi)
        return y_preds, loss_aux


class Bike_Taxi_Dataset(Dataset):
    def __init__(self, config, bike_data, taxi_data):
        self.bike_data = bike_data
        self.taxi_data = taxi_data

    def __len__(self):
        return len(self.taxi_data)

    def __getitem__(self, idx):
        bike = self.bike_data[idx][0]
        taxi = self.taxi_data[idx][0]

        target_bike = self.bike_data[idx][1]
        target_taxi = self.taxi_data[idx][1]

        return bike, taxi, target_bike, target_taxi


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Load prepared New York data and train or evaluate the model."""
    setup_seed(20)
    config = Config.my_config(gpu_id=0)
    executor.init_path(config)

    x_train_taxi = np.load(config.x_train_taxi).astype(np.float32)
    y_train_taxi = np.load(config.y_train_taxi).astype(np.float32)
    x_val_taxi = np.load(config.x_val_taxi).astype(np.float32)
    y_val_taxi = np.load(config.y_val_taxi).astype(np.float32)
    x_test_taxi = np.load(config.x_test_taxi).astype(np.float32)
    y_test_taxi = np.load(config.y_test_taxi).astype(np.float32)

    x_train_bike = np.load(config.x_train_bike).astype(np.float32)
    y_train_bike = np.load(config.y_train_bike).astype(np.float32)
    x_val_bike = np.load(config.x_val_bike).astype(np.float32)
    y_val_bike = np.load(config.y_val_bike).astype(np.float32)
    x_test_bike = np.load(config.x_test_bike).astype(np.float32)
    y_test_bike = np.load(config.y_test_bike).astype(np.float32)
    print(executor.get_local_time(), 'Prepared data loaded')

    scaler_bike = get_scalar(config.scaler_type, x_train_bike, x_train_bike)
    scaler_taxi = get_scalar(config.scaler_type, x_train_taxi, x_train_taxi)
    bike_train_data, bike_eval_data, bike_test_data, scaler_bike = scaler_data(
        config,
        x_train_bike,
        y_train_bike,
        x_val_bike,
        y_val_bike,
        x_test_bike,
        y_test_bike,
        scaler_bike,
    )
    taxi_train_data, taxi_eval_data, taxi_test_data, scaler_taxi = scaler_data(
        config,
        x_train_taxi,
        y_train_taxi,
        x_val_taxi,
        y_val_taxi,
        x_test_taxi,
        y_test_taxi,
        scaler_taxi,
    )

    train_dataset = Bike_Taxi_Dataset(config, bike_train_data, taxi_train_data)
    val_dataset = Bike_Taxi_Dataset(config, bike_eval_data, taxi_eval_data)
    test_dataset = Bike_Taxi_Dataset(config, bike_test_data, taxi_test_data)

    model = my_model(config).to(config.device)
    config.optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=0.000
    )
    config.scheduler = torch.optim.lr_scheduler.StepLR(
        config.optimizer, step_size=50, gamma=0.8
    )
    config.scaler_bike = scaler_bike
    config.scaler_taxi = scaler_taxi

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    log_info = "Prepared dataset ready"
    print(executor.get_local_time(), log_info)
    executor.logging(log_info, config)

    if config.training == 1:
        executor.train(model, train_loader, test_loader, config)

    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint, strict=False)
    executor.evaluate(model, val_loader, config, show=[])


if __name__ == '__main__':
    main()
