# CoM-STGNN

## 1. Overview

CoM-STGNN is a multi-view spatio-temporal graph neural network for the joint
prediction of multimodal origin-destination (OD) travel demand. This
repository is the public release of the model implementation and the
processed input data for the New York case study. The paper evaluates the
model using both Wuhan and New York case studies; however, the Wuhan data
cannot be released publicly, so this repository provides only the New York
case study with its processed input data and prepared graph kernels.

The model is presented in the following paper:

> Sun, X., Zhou, Y., Li, Q., and Thill, J.-C. (2026). A Multi-View
> Spatio-Temporal Learning Model for the Co-Prediction of Multimodal
> Origin-Destination Travel Demand. *Transactions in GIS*, 30, e70323.

Paper: <https://onlinelibrary.wiley.com/doi/abs/10.1111/tgis.70323>

## 2. Model

CoM-STGNN jointly predicts taxi and shared-bike OD flows in Manhattan. It
uses a flow-centric dual graph that represents OD flows as graph nodes, so
relationships between flows can be modeled directly rather than being
represented only as edges between regions.

The model contains three complementary graph views for each transportation
mode:

- **Spatial adjacency graph:** represents spatial relationships between OD
  flows with related origins and destinations.
- **Mobility feature graph:** represents relationships associated with flow
  intensity and high-volume OD flows.
- **Built-environment and flow semantic graph:** represents semantic
  similarity learned from built-environment indicators and multimodal flow
  dynamics.

The view-specific branches use GCN and TGCN modules to learn intra-modal
spatio-temporal features. The self-attention module then learns interactions
between transportation modes within each view. Finally, learnable view
weights fuse the complementary features into a joint multimodal forecast. The
training objective combines the forecasting loss with a cross-modal
feature-alignment loss.

## 3. Code and data

### Running the model

Run the following commands from the repository root:

```bash
pip install -r requirements.txt
python main.py --training 1 --epochs 500
```

The default configuration uses six historical half-hour time steps to predict
the next half-hour step. It uses log normalization, Adam optimization with a
learning rate of `0.001`, batch size `4`, dropout `0.3`, three graph kernels,
and a temporal kernel size of `3`. CUDA is selected automatically
when available.

The command uses the supplied processed arrays and graph kernels directly.
Checkpoints, logs, and evaluation outputs are saved under `save/`. Generated
result files are timestamped and include the corresponding mode name.

### Large data files

The prepared arrays and graph kernels are managed with Git Large File Storage
(Git LFS) because of their size. Install Git LFS before uploading or cloning
the repository. For the initial upload, run the following commands from the
repository root:

```bash
git lfs install
git add .gitattributes .gitignore README.md main.py model utils data
git commit -m "Add CoM-STGNN public release"
git push
```

After cloning, download the actual data objects with:

```bash
git lfs install
git lfs pull
```

Without Git LFS, the data files may appear as pointer files rather than the
prepared arrays and graph kernels. See the
[GitHub Git LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
for additional details.

### Prepared New York data

The included processed data correspond to the New York experiment in the
paper.

| Item | Setting |
| --- | --- |
| Transportation modes | Taxi and Citi Bike |
| Study area | Manhattan, New York |
| Spatial units | 69 predefined taxi zones |
| Time resolution | 0.5 hour |
| Historical input | 6 time steps (3 hours) |
| Forecast horizon | 1 time step (0.5 hour) |
| Data period | July-September 2023 |
| Split | 70% training, 10% validation, 20% testing |

The source data include TLC taxi trips, Citi Bike trips, street-centerline
data, and points-of-interest data.

### Repository contents

- `main.py`: model definition, complete training and evaluation workflow, and
  the commented optional kernel-regeneration block.
- `model/Layer_GCN.py`: GCN module and optional graph-kernel generation
  helpers.
- `model/Layer_TGCN.py`: TGCN module with graph convolution.
- `model/Layer_attention.py`: self-attention module for cross-modal feature
  fusion.
- `utils/config.py`: command-line and experiment configuration.
- `utils/normalization.py`: input and target normalization.
- `utils/loss.py`: training loss and metric functions.
- `utils/evaluator.py`: metric aggregation and result export.
- `utils/executor.py`: training, validation, checkpointing, and prediction
  export.
- `data/`: processed taxi and Citi Bike arrays and precomputed graph kernels.

The `data/` directory contains the following prepared files:

- `x_train_bike.npy`, `y_train_bike.npy`: Citi Bike training inputs and
  targets.
- `x_train_taxi.npy`, `y_train_taxi.npy`: taxi training inputs and targets.
- `x_val_bike.npy`, `y_val_bike.npy`: Citi Bike validation inputs and targets.
- `x_val_taxi.npy`, `y_val_taxi.npy`: taxi validation inputs and targets.
- `x_test_bike.npy`, `y_test_bike.npy`: Citi Bike test inputs and targets.
- `x_test_taxi.npy`, `y_test_taxi.npy`: taxi test inputs and targets.
- `sa_Lk.pt`: shared spatial graph kernels.
- `od_bike_Lk.pt`: Citi Bike OD graph kernels.
- `od_taxi_Lk.pt`: taxi OD graph kernels.
- `od_builtupenv_Lk.pt`: built-environment graph kernels.
