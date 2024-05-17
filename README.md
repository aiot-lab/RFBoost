# RFBoost

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Datasets and PyTorch code for **RFBoost: Understanding and Boosting Deep WiFi Sensing via Physical Data Augmentation**.

## Prerequisites

- Clone this repo and download the dataset from the [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3008874_connect_hku_hk/EQr23WGSqOlJqlfqf7j6ThQBKT45tbPCEpEgSV9wNhwVrg?e=tNNf3u)(password:hku-aiot-rfboost24):
  ```bash
  unzip NPZ-pp.zip -d "dataset/NPZ-pp/"

- (Optional) Setup cache path in `config.yaml`:
  ```yaml
  cache_folder: "/path/to/cache/dir/"
  ```

- Use Conda to manage python environment:
  ```bash
  % create rfboost-pytorch2
  conda env create -f environment.yml
  ```

## How to Run

1. Start the batch runner with:
   ```bash
   python source/batch_runner.py
   ```
2. If everything goes well, training logs are recorded in `./log/<dataset>/<model>/`, final results are available under `./record`, and TensorBoard logs are located at `./runs`.

## Support methods

The current version supports data augmentation methods for the Widar3 dataset and models using DFS input. In `batch_runner.py` file, uncomment the method you want to use. Available options include "PCA", "All Subcarriers", "RDA" and "ISS-6". 

Note that customized augmentation method will be defined in "augment.py". (TODO: We will refactor the definition logic in the future.)

## Files and Directories

### About `source/batch_runner.py`: task runner
This is a multi-task queue that allows multiple augmentation combinations to be submitted at once to test performance. Currently, you can adjust the Dataset, Model, default_window, augmentation, hyperparameters, and so on.

By default, it uses the RFNet model and Widar3 dataset with default parameters, testing for Cross-RX evalution.

### About `dataset/` & `source/Datasets.py`: Dataset and Splits
The original data are stored in the `dataset/` directory, but for different tasks, different data splits are needed. So that we save split files in the `source/<model>/` folder. By default, `main.py` also supports K-fold cross-validation.

### About `source/augment.py`: Repo of augmentation methods
Users can write their own augmentation rules in this file.

## Notes

This repository is built upon [UniTS repo](https://github.com/Shuheng-Li/UniTS-Sensory-Time-Series-Classification). We owe our gratitude for their initial work.

## Citation

```
@article{hou2024rfboost,
author = {Hou, Weiying and Wu, Chenshu},
title = {RFBoost: Understanding and Boosting Deep WiFi Sensing via Physical Data Augmentation},
year = {2024},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
}
```

## License

This project is licensed under the GPL v3 License - see the [LICENSE](source/LICENSE) file for details.
