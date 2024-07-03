# SimGAN

Implementation of [SimGAN](https://arxiv.org/pdf/1612.07828v1) using Tensorflow 2.

## Getting Started

1. Install required python packages

```
pip install -r requirements.txt
```

2. Dataset

You can find the dataset from Kaggle [here](https://www.kaggle.com/datasets/4quant/eye-gaze). Both real (MPIIGaze) and Simulated (UnityEyes) can be found there in ready to go h5 format. ( `real_gaze.h5` and `gaze.h5` )

3. Training

To start training, run the following script:

```
python train.py <real_h5_file> <simulated_h5_file> [--refiner_model <REF_MODEL_PATH>] [--discriminator_model <DISC_MODEL_PATH>]
```

**Args**:
- `<real_h5_file>` - Path to the real dataset in h5 format (`real_gaze.h5`).
- `<synthetic_h5_file>` - Path to the synthetic dataset in h5 format (`gaze.h5`)
- `<REF_MODEL_PATH>` - (Optional) Refiner model to start training from.
- `<DISC_MODEL_PATH>` - (Optional) Discriminator model to start training from.

- **Output**: Every [`DEBUG_INTERVAL`](https://github.com/mnjm/SimGAN/blob/main/simgan.py#L27) steps, Intermediate model checkpoints (refiner and discriminator) are saved in the `cache` directory.
