# partial-cliques

[Installation](#installation) - [Example](#example) - [Citation](#citation)


## Installation
To avoid any conflict with your existing Python setup, we suggest to work in a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
Follow these [instructions](https://github.com/google/jax#installation) to install the version of JAX corresponding to your versions of CUDA and CuDNN.
```bash
git clone git@github.com:CristianManta/partial-cliques.git
cd partial-cliques
pip install -r requirements.txt
```

## Example
You can train partial-cliques on a randomly generated dataset of 100 observations over 5 latent variables using the following command:
```bash
python train.py --off_wandb --batch_size 256 --num_variables 5 --num_samples 100
```