# World Models in JAX

A JAX-based reimplementation of the 2018 *World Models* paper by Ha & Schmidhuber, and variants of the Dreamer models by Danijar Hafner.

To build the environment, just run uv sync from the root directory.
```python
uv sync
```

# *World Models* by Ha & Schmidhuber
## Trained Policy
Here's an evolved policy solving the CarRacing environment - achieving 906 reward.

[![Evolved policy solving the CarRacing environment](assets/policy.gif)](assets/policy.mp4)

It's a bit wonky, with the car oscillating from side-to-side, which I reckon is for two reasons:
1. Data collection via random initialisation doesn't cover that much of the policy space, so most of the random data collection looks like sticky actions: the car either drives forwards, sits still, or spins left/right. That means the VAE doesn't learn to distinguish road position very well and the signal for the policy is limited. To be fair the policy does fairly well (IQM 850 reward - see `notebooks/train_controller.ipynb`), so I'm not sure whether the oscillation is harming performance that much.
2. The paper uses an evolutionary method called CMA-ES to train the car driving controller. This is horribly slow, and it's all CPU-bound because the CarRacing environment uses Box2D. I used something like 25x less compute (which still took 8 hours on a 16-core machine) for this final optimisation step than the original paper, so the controller is pretty suboptimal.

## How to Run It
There are a few steps to the paper. You need to:
1. Generate data with random agents (that's agents with different random initialisations, not random action sampling)
2. Train a VAE to reconstruct the images
2. Use the VAE to label the dataset with latent vectors, for use in RNN training.
3. Train an RNN to predict trajectories in the VAE latent space
4. Train a controller which only receives the RNN hidden state and VAE latent as inputs, to control the car.

In this repo, these are done by:
```bash
uv run scripts/collect_data.py
````
```bash
uv run scripts/train_vae.py
````
```bash
uv run scripts/label_vae_dataset.py
````
```bash
uv run scripts/train_rnn.py
````
```bash
uv run scripts/train_controller.py
````

Each script exposes command-line arguments - run `uv run ... --help` to get a list of arguments.

## Imagining
You can also interact with the world model yourself by running

```bash
uv run scripts/imagine.py
```



# Tests
To run tests, from root run
```bash
uv run pytest
```
