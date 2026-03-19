# WorldNN

A simulation framework for studying how information loss in perception-action loops determines whether an agent can learn to act effectively.

## What it does

WorldNN models a complete perception-action chain where an organism must push a rock to a target through lossy channels:

```
Matter (4D state) → Emission → Channel (noise) → VAE → Organism → Action
```

Every stage is independently controllable: perception quality, channel noise, VAE compression, and organism embedding capacity.

## Result

Coordination quality **C_i** — cosine alignment between the learned policy and the optimal action — predicts task performance with **r = -0.87** across 50 conditions. C_i ≥ 0.6 guarantees learning; C_i < 0.5 guarantees failure.

![C_i predicts performance](results/obj013_coordination_quality.png)

## Usage

```bash
pip install -e ".[dev]"
pytest tests/ -v
python experiments/coordination_quality.py  # GPU recommended
```

## Structure

```
src/worldnn/     # Core: matter, channels, environment (VAE), organism, training
experiments/     # One script per objective (13 completed)
results/         # Figures and data
```
