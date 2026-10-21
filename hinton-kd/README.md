# Hinton Knowledge Distillation Reproduction

## Objective

Reproduce the core knowledge distillation method introduced in:

Hinton, G., Vinyals, O., and Dean, J.  
*Distilling the Knowledge in a Neural Network* (2015).

## Environment

- OS: Ubuntu 22.04 on WSL2
- Python: 3.11
- Environment manager: uv
- Deep learning framework: PyTorch
- Initial device: CPU、GPU

## Experiment Plan

1. Train a teacher model using supervised learning.
2. Train a smaller student model using ground-truth labels.
3. Train another student using ground-truth labels and teacher soft targets.
4. Compare accuracy, model size, and inference latency.
5. Study the effects of temperature and distillation loss weight.

## Reproducibility

- Record dependency versions in `uv.lock`.
- Record random seeds and hyperparameters.
- Keep training and evaluation procedures consistent across experiments.
- Save experiment logs and results under `outputs/`.

## Reference

Hinton et al., 2015: https://arxiv.org/abs/1503.02531
