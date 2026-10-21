# Hinton Knowledge Distillation Reproduction

## Objective

Reproduce the core knowledge distillation method introduced in:

Hinton, G., Vinyals, O., and Dean, J.  
*Distilling the Knowledge in a Neural Network* (2015).


## Structure
KD_Learning/                          # git 根 = 包根
├── pyproject.toml                    # 上移;包名 src → kdlib(可自定)
├── uv.lock / .python-version / .gitignore / .vscode/
├── src/kdlib/
│   ├── paths.py / data.py / models.py / training.py / metrics.py
│   ├── features.py                   # ★新增:中间层特征提取(hook)
│   ├── losses/                       # ★新增:每个 KD 方法一个模块
│   │   ├── __init__.py               # REGISTRY 注册表
│   │   ├── hinton.py                 # 现有 distillation_loss 搬过来
│   │   ├── fitnet.py / attention.py / pkt.py / rkd.py / dkd.py
│   └── cli.py                        # 统一 runner(替代散落脚本)
├── experiments/                      # 可选薄脚本,或全部走 CLI
└── outputs/
    ├── checkpoints/teacher.pt        # ★共享冻结资产,所有方法复用
    ├── checkpoints/student_init.pt
    ├── runs.jsonl                    # ★每次实验 append 一行
    └── figures/



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
