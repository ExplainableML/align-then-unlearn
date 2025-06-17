<div align="center">
    
# Align-then-Unlearn: Embedding Alignment for LLM Unlearning <br/> _ICML 2025 Workshop MUGen_
[![Paper](http://img.shields.io/badge/paper-arxiv.2506.13181-B31B1B.svg)](https://arxiv.org/abs/2506.13181)
[![Paper](https://img.shields.io/badge/paper-OpenReview-8C1B13.svg)](https://openreview.net/forum?id=pyhbguXKXQ)

Philipp Spohn <sup>1</sup> &#8198; Leander Girrbach<sup>1,2</sup> &#8198; Jessica Bader<sup>1,2</sup> &#8198; Zeynep Akata<sup>1,2</sup>

<sup>1</sup>Technical University of Munich &#8198; <sup>2</sup> MCML, MDSI, Helmholtz Munich
</div>

**Paper:** [arxiv.org/abs/2506.13181](https://arxiv.org/abs/2506.13181)

**Abstract:**
As large language models (LLMs) are trained on massive datasets, they have raised significant privacy and ethical concerns due to their potential to inadvertently retain sensitive information. Unlearning seeks to selectively remove specific data from trained models, such as personal information or copyrighted content. Current approaches targeting specific output sequences at the token level often fail to achieve complete forgetting and remain susceptible to prompt rephrasing. We propose Align-then-Unlearn, a novel framework that performs unlearning in the semantic embedding space rather than directly on output tokens. Alignthen-Unlearn first augments the LLM with an embedding prediction module trained to anticipate future context representations. Unlearning is then achieved by fine-tuning the model to minimize the similarity between these predicted embeddings and a target embedding that represents the concept to be removed. Initial results show that Alignthen-Unlearn effectively removes targeted knowledge with minimal degradation in overall model utility. These findings suggest that embedding-based unlearning offers a promising and robust approach to removing conceptual knowledge.

## Setup
- Install the project with `pip install -e .`
- Run the `data/rwku/download_rwku_data.sh` script to download the necessary datasets.
- Adapt the config files to your setup (change the wandb entity in `config/train.yaml`, adapt the launcher configs in `config/hydra/launcher`)

## How to use it

```bash
# Basic training
python launch_training.py

# Launch SLURM job
python launch_training.py -m hydra/launcher=lrz-a100

# Launch multiple SLURM jobs for all targets in celebs-1 config
python launch_training.py -m hydra/launcher=lrz-a100 experiment=celebs-1

# Use GA / NPO for unlearning (WIP, NOT WELL TESTED YET!)
python launch_training.py task=unlearning_ga
python launch_training.py task=unlearning_npo
```

## Acknowledgements
- Based on template by Marten Lienen (https://github.com/martenlienen)
- Some of the code adopted from the RWKU benchmark (https://github.com/jinzhuoran/RWKU)

## Citation
```
@article{spohn2025align,
  title={Align-then-Unlearn: Embedding Alignment for LLM Unlearning},
  author={Spohn, Philipp and Girrbach, Leander and Bader, Jessica and Akata, Zeynep},
  journal={ICML 2025 Workshop on Machine Unlearning for Generative AI},
  year={2025}
}
```
