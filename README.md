# Align-then-Unlearn: Embedding Alignment for LLM Unlearning

**Paper:** [TODO]()

**Abstract:**
As large language models (LLMs) are trained on
massive datasets, they have raised significant privacy and ethical concerns due to their potential
to inadvertently retain sensitive information. Unlearning seeks to selectively remove specific data
from trained models, such as personal information
or copyrighted content. Current approaches targeting specific output sequences at the token level
but often fail to achieve complete forgetting and
remain susceptible to prompt rephrasing. We propose Align-then-Unlearn, a novel framework that
performs unlearning in the semantic embedding
space rather than directly on output tokens. Alignthen-Unlearn first augments the LLM with an embedding prediction module trained to anticipate
future context representations. Unlearning is then
achieved by fine-tuning the model to minimize the
similarity between these predicted embeddings
and a target embedding that represents the concept to be removed. Initial results show that Alignthen-Unlearn effectively removes targeted knowledge with minimal degradation in overall model
utility. These findings suggest that embeddingbased unlearning offers a promising and robust
approach to removing conceptual knowledge.

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
TODO
```