from omegaconf import DictConfig
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from project.eval import eval_llm
from hydra.utils import instantiate
import torch
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from project.utils import get_logger, log_hyperparameters
from lightning.pytorch.loggers import Logger
from project.utils.callbacks import get_default_callbacks

log = get_logger()


class UnlearningGA:
    def __init__(
        self,
        global_config: DictConfig,
        target_id: str,
        target_name: str,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        logger: Logger,
        **kwargs,
    ):
        self.global_config = global_config
        self.task_config = global_config.task
        self.target_id = target_id
        self.target_name = target_name
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.logger = logger

    def unlearn(self):
        log.info("Task: unlearning_ga")
        task_config = self.global_config.task

        log.info("Loading data")
        unlearning_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            target_ids=[self.target_id],
        )
        unlearning_datamodule.prepare_data()
        unlearning_datamodule.setup("train")

        log.info("Instantiating UnlearningGATrainingModule")
        task = UnlearningGATrainingModule(
            pre_trained_llm=self.pre_trained_llm,
            pre_trained_llm_tokenizer=self.pre_trained_llm_tokenizer,
            **task_config.training_module,
        )

        log_hyperparameters(self.logger, self.global_config, [("pre_trained_llm", self.pre_trained_llm)])

        log.info("Instantiating trainer")
        trainer = Trainer(
            **self.global_config.trainer,
            callbacks=get_default_callbacks(),
            logger=self.logger,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            enable_checkpointing=False,
        )

        log.info("Starting initial evaluation!")
        if not self.global_config.skip_initial_eval:
            results = eval_llm(
                self.pre_trained_llm,
                self.pre_trained_llm_tokenizer,
                self.target_id,
                trainer.strategy.root_device,
                0,
            )
            trainer.logger.log_metrics(results)
        else:
            log.info("Skipping initial evaluation!")

        log.info("Starting training!")
        for idx, stage in enumerate(task_config.stages):
            assert stage["type"] == "unlearning"
            log.info(
                f"Starting stage {idx + 1} ({stage['type']}) of {len(task_config.stages)}"
            )
            new_max_steps = (
                stage["steps"] if idx == 0 else stage["steps"] + trainer.max_steps
            )
            log.info(f"Setting max steps to {new_max_steps}")
            trainer.fit_loop.epoch_loop.max_steps = new_max_steps
            trainer.fit(task, datamodule=unlearning_datamodule)
            log.info(f"Stage {idx + 1} ({stage['type']}) completed!")
            log.info("Starting testing!")
            results = eval_llm(
                self.pre_trained_llm,
                self.pre_trained_llm_tokenizer,
                self.target_id,
                trainer.strategy.root_device,
                0,
            )
            trainer.logger.log_metrics(results)
        log.info("Unlearning complete!")


class UnlearningGATrainingModule(pl.LightningModule):
    def __init__(
        self,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        lr: float,
        weight_decay: float,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("pre_trained_llm", "pre_trained_llm_tokenizer"))
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.automatic_optimization = True

        for param in self.pre_trained_llm.parameters():
            param.requires_grad = True

    def training_step(self, batch, batch_idx):
        input_ids = batch["primary_input_ids"]  # shape (batch_size, max_length)
        attention_mask = batch["attention_mask"]  # shape (batch_size, max_length)
        labels = batch["primary_labels"]  # shape (batch_size, max_length)

        batch_size, seq_len = input_ids.shape

        outputs = self.pre_trained_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = -outputs.loss  # gradient ascent

        self.log("train/loss", loss, batch_size=batch_size)

        return {"loss": loss}

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.pre_trained_llm.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            ),
        ]
