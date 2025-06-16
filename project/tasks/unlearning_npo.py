from copy import deepcopy
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


class UnlearningNPO:
    def __init__(
        self,
        global_config: DictConfig,
        target_id: str,
        target_name: str,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        logger: Logger,
        beta: float,
        **kwargs,
    ):
        self.global_config = global_config
        self.task_config = global_config.task
        self.target_id = target_id
        self.target_name = target_name
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.logger = logger
        self.beta = beta

    def unlearn(self):
        log.info("Task: unlearning_npo")
        task_config = self.global_config.task

        log.info("Creating reference model (model_ref)")
        model_ref = deepcopy(self.pre_trained_llm)
        model_ref.eval()
        for param in model_ref.parameters():
            param.requires_grad = False

        log.info("Loading data")
        unlearning_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            target_ids=[self.target_id],
        )
        unlearning_datamodule.prepare_data()
        unlearning_datamodule.setup("train")

        log.info("Instantiating UnlearningNPOTrainingModule")
        task_module = UnlearningNPOTrainingModule(
            model_theta=self.pre_trained_llm,
            model_ref=model_ref,
            pre_trained_llm_tokenizer=self.pre_trained_llm_tokenizer,
            beta=self.beta,
            lr=task_config.training_module.lr,
            weight_decay=task_config.training_module.weight_decay,
            **task_config.training_module,
        )

        log_hyperparameters(self.logger, self.global_config, [("model_theta", self.pre_trained_llm)])

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
            current_max_steps = trainer.max_steps if trainer.max_steps is not None and trainer.max_steps > 0 else 0
            if idx == 0 :
                 new_max_steps = stage["steps"]
            else:
                 new_max_steps = current_max_steps + stage["steps"]

            log.info(f"Setting max steps to {new_max_steps}")
            trainer.fit_loop.epoch_loop.max_steps = new_max_steps
            trainer.fit(task_module, datamodule=unlearning_datamodule)
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


class UnlearningNPOTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model_theta: AutoModelForCausalLM,
        model_ref: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        lr: float,
        weight_decay: float,
        beta: float,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model_theta", "model_ref", "pre_trained_llm_tokenizer"))
        self.model_theta = model_theta
        self.model_ref = model_ref
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.automatic_optimization = True

        for param in self.model_theta.parameters():
            param.requires_grad = True
        
        self.model_ref.eval()
        for param in self.model_ref.parameters():
            param.requires_grad = False


    def training_step(self, batch, batch_idx):
        input_ids = batch["primary_input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["primary_labels"]

        batch_size = input_ids.shape[0]

        outputs_theta = self.model_theta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        log_probs_theta = -outputs_theta.loss

        self.model_ref.to(input_ids.device)
        with torch.no_grad():
            outputs_ref = self.model_ref(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            log_probs_ref = -outputs_ref.loss
        
        npo_loss_term = -self.hparams.beta * (log_probs_theta - log_probs_ref)
        loss = -(2 / self.hparams.beta) * torch.nn.functional.logsigmoid(npo_loss_term)
        
        self.log("train/loss", loss, batch_size=batch_size, prog_bar=True)
        self.log("train/log_probs_theta", log_probs_theta, batch_size=batch_size)
        self.log("train/log_probs_ref", log_probs_ref, batch_size=batch_size)

        return {"loss": loss}

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.model_theta.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            ),
        ]
