from omegaconf import DictConfig
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from project.eval import eval_llm
from hydra.utils import instantiate
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from typing import Literal
from project.utils.mean_pool import mean_pooling_reference_encoder
from project.utils import get_logger, log_hyperparameters
from lightning.pytorch.loggers import Logger
from project.utils.callbacks import get_default_callbacks
from project.utils.get_data_root import get_data_root
import os
import random

log = get_logger()


class UnlearningATU:
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
        log.info("Task: unlearning_atu")

        log.info("Validating config")
        for stage in self.task_config.stages:
            assert stage["type"] in ["training", "unlearning"], f"Invalid stage: {stage['type']}"
            assert stage["steps"] > 0, f"Steps must be greater than 0: {stage['steps']}"
            if stage["type"] == "unlearning":
                assert stage["threshold"] is not None, "Threshold must be set for unlearning stage"

        log.info("Instantiating text encoder")
        text_encoder, text_encoder_tokenizer = instantiate(self.task_config.text_encoder)

        log.info("Instantiating embedding prediction model")
        log.info(
            f"Pre-trained model hidden size: {self.pre_trained_llm.config.hidden_size}"
        )
        log.info(f"Text encoder hidden size: {text_encoder.config.hidden_size}")
        embedding_prediction_model = instantiate(
            self.task_config.embedding_prediction_model,
            input_dim=self.pre_trained_llm.config.hidden_size,
            output_dim=text_encoder.config.hidden_size,
        )

        log_hyperparameters(
            self.logger,
            self.global_config,
            [
                ("pre_trained_llm", self.pre_trained_llm),
                ("text_encoder", text_encoder),
                ("embedding_prediction_model", embedding_prediction_model),
            ],
        )
        
        log.info("Loading training data")
        other_target_ids = []
        subfolders = [f for f in os.listdir(get_data_root()) if f != self.target_id]
        other_target_ids = subfolders[:self.task_config.num_other_targets]
        random.shuffle(other_target_ids)

        training_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            secondary_tokenizer=text_encoder_tokenizer,
            target_ids=[self.target_id] + other_target_ids,
        )
        training_datamodule.prepare_data()
        training_datamodule.setup("train")

        log.info("Loading unlearning data")
        unlearning_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            secondary_tokenizer=text_encoder_tokenizer,
            target_ids=[self.target_id],
        )
        unlearning_datamodule.prepare_data()
        unlearning_datamodule.setup("train")

        log.info("Instantiating UnlearningATU")
        task = UnlearningATUTrainingModule(
            embedding_prediction_model=embedding_prediction_model,
            pre_trained_llm=self.pre_trained_llm,
            pre_trained_llm_tokenizer=self.pre_trained_llm_tokenizer,
            text_encoder=text_encoder,
            text_encoder_tokenizer=text_encoder_tokenizer,
            unlearning_target=self.target_name,
            **self.task_config.training_module,
        )

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
        for idx, stage in enumerate(self.task_config.stages):
            log.info(
                f"Starting stage {idx + 1} ({stage['type']}) of {len(self.task_config.stages)}"
            )
            new_max_steps = (
                stage["steps"] if idx == 0 else stage["steps"] + trainer.max_steps
            )
            log.info(f"Setting max steps to {new_max_steps}")
            trainer.fit_loop.epoch_loop.max_steps = new_max_steps
            task.update_stage(stage["type"])
            if stage["type"] == "training":
                trainer.fit(task, datamodule=training_datamodule)
            elif stage["type"] == "unlearning":
                task.update_unlearning_threshold(stage["threshold"])
                trainer.fit(task, datamodule=unlearning_datamodule)
            else:
                raise ValueError(f"Invalid stage: {stage['type']}")
            log.info(f"Stage {idx + 1} ({stage['type']}) completed!")
            if stage["type"] == "unlearning":
                log.info("Starting testing!")
                results = eval_llm(
                    self.pre_trained_llm,
                    self.pre_trained_llm_tokenizer,
                    self.target_id,
                    device=trainer.strategy.root_device,
                    stage_number=idx + 1,
                )
                trainer.logger.log_metrics(results)
        log.info("Unlearning complete!")


class UnlearningATUTrainingModule(pl.LightningModule):
    def __init__(
        self,
        embedding_prediction_model: nn.Module,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        text_encoder: AutoModel,
        text_encoder_tokenizer: AutoTokenizer,
        unlearning_target: str,
        training_warmup_steps: int,
        training_lr: float,
        training_weight_decay: float,
        unlearning_lr: float,
        unlearning_weight_decay: float,
        pretrained_model_hook_layer: int,
        clip_grad_norm: float,
        stage: Literal["training", "unlearning"] = "training",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=("embedding_prediction_model", "pre_trained_llm", "text_encoder")
        )
        self.unlearning_similarity_threshold = None
        self.embedding_prediction_model = embedding_prediction_model
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.text_encoder = text_encoder
        self.text_encoder_tokenizer = text_encoder_tokenizer
        self.unlearning_target = unlearning_target
        self.stage = stage
        self.update_stage(stage)
        self.automatic_optimization = False

        log.info(f"Unlearning target: {unlearning_target}, computing embedding...")
        unlearning_target_tokens = text_encoder_tokenizer.encode(
            unlearning_target, add_special_tokens=True
        )
        unlearning_target_tokens = torch.tensor(
            unlearning_target_tokens, device=self.device
        ).unsqueeze(0)  # (1, seq_len)
        self.unlearning_target_embedding = mean_pooling_reference_encoder(
            text_encoder(
                input_ids=unlearning_target_tokens,
                output_hidden_states=True,
            ),
            attention_mask=torch.ones_like(unlearning_target_tokens, device=self.device),
        )[0]

    def on_fit_start(self):
        self.unlearning_target_embedding = self.unlearning_target_embedding.to(
            self.device
        )

    def _disable_grad(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def _enable_grad(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    def update_stage(self, stage: Literal["training", "unlearning"]):
        log.info(f"Updating stage to {stage}")
        self.stage = stage
        if self.stage == "training":
            self._disable_grad(self.text_encoder)
            self._disable_grad(self.pre_trained_llm)
            self._enable_grad(self.embedding_prediction_model)
            self.pre_trained_llm.eval()
            self.text_encoder.eval()
            self.embedding_prediction_model.train()
        elif self.stage == "unlearning":
            self._disable_grad(self.text_encoder)
            self._enable_grad(self.embedding_prediction_model)
            self._enable_grad(self.pre_trained_llm)
            self.pre_trained_llm.train()
            self.embedding_prediction_model.eval()
            self.text_encoder.eval()

    def train(self, mode=True):
        if mode:
            super().train()
            if self.stage == "training":
                self.text_encoder.eval()
                self.pre_trained_llm.eval()
            elif self.stage == "unlearning":
                self.text_encoder.eval()
                self.embedding_prediction_model.eval()
        else:
            super().train(False)
        return self

    def update_unlearning_threshold(self, threshold: float):
        log.info(f"Updating unlearning threshold to {threshold}")
        self.unlearning_similarity_threshold = threshold

    def training_step(self, batch, batch_idx):
        input_ids = batch["primary_input_ids"]  # shape (batch_size, max_length)
        context_windows = batch[
            "secondary_context_windows"
        ]  # shape (batch_size, max_length, context_window_length)
        has_full_window = batch["has_full_window"]  # shape (batch_size, max_length)
        attention_mask = batch["attention_mask"]  # shape (batch_size, max_length)

        batch_size, seq_len, window_len = context_windows.shape

        opt_list = self.optimizers()

        if self.stage == "training":
            # Forward pass through the reference encoder to get the target embeddings
            with torch.no_grad():
                reference_outputs = self.text_encoder(
                    input_ids=context_windows.view(-1, context_windows.size(-1)),
                    output_hidden_states=True,
                )
                attention_mask_enc = torch.ones_like(
                    context_windows.view(-1, context_windows.size(-1))
                )
                target_embeddings = mean_pooling_reference_encoder(
                    reference_outputs, attention_mask_enc
                ).view(batch_size, seq_len, -1)

            # Forward pass through the pretrained model
            with torch.no_grad():
                pretrained_outputs = self.pre_trained_llm(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    attention_mask=attention_mask,
                )
                hidden_states = pretrained_outputs.hidden_states[
                    self.hparams.pretrained_model_hook_layer
                ]

            # Forward pass through embedding prediction model
            outputs = self.embedding_prediction_model(hidden_states)
            loss = -torch.nn.functional.cosine_similarity(
                outputs, target_embeddings, dim=-1
            ) # shape (batch_size, max_length)
            loss = loss * has_full_window
            loss = loss.sum() / (has_full_window.sum() + 1e-8)
            assert not torch.isnan(loss), "Loss is NaN"

            opt_list[0].zero_grad()
            self.manual_backward(loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.embedding_prediction_model.parameters(),
                max_norm=self.hparams.clip_grad_norm,
            )
            self.log("train/grad_norm_embedding_clip", grad_norm, batch_size=batch_size)
            opt_list[0].step()

            if batch_idx == 0:
                assert not torch.all(outputs == 0), "Outputs are all zero"
                assert not torch.all(target_embeddings == 0), (
                    "Target embeddings are all zero"
                )

        elif self.stage == "unlearning":
            # Forward pass through the pretrained model
            pretrained_outputs = self.pre_trained_llm(
                input_ids=input_ids, output_hidden_states=True, attention_mask=attention_mask
            )
            hidden_states = pretrained_outputs.hidden_states[
                self.hparams.pretrained_model_hook_layer
            ]

            # Forward pass through embedding prediction model
            outputs = self.embedding_prediction_model(
                hidden_states
            )  # shape (batch_size, seq_len, emb_dim)

            # Minimize cosine similarity between outputs and unlearning target if it exceeds the threshold
            assert self.unlearning_similarity_threshold is not None, (
                "Unlearning similarity threshold must be set before unlearning stage"
            )
            loss = torch.nn.functional.relu(
                torch.nn.functional.cosine_similarity(
                    outputs,
                    self.unlearning_target_embedding.unsqueeze(0).unsqueeze(0),
                    dim=-1,
                )
                - self.unlearning_similarity_threshold
            ).mean()
            assert not torch.isnan(loss), "Loss is NaN"

            opt_list[1].zero_grad()
            self.manual_backward(loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.pre_trained_llm.parameters(), max_norm=self.hparams.clip_grad_norm
            )
            self.log("train/grad_norm_pre_clip", grad_norm, batch_size=batch_size)
            opt_list[1].step()
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

        self.log("train/loss", loss.mean(), batch_size=batch_size)
        if self.stage == "training":
            self.log("train/training_loss", loss.mean(), batch_size=batch_size)
        else:
            self.log("train/unlearning_loss", loss.mean(), batch_size=batch_size)
            self.log("train/unlearning_threshold", self.unlearning_similarity_threshold)
        return {"loss": loss}

    def configure_optimizers(self):
        return [
            torch.optim.Adam(
                self.embedding_prediction_model.parameters(),
                lr=self.hparams.training_lr,
                weight_decay=self.hparams.training_weight_decay,
            ),
            torch.optim.SGD(
                self.pre_trained_llm.parameters(),
                lr=self.hparams.unlearning_lr,
                weight_decay=self.hparams.unlearning_weight_decay,
            ),
        ]
