from pathlib import Path

import lightning.pytorch as pl
import torch
from project.utils.logging import get_logger
import json
from torch.utils.data import DataLoader, Dataset, IterableDataset, ConcatDataset
from transformers import AutoTokenizer
from typing import Dict, List
from itertools import cycle



log = get_logger()

data_root = Path(__file__).parent.parent / "data/rwku/benchmark/Target"


class RWKUPositiveDataset(Dataset):
    def __init__(
        self,
        target_id: str,
        max_input_length: int,
        context_window_length: int,
        primary_tokenizer: AutoTokenizer,
        secondary_tokenizer: AutoTokenizer | None = None,
    ):
        primary_tokenizer.pad_token = primary_tokenizer.eos_token
        if secondary_tokenizer is not None:
            secondary_tokenizer.pad_token = secondary_tokenizer.eos_token
        self.primary_tokenizer = primary_tokenizer
        self.secondary_tokenizer = secondary_tokenizer
        self.max_input_length = max_input_length
        self.context_window_length = context_window_length

        self.texts = []
        with open(data_root / target_id / "positive_phi.json", "r") as f:
            entries = json.load(f)
            for entry in entries:
                self.texts.append(entry["text"])

        assert len(self.texts) > 0, f"No positive texts found for target {target_id}"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._process_text(self.texts[index])

    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        self.primary_tokenizer_output = self.primary_tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=self.max_input_length + 1,
            padding="max_length",
            padding_side="right",
            truncation=True,
        )
        primary_tokens = self.primary_tokenizer_output["input_ids"]
        primary_offsets = self.primary_tokenizer_output["offset_mapping"]

        primary_input_ids = primary_tokens[: self.max_input_length]
        primary_labels = primary_tokens[1 : self.max_input_length + 1]

        if self.secondary_tokenizer is None:
            return {
                "primary_input_ids": torch.tensor(primary_input_ids),
                "primary_labels": torch.tensor(primary_labels),
                "attention_mask": torch.tensor(
                    primary_input_ids != self.primary_tokenizer.pad_token_id
                ),
            }

        self.secondary_tokenizer_output = self.secondary_tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding_side="right",
        )
        secondary_tokens = self.secondary_tokenizer_output["input_ids"]
        secondary_offsets = self.secondary_tokenizer_output["offset_mapping"]

        secondary_context_windows = []

        for ptoken, (pstart, pend) in zip(
            primary_tokens[: self.max_input_length],
            primary_offsets[: self.max_input_length],
        ):
            if ptoken == self.primary_tokenizer.pad_token_id:
                secondary_context_windows.append(
                    torch.tensor(
                        [self.secondary_tokenizer.pad_token_id]
                        * self.context_window_length
                    )
                )
                continue

            context_window_for_token = []

            for stoken, (sstart, send) in zip(secondary_tokens, secondary_offsets):
                if stoken == self.secondary_tokenizer.pad_token_id:
                    break
                if send > pend:
                    context_window_for_token.append(stoken)
                if len(context_window_for_token) == self.context_window_length:
                    break

            context_window_for_token = context_window_for_token + [
                self.secondary_tokenizer.pad_token_id
            ] * (self.context_window_length - len(context_window_for_token))

            secondary_context_windows.append(torch.tensor(context_window_for_token))

        has_full_window = [
            (window != self.secondary_tokenizer.pad_token_id).all().item()
            for window in secondary_context_windows
        ]

        return {
            "primary_input_ids": torch.tensor(primary_input_ids),
            "primary_labels": torch.tensor(primary_labels),
            "secondary_context_windows": torch.stack(secondary_context_windows),
            "attention_mask": torch.tensor(
                primary_input_ids != self.primary_tokenizer.pad_token_id
            ),
            "has_full_window": torch.tensor(has_full_window),
        }

class RWKUPositiveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        target_ids: List[str],
        batch_size: int,
        num_workers: int,
        max_input_length: int,
        context_window_length: int,
        primary_tokenizer: AutoTokenizer,
        secondary_tokenizer: AutoTokenizer | None = None,
        **kwargs,
    ):
        super().__init__()
        self.target_ids = target_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.primary_tokenizer = primary_tokenizer
        self.secondary_tokenizer = secondary_tokenizer
        self.max_input_length = max_input_length
        self.context_window_length = context_window_length
        self.datasets = [
            RWKUPositiveDataset(
                target_id=target_id,
                primary_tokenizer=self.primary_tokenizer,
                secondary_tokenizer=self.secondary_tokenizer,
                max_input_length=self.max_input_length,
                context_window_length=self.context_window_length,
            )
            for target_id in self.target_ids
        ]
        self.merged_dataset = ConcatDataset(self.datasets)

    def _infinite_loader(self, loader: DataLoader):
        while True:
            for batch in iter(loader):
                yield batch

    def train_dataloader(self):
        return self._infinite_loader(
            DataLoader(
                self.merged_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
        )


def main():
    primary_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct"
    )
    secondary_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )
    dataset = RWKUPositiveDataset(
        "1_Stephen_King", primary_tokenizer, secondary_tokenizer, 512, 3
    )

    def print_idx(item_idx: int, token_idx: int):
        item = dataset[item_idx]
        print("-" * 10 + f"Index: ({item_idx}, {token_idx})" + "-" * 10)
        print(primary_tokenizer.decode(item["primary_input_ids"][token_idx]))
        print(primary_tokenizer.decode(item["primary_labels"][token_idx]))
        print(secondary_tokenizer.decode(item["secondary_context_windows"][token_idx]))

    print_idx(0, 0)
    print_idx(0, 1)
    print_idx(0, 2)
    print_idx(0, 3)
    print_idx(0, 4)
    print_idx(0, 5)
    print_idx(0, len(dataset[0]["primary_input_ids"]) - 2)
    print_idx(0, len(dataset[0]["primary_input_ids"]) - 1)


if __name__ == "__main__":
    main()
