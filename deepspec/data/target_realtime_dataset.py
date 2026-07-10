"""Target-cache storage protocol, writers, dataset, collator, and validation."""

import json
import mmap
import os
import queue
import shutil
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from deepspec.data.parser import preprocess_record


class RealtimeDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str, max_open_shards: int = 4):
        super().__init__()
        # self.cache_dir = os.path.abspath(cache_dir)
        # self.manifest = load_target_cache_manifest(self.cache_dir)
        # self.num_samples = int(self.manifest["num_samples"])
        # self.hidden_size = int(self.manifest["hidden_size"])
        # self.target_layer_ids = [int(layer_id) for layer_id in self.manifest["target_layer_ids"]]
        # self.num_target_layers = len(self.target_layer_ids)
        # self.index_path = os.path.join(self.cache_dir, "samples.idx")
        # self.index_file = None
        # self.index_mmap = None
        # self.max_open_shards = max_open_shards
        # self.shard_handles = OrderedDict()
        # self.shard_mmaps = OrderedDict()
        # self.shard_paths = {
        #     int(shard["shard_id"]): build_target_cache_shard_path(
        #         self.cache_dir,
        #         shard["file_name"],
        #     )
        #     for shard in self.manifest["shards"]
        # }

    def __len__(self):
        return self.num_samples

    def close(self):
        pass

    def __del__(self):  # pragma: no cover
        self.close()


    def __getitem__(self, index: int):
        if not (0 <= int(index) < self.num_samples):
            raise IndexError(index)
        

        
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": target_hidden_states,
            "target_last_hidden_states": target_last_hidden_states,
        }


def _pad_1d_batch(features: List[Dict], key: str):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    dtype = features[0][key].dtype
    out = torch.zeros((batch_size, max_length), dtype=dtype)
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]
    return out


def _pad_hidden_batch(features: List[Dict], key: str):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    hidden_dim = features[0][key].shape[1]
    dtype = features[0][key].dtype
    out = torch.zeros((batch_size, max_length, hidden_dim), dtype=dtype)
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]
    return out


class ConversationCollator:
    def __init__(
        self,
        tokenizer,
        chat_template,
        max_length,
        min_loss_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = int(max_length)
        self.min_loss_tokens = int(min_loss_tokens)

    def _process_feature(self, item):
        processed = preprocess_record(
            record=item,
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            max_length=self.max_length,
        )
        if int(processed["loss_mask"].sum().item()) < self.min_loss_tokens:
            return None
        return processed

    def __call__(self, features: List[Dict]):
        features = [self._process_feature(item) for item in features]
        features = [item for item in features if item is not None]
        if not features:
            return None
        batch = {}
        for key in ("input_ids", "attention_mask", "loss_mask"):
            batch[key] = _pad_1d_batch(features, key)
        return batch


class RealtimeCollator:
    def __call__(self, features: List[Dict]):
        batch = {}
        for key in ("input_ids", "loss_mask"):
            batch[key] = _pad_1d_batch(features, key)
        attention_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        for i, item in enumerate(features):
            attention_mask[i, : item["input_ids"].shape[0]] = 1
        batch["attention_mask"] = attention_mask
        for key in ("target_hidden_states", "target_last_hidden_states"):
            batch[key] = _pad_hidden_batch(features, key)
        return batch
