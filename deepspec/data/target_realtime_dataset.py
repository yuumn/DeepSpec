"""JSONL-backed dataset and collators for on-device target feature generation."""

import os
from typing import Dict, List

import torch
import torch.distributed as dist

from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.parser import preprocess_record


ONLINE_DATASET_STATE_VERSION = 1


def normalize_data_paths(data_paths) -> List[str]:
    if isinstance(data_paths, (str, os.PathLike)):
        data_paths = [data_paths]
    assert data_paths, "data.train_data_path must contain at least one JSONL path."
    normalized = sorted(os.path.abspath(os.fspath(path)) for path in data_paths)
    missing = [path for path in normalized if not os.path.isfile(path)]
    assert not missing, f"Training JSONL files do not exist: {missing}"
    return normalized


def _pad_1d_batch(features: List[Dict], key: str):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    dtype = features[0][key].dtype
    out = torch.zeros((batch_size, max_length), dtype=dtype)
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]
    return out


def _open_jsonl_dataset_distributed(
    *, data_paths, global_rank: int, world_size: int
) -> JsonLineDataset:
    if world_size == 1:
        return JsonLineDataset(data_paths=data_paths)

    # JsonLineDataset stores only byte offsets in its small local metadata
    # index. Let one process per node build it first instead of making every
    # GPU process scan multi-GB JSONL files concurrently.
    local_world_size = max(torch.cuda.device_count(), 1)
    is_local_main = global_rank % local_world_size == 0
    dataset = JsonLineDataset(data_paths=data_paths) if is_local_main else None
    dist.barrier()
    if dataset is None:
        dataset = JsonLineDataset(data_paths=data_paths)
    return dataset


class RealtimeDataset(torch.utils.data.Dataset):
    """Read conversations from JSONL and tokenize them on demand.

    The initial distributed scan only builds the stable list of samples that
    pass ``min_loss_tokens``. It does not persist token tensors or target hidden
    states. The list is stored in checkpoints so resume does not need to scan
    the entire JSONL dataset again.
    """

    def __init__(
        self,
        *,
        data_paths,
        tokenizer,
        chat_template: str,
        max_length: int,
        min_loss_tokens: int,
        device,
        global_rank: int,
        world_size: int,
        checkpoint_state=None,
    ):
        super().__init__()
        self.data_paths = normalize_data_paths(data_paths)
        self.tokenizer = tokenizer
        self.chat_template = str(chat_template)
        self.max_length = int(max_length)
        self.min_loss_tokens = int(min_loss_tokens)
        self.device = device
        self.global_rank = int(global_rank)
        self.world_size = int(world_size)
        self.source_dataset = _open_jsonl_dataset_distributed(
            data_paths=self.data_paths,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )

        if checkpoint_state is None:
            valid_indices = self._scan_valid_indices()
        else:
            valid_indices = self._load_checkpoint_indices(checkpoint_state)

        self._valid_indices_tensor = valid_indices.to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        self.signature = self._build_signature(self._valid_indices_tensor.numel())

        if checkpoint_state is not None:
            saved_signature = checkpoint_state.get("signature")
            assert saved_signature == self.signature, (
                "Training JSONL dataset or preprocessing configuration changed "
                "since the checkpoint was saved.\n"
                f"saved={saved_signature}\ncurrent={self.signature}"
            )

        assert self._valid_indices_tensor.numel() > 0, (
            "No training samples remain after applying min_loss_tokens="
            f"{self.min_loss_tokens}."
        )
        self.source_dataset.close()
        if self.global_rank == 0:
            print(
                "Online JSONL dataset: "
                f"{len(self)}/{len(self.source_dataset)} valid samples",
                flush=True,
            )

    def _preprocess_source_index(self, source_index: int):
        processed = preprocess_record(
            record=self.source_dataset[int(source_index)],
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            max_length=self.max_length,
        )
        # Match the compact transfer dtypes used by the former target-cache
        # protocol. CUDAPrefetcher converts input_ids back to int64 on device.
        processed["input_ids"] = processed["input_ids"].to(torch.int32)
        processed["loss_mask"] = processed["loss_mask"].to(torch.uint8)
        return processed

    def _scan_valid_indices(self) -> torch.Tensor:
        local_valid_indices = []
        local_total = len(
            range(self.global_rank, len(self.source_dataset), self.world_size)
        )
        for local_position, source_index in enumerate(
            range(self.global_rank, len(self.source_dataset), self.world_size),
            start=1,
        ):
            processed = self._preprocess_source_index(source_index)
            if int(processed["loss_mask"].sum().item()) >= self.min_loss_tokens:
                local_valid_indices.append(source_index)
            if (
                self.global_rank == 0
                and (local_position % 10_000 == 0 or local_position == local_total)
            ):
                print(
                    "[online dataset scan rank 0] "
                    f"{local_position}/{local_total} local source samples",
                    flush=True,
                )

        local_indices = torch.tensor(
            local_valid_indices,
            dtype=torch.int64,
            device=self.device,
        )
        if self.world_size == 1:
            return local_indices.cpu()

        local_size = torch.tensor(
            [local_indices.numel()], dtype=torch.int64, device=self.device
        )
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(gathered_sizes, local_size)
        sizes = [int(size.item()) for size in gathered_sizes]
        padded_size = max(max(sizes), 1)
        padded_indices = torch.full(
            (padded_size,), -1, dtype=torch.int64, device=self.device
        )
        padded_indices[: local_indices.numel()] = local_indices
        gathered_indices = [
            torch.empty_like(padded_indices) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_indices, padded_indices)
        valid_indices = torch.cat(
            [indices[:size].cpu() for indices, size in zip(gathered_indices, sizes)]
        )
        valid_indices = valid_indices.sort().values
        assert valid_indices.numel() == torch.unique(valid_indices).numel(), (
            "Distributed valid-sample scan produced duplicate source indices."
        )
        return valid_indices

    def _load_checkpoint_indices(self, checkpoint_state) -> torch.Tensor:
        assert int(checkpoint_state.get("version", -1)) == ONLINE_DATASET_STATE_VERSION, (
            "Unsupported online dataset checkpoint state version: "
            f"{checkpoint_state.get('version')}"
        )
        valid_indices = checkpoint_state.get("valid_indices")
        assert isinstance(valid_indices, torch.Tensor), (
            "Checkpoint dataset state is missing valid_indices tensor."
        )
        valid_indices = valid_indices.to(device="cpu", dtype=torch.int64).contiguous()
        assert valid_indices.ndim == 1, "valid_indices must be one-dimensional."
        if valid_indices.numel() > 0:
            assert int(valid_indices[0]) >= 0
            assert int(valid_indices[-1]) < len(self.source_dataset)
            assert bool(torch.all(valid_indices[1:] > valid_indices[:-1])), (
                "Checkpoint valid_indices must be strictly increasing."
            )
        return valid_indices

    def _build_signature(self, num_valid_samples: int):
        files = []
        for path in self.data_paths:
            stat = os.stat(path)
            files.append(
                {
                    "path": path,
                    "size": int(stat.st_size),
                    "mtime_ns": int(
                        getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
                    ),
                }
            )
        tokenizer_vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if tokenizer_vocab_size is None:
            tokenizer_vocab_size = len(self.tokenizer)
        return {
            "version": ONLINE_DATASET_STATE_VERSION,
            "files": files,
            "num_source_samples": int(len(self.source_dataset)),
            "num_valid_samples": int(num_valid_samples),
            "chat_template": self.chat_template,
            "max_length": self.max_length,
            "min_loss_tokens": self.min_loss_tokens,
            "tokenizer": {
                "class": type(self.tokenizer).__name__,
                "name_or_path": str(self.tokenizer.name_or_path),
                "vocab_size": int(tokenizer_vocab_size),
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            },
        }

    def state_dict(self):
        current_signature = self._build_signature(len(self))
        assert current_signature == self.signature, (
            "Training JSONL files or preprocessing configuration changed "
            "during training; refusing to save an inconsistent checkpoint."
        )
        return {
            "version": ONLINE_DATASET_STATE_VERSION,
            "signature": self.signature,
            "valid_indices": self._valid_indices_tensor,
        }

    def __len__(self):
        return int(self._valid_indices_tensor.numel())

    def __getitem__(self, index: int):
        if not (0 <= int(index) < len(self)):
            raise IndexError(index)
        source_index = int(self._valid_indices_tensor[int(index)])
        processed = self._preprocess_source_index(source_index)
        assert int(processed["loss_mask"].sum().item()) >= self.min_loss_tokens, (
            "A checkpointed valid sample no longer passes min_loss_tokens; "
            "the JSONL data or tokenizer changed during training."
        )
        return processed

    def close(self):
        source_dataset = getattr(self, "source_dataset", None)
        if source_dataset is not None:
            source_dataset.close()

    def __del__(self):  # pragma: no cover
        self.close()


class ConversationCollator:
    """Raw-record collator retained for standalone data-preparation scripts."""

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
        return RealtimeCollator()(features)


class RealtimeCollator:
    def __call__(self, features: List[Dict]):
        return {
            key: _pad_1d_batch(features, key)
            for key in ("input_ids", "attention_mask", "loss_mask")
        }
