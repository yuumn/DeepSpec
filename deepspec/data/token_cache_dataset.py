"""Disk-backed cache for tokenized training samples only.

The cache deliberately contains no target-model activations.  Each record stores
``input_ids``, ``attention_mask`` and ``loss_mask``; target hidden states are
computed on the training GPU immediately before the draft-model forward pass.
"""

import hashlib
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


TOKEN_CACHE_VERSION = 1
TOKEN_CACHE_DATASET_STATE_VERSION = 1
TOKEN_CACHE_INDEX_RECORD_STRUCT = struct.Struct("<QIIQQQ")
TOKEN_CACHE_INDEX_RECORD_SIZE = TOKEN_CACHE_INDEX_RECORD_STRUCT.size
TOKEN_CACHE_TOKEN_DTYPE = "int32"
TOKEN_CACHE_MASK_DTYPE = "uint8"


def atomic_json_dump(payload, path: str):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _tensor_to_bytes(tensor: torch.Tensor, dtype: torch.dtype) -> bytes:
    cpu_tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    return cpu_tensor.numpy().tobytes()


def expected_token_cache_tensor_nbytes(seq_len: int):
    seq_len = int(seq_len)
    return {
        "input_ids": seq_len * 4,
        "attention_mask": seq_len,
        "loss_mask": seq_len,
    }


def pack_token_cache_index_record(
    *,
    sample_id: int,
    shard_id: int,
    seq_len: int,
    input_ids_offset: int,
    attention_mask_offset: int,
    loss_mask_offset: int,
):
    return TOKEN_CACHE_INDEX_RECORD_STRUCT.pack(
        int(sample_id),
        int(shard_id),
        int(seq_len),
        int(input_ids_offset),
        int(attention_mask_offset),
        int(loss_mask_offset),
    )


def unpack_token_cache_index_record(buffer, offset: int = 0):
    (
        sample_id,
        shard_id,
        seq_len,
        input_ids_offset,
        attention_mask_offset,
        loss_mask_offset,
    ) = TOKEN_CACHE_INDEX_RECORD_STRUCT.unpack_from(buffer, offset)
    return {
        "sample_id": sample_id,
        "shard_id": shard_id,
        "seq_len": seq_len,
        "input_ids_offset": input_ids_offset,
        "attention_mask_offset": attention_mask_offset,
        "loss_mask_offset": loss_mask_offset,
    }


def compute_local_sample_range(*, num_samples: int, rank: int, world_size: int):
    base = int(num_samples) // int(world_size)
    remainder = int(num_samples) % int(world_size)
    start = int(rank) * base + min(int(rank), remainder)
    local_count = base + (1 if int(rank) < remainder else 0)
    return start, start + local_count


def prepare_token_cache_output_dir(output_dir: str):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            f"Token cache output dir is not empty: {output_dir}. "
            "Use a new output directory."
        )
    os.makedirs(os.path.join(output_dir, "_tmp"), exist_ok=True)


@dataclass
class LocalTokenCacheWriteSummary:
    global_rank: int
    source_sample_start: int
    source_sample_end: int
    num_local_samples: int
    local_shards: list[dict]

    def to_json(self):
        return {
            "global_rank": int(self.global_rank),
            "source_sample_start": int(self.source_sample_start),
            "source_sample_end": int(self.source_sample_end),
            "num_local_samples": int(self.num_local_samples),
            "local_shards": list(self.local_shards),
        }


@dataclass(frozen=True)
class TokenCacheSampleBytes:
    sample_id: int
    seq_len: int
    input_ids: bytes
    attention_mask: bytes
    loss_mask: bytes


def build_token_cache_sample_bytes(
    *,
    sample_id: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
):
    seq_len = int(input_ids.shape[0])
    assert attention_mask.shape == input_ids.shape
    assert loss_mask.shape == input_ids.shape
    return TokenCacheSampleBytes(
        sample_id=int(sample_id),
        seq_len=seq_len,
        input_ids=_tensor_to_bytes(input_ids, torch.int32),
        attention_mask=_tensor_to_bytes(attention_mask, torch.uint8),
        loss_mask=_tensor_to_bytes(loss_mask, torch.uint8),
    )


class LocalTokenCacheWriter:
    def __init__(self, *, rank_dir: str, max_shard_bytes: int):
        self.rank_dir = rank_dir
        self.max_shard_bytes = int(max_shard_bytes)
        self.index_handle = open(os.path.join(rank_dir, "samples.local.idx"), "wb")
        self.current_shard_id = -1
        self.current_shard_handle = None
        self.current_shard_name = None
        self.current_shard_size = 0
        self.current_shard_hasher = None
        self.local_shards = []
        self.num_local_samples = 0

    def _close_current_shard(self):
        if self.current_shard_handle is None:
            return
        self.current_shard_handle.flush()
        os.fsync(self.current_shard_handle.fileno())
        self.current_shard_handle.close()
        self.local_shards.append(
            {
                "file_name": self.current_shard_name,
                "nbytes": int(self.current_shard_size),
                "sha256": self.current_shard_hasher.hexdigest(),
            }
        )
        self.current_shard_handle = None
        self.current_shard_name = None
        self.current_shard_hasher = None

    def close(self):
        self._close_current_shard()
        if getattr(self, "index_handle", None) is not None:
            self.index_handle.flush()
            os.fsync(self.index_handle.fileno())
            self.index_handle.close()
            self.index_handle = None

    def _open_new_shard(self):
        self._close_current_shard()
        self.current_shard_id += 1
        self.current_shard_name = f"shard-local-{self.current_shard_id:05d}.bin"
        self.current_shard_handle = open(
            os.path.join(self.rank_dir, self.current_shard_name), "wb"
        )
        self.current_shard_size = 0
        self.current_shard_hasher = hashlib.sha256()

    def _ensure_shard(self, sample_nbytes: int):
        if self.current_shard_handle is None:
            self._open_new_shard()
        elif (
            self.current_shard_size > 0
            and self.current_shard_size + int(sample_nbytes) > self.max_shard_bytes
        ):
            self._open_new_shard()

    def _write_payload(self, payload: bytes) -> int:
        offset = self.current_shard_size
        self.current_shard_handle.write(payload)
        self.current_shard_hasher.update(payload)
        self.current_shard_size += len(payload)
        return offset

    def write_sample_bytes(self, sample: TokenCacheSampleBytes):
        sample_nbytes = (
            len(sample.input_ids) + len(sample.attention_mask) + len(sample.loss_mask)
        )
        self._ensure_shard(sample_nbytes)
        input_ids_offset = self._write_payload(sample.input_ids)
        attention_mask_offset = self._write_payload(sample.attention_mask)
        loss_mask_offset = self._write_payload(sample.loss_mask)
        self.index_handle.write(
            pack_token_cache_index_record(
                sample_id=sample.sample_id,
                shard_id=self.current_shard_id,
                seq_len=sample.seq_len,
                input_ids_offset=input_ids_offset,
                attention_mask_offset=attention_mask_offset,
                loss_mask_offset=loss_mask_offset,
            )
        )
        self.num_local_samples += 1


class AsyncTokenCacheWriter:
    def __init__(
        self,
        *,
        rank_dir: str,
        max_shard_bytes: int,
        max_queue_size: int = 128,
    ):
        self.writer = LocalTokenCacheWriter(
            rank_dir=rank_dir,
            max_shard_bytes=max_shard_bytes,
        )
        self.queue = queue.Queue(maxsize=int(max_queue_size))
        self.sentinel = object()
        self.num_local_samples = 0
        self._closed = False
        self._exception = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"token-cache-writer-{os.path.basename(rank_dir)}",
        )
        self.thread.start()

    @property
    def local_shards(self):
        return self.writer.local_shards

    def _run(self):
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self.sentinel:
                        break
                    self.writer.write_sample_bytes(item)
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self._exception = exc
        finally:
            try:
                self.writer.close()
            except BaseException as exc:
                if self._exception is None:
                    self._exception = exc

    def _raise_if_failed(self):
        if self._exception is not None:
            raise RuntimeError("Async token cache writer failed.") from self._exception

    def _put(self, item):
        while True:
            self._raise_if_failed()
            try:
                self.queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    def write_sample(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        sample = build_token_cache_sample_bytes(
            sample_id=self.num_local_samples,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
        self._put(sample)
        self.num_local_samples += 1

    def close(self):
        if self._closed:
            self._raise_if_failed()
            return
        if self._exception is None:
            self._put(self.sentinel)
        self.thread.join()
        self._closed = True
        self._raise_if_failed()
        assert self.writer.num_local_samples == self.num_local_samples, (
            "Async token cache writer lost samples: "
            f"{self.writer.num_local_samples} != {self.num_local_samples}"
        )


def load_local_token_cache_summary(rank_dir: str):
    with open(os.path.join(rank_dir, "summary.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_global_token_cache_shard_map(summaries):
    shard_map = {}
    shards = []
    next_shard_id = 0
    for summary in sorted(summaries, key=lambda item: int(item["source_sample_start"])):
        local_map = []
        for local_shard in summary["local_shards"]:
            local_map.append(next_shard_id)
            shards.append(
                {
                    "shard_id": next_shard_id,
                    "file_name": f"shard-{next_shard_id:05d}.bin",
                    "nbytes": int(local_shard["nbytes"]),
                    "sha256": str(local_shard["sha256"]),
                }
            )
            next_shard_id += 1
        shard_map[int(summary["global_rank"])] = local_map
    return shard_map, shards


def rename_local_token_cache_shards(*, output_dir: str, rank_dir: str, summary, shard_map):
    local_map = shard_map[int(summary["global_rank"])]
    for local_shard_id, local_shard in enumerate(summary["local_shards"]):
        source = os.path.join(rank_dir, local_shard["file_name"])
        target = os.path.join(
            output_dir,
            f"shard-{local_map[local_shard_id]:05d}.bin",
        )
        os.replace(source, target)


def finalize_token_cache_index(*, output_dir: str, summaries, shard_map):
    index_tmp_path = os.path.join(output_dir, "samples.idx.tmp")
    hasher = hashlib.sha256()
    next_expected_sample_id = 0
    with open(index_tmp_path, "wb") as output_handle:
        for summary in sorted(
            summaries,
            key=lambda item: int(item["source_sample_start"]),
        ):
            rank_dir = os.path.join(
                output_dir,
                "_tmp",
                f"rank_{int(summary['global_rank'])}",
            )
            local_index_path = os.path.join(rank_dir, "samples.local.idx")
            with open(local_index_path, "rb") as local_handle:
                local_bytes = local_handle.read()
            assert len(local_bytes) % TOKEN_CACHE_INDEX_RECORD_SIZE == 0
            next_local_sample_id = 0
            for offset in range(0, len(local_bytes), TOKEN_CACHE_INDEX_RECORD_SIZE):
                record = unpack_token_cache_index_record(local_bytes, offset)
                assert int(record["sample_id"]) == next_local_sample_id
                record["sample_id"] = next_expected_sample_id
                record["shard_id"] = shard_map[int(summary["global_rank"])][
                    int(record["shard_id"])
                ]
                packed = pack_token_cache_index_record(**record)
                output_handle.write(packed)
                hasher.update(packed)
                next_local_sample_id += 1
                next_expected_sample_id += 1
        output_handle.flush()
        os.fsync(output_handle.fileno())
    index_path = os.path.join(output_dir, "samples.idx")
    os.replace(index_tmp_path, index_path)
    return {
        "num_samples": next_expected_sample_id,
        "nbytes": os.path.getsize(index_path),
        "sha256": hasher.hexdigest(),
    }


def build_token_cache_manifest(
    *,
    num_samples: int,
    num_source_samples: int,
    shards,
    index_metadata,
    extra_fields=None,
):
    manifest = {
        "version": TOKEN_CACHE_VERSION,
        "num_samples": int(num_samples),
        "num_source_samples": int(num_source_samples),
        "num_shards": len(shards),
        "token_dtype": TOKEN_CACHE_TOKEN_DTYPE,
        "mask_dtype": TOKEN_CACHE_MASK_DTYPE,
        "index_record_size": TOKEN_CACHE_INDEX_RECORD_SIZE,
        "index": dict(index_metadata),
        "shards": list(shards),
    }
    if extra_fields:
        manifest.update(extra_fields)
    return manifest


def write_token_cache_manifest(*, output_dir: str, manifest):
    atomic_json_dump(manifest, os.path.join(output_dir, "manifest.json"))


def cleanup_token_cache_tmp_dir(output_dir: str):
    tmp_dir = os.path.join(output_dir, "_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


def load_token_cache_manifest(cache_dir: str):
    manifest_path = os.path.join(cache_dir, "manifest.json")
    assert os.path.exists(manifest_path), f"Missing token cache manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_token_cache_manifest(cache_dir=cache_dir, manifest=manifest)
    return manifest


def validate_token_cache_manifest(*, cache_dir: str, manifest):
    required_fields = {
        "version",
        "num_samples",
        "num_source_samples",
        "num_shards",
        "token_dtype",
        "mask_dtype",
        "index_record_size",
        "index",
        "shards",
        "cached_fields",
    }
    missing = sorted(required_fields - set(manifest))
    assert not missing, f"Token cache manifest is missing fields: {missing}"
    assert int(manifest["version"]) == TOKEN_CACHE_VERSION
    assert manifest["token_dtype"] == TOKEN_CACHE_TOKEN_DTYPE
    assert manifest["mask_dtype"] == TOKEN_CACHE_MASK_DTYPE
    assert int(manifest["index_record_size"]) == TOKEN_CACHE_INDEX_RECORD_SIZE
    assert int(manifest["num_samples"]) > 0, "Token cache contains no valid samples."
    assert int(manifest["num_source_samples"]) >= int(manifest["num_samples"])
    assert manifest["cached_fields"] == [
        "input_ids",
        "attention_mask",
        "loss_mask",
    ], f"Unsupported token cache fields: {manifest['cached_fields']}"

    shards = manifest["shards"]
    assert int(manifest["num_shards"]) == len(shards)
    for expected_shard_id, shard in enumerate(shards):
        assert int(shard["shard_id"]) == expected_shard_id
        assert "nbytes" in shard and "sha256" in shard
        shard_path = os.path.join(cache_dir, shard["file_name"])
        assert os.path.exists(shard_path), f"Missing token cache shard: {shard_path}"
        assert os.path.getsize(shard_path) == int(shard["nbytes"]), (
            f"Token cache shard size changed: {shard_path}"
        )

    index_path = os.path.join(cache_dir, "samples.idx")
    assert os.path.exists(index_path), f"Missing token cache index: {index_path}"
    assert "nbytes" in manifest["index"] and "sha256" in manifest["index"]
    expected_index_size = int(manifest["num_samples"]) * TOKEN_CACHE_INDEX_RECORD_SIZE
    assert os.path.getsize(index_path) == expected_index_size
    assert int(manifest["index"]["nbytes"]) == expected_index_size


def tokenizer_signature(tokenizer):
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": str(tokenizer.name_or_path),
        "vocab_size": int(vocab_size),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def validate_train_token_cache(
    *,
    train_dataset,
    tokenizer,
    target_model_name_or_path,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int,
):
    manifest = train_dataset.manifest
    expected = {
        "target_model_name_or_path": str(target_model_name_or_path),
        "chat_template": str(chat_template),
        "max_length": int(max_length),
        "min_loss_tokens": int(min_loss_tokens),
        "tokenizer": tokenizer_signature(tokenizer),
    }
    actual = {key: manifest.get(key) for key in expected}
    assert actual == expected, (
        "Token cache preprocessing configuration does not match training.\n"
        f"cache={actual}\ntraining={expected}"
    )


class TokenCacheDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: str,
        *,
        checkpoint_state=None,
        max_open_shards: int = 4,
    ):
        super().__init__()
        assert cache_dir, "data.token_cache_path must be set."
        self.cache_dir = os.path.abspath(cache_dir)
        self.manifest = load_token_cache_manifest(self.cache_dir)
        self.num_samples = int(self.manifest["num_samples"])
        self.index_path = os.path.join(self.cache_dir, "samples.idx")
        self.index_file = None
        self.index_mmap = None
        self.max_open_shards = int(max_open_shards)
        self.shard_handles = OrderedDict()
        self.shard_mmaps = OrderedDict()
        self.shard_paths = {
            int(shard["shard_id"]): os.path.join(
                self.cache_dir,
                shard["file_name"],
            )
            for shard in self.manifest["shards"]
        }
        self.signature = self._build_signature()
        if checkpoint_state is not None:
            assert int(checkpoint_state.get("version", -1)) == (
                TOKEN_CACHE_DATASET_STATE_VERSION
            ), "Unsupported token-cache dataset checkpoint state."
            assert checkpoint_state.get("signature") == self.signature, (
                "Token cache changed since the checkpoint was saved.\n"
                f"saved={checkpoint_state.get('signature')}\n"
                f"current={self.signature}"
            )

    def _build_signature(self):
        current_manifest = load_token_cache_manifest(self.cache_dir)
        return {
            "dataset_type": "token_cache",
            "version": TOKEN_CACHE_DATASET_STATE_VERSION,
            "manifest_sha256": _canonical_json_sha256(current_manifest),
            "num_samples": int(current_manifest["num_samples"]),
            "index_sha256": str(current_manifest["index"]["sha256"]),
            "shards": [
                {
                    "sha256": str(shard["sha256"]),
                    "nbytes": int(shard["nbytes"]),
                }
                for shard in current_manifest["shards"]
            ],
        }

    def state_dict(self):
        current_signature = self._build_signature()
        assert current_signature == self.signature, (
            "Token cache changed during training; refusing to save an "
            "inconsistent checkpoint."
        )
        return {
            "version": TOKEN_CACHE_DATASET_STATE_VERSION,
            "signature": self.signature,
            "cache_dir": self.cache_dir,
            "num_samples": self.num_samples,
            "cached_fields": ["input_ids", "attention_mask", "loss_mask"],
        }

    def __len__(self):
        return self.num_samples

    def close(self):
        for shard_mmap in getattr(self, "shard_mmaps", {}).values():
            shard_mmap.close()
        for handle in getattr(self, "shard_handles", {}).values():
            handle.close()
        if hasattr(self, "shard_mmaps"):
            self.shard_mmaps.clear()
        if hasattr(self, "shard_handles"):
            self.shard_handles.clear()
        if getattr(self, "index_mmap", None) is not None:
            self.index_mmap.close()
            self.index_mmap = None
        if getattr(self, "index_file", None) is not None:
            self.index_file.close()
            self.index_file = None

    def __del__(self):  # pragma: no cover
        self.close()

    def __getstate__(self):  # pragma: no cover
        state = dict(self.__dict__)
        state["index_file"] = None
        state["index_mmap"] = None
        state["shard_handles"] = OrderedDict()
        state["shard_mmaps"] = OrderedDict()
        return state

    def _ensure_index_mmap(self):
        if self.index_mmap is None:
            self.index_file = open(self.index_path, "rb")
            self.index_mmap = mmap.mmap(
                self.index_file.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )

    def _get_shard_mmap(self, shard_id: int):
        shard_id = int(shard_id)
        if shard_id in self.shard_mmaps:
            self.shard_mmaps.move_to_end(shard_id)
            self.shard_handles.move_to_end(shard_id)
            return self.shard_mmaps[shard_id]
        handle = open(self.shard_paths[shard_id], "rb")
        self.shard_handles[shard_id] = handle
        self.shard_mmaps[shard_id] = mmap.mmap(
            handle.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )
        while len(self.shard_mmaps) > self.max_open_shards:
            evicted_id, evicted_mmap = self.shard_mmaps.popitem(last=False)
            evicted_mmap.close()
            self.shard_handles.pop(evicted_id).close()
        return self.shard_mmaps[shard_id]

    def _read_tensor(
        self,
        *,
        shard_mmap,
        offset: int,
        seq_len: int,
        np_dtype,
        torch_dtype,
        nbytes: int,
    ):
        assert int(offset) + int(nbytes) <= shard_mmap.size()
        array = np.frombuffer(
            shard_mmap,
            dtype=np_dtype,
            count=int(seq_len),
            offset=int(offset),
        ).copy()
        tensor = torch.from_numpy(array)
        if tensor.dtype != torch_dtype:
            tensor = tensor.to(dtype=torch_dtype)
        return tensor

    def __getitem__(self, index: int):
        if not (0 <= int(index) < self.num_samples):
            raise IndexError(index)
        self._ensure_index_mmap()
        record = unpack_token_cache_index_record(
            self.index_mmap,
            int(index) * TOKEN_CACHE_INDEX_RECORD_SIZE,
        )
        assert int(record["sample_id"]) == int(index)
        seq_len = int(record["seq_len"])
        assert seq_len > 0
        shard_mmap = self._get_shard_mmap(int(record["shard_id"]))
        nbytes = expected_token_cache_tensor_nbytes(seq_len)
        return {
            "input_ids": self._read_tensor(
                shard_mmap=shard_mmap,
                offset=record["input_ids_offset"],
                seq_len=seq_len,
                np_dtype=np.int32,
                torch_dtype=torch.int32,
                nbytes=nbytes["input_ids"],
            ),
            "attention_mask": self._read_tensor(
                shard_mmap=shard_mmap,
                offset=record["attention_mask_offset"],
                seq_len=seq_len,
                np_dtype=np.uint8,
                torch_dtype=torch.uint8,
                nbytes=nbytes["attention_mask"],
            ),
            "loss_mask": self._read_tensor(
                shard_mmap=shard_mmap,
                offset=record["loss_mask_offset"],
                seq_len=seq_len,
                np_dtype=np.uint8,
                torch_dtype=torch.uint8,
                nbytes=nbytes["loss_mask"],
            ),
        }


def _pad_1d_batch(features: List[Dict], key: str):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    out = torch.zeros(
        (batch_size, max_length),
        dtype=features[0][key].dtype,
    )
    for sample_index, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[sample_index, :seq_len] = item[key]
    return out


class TokenCacheCollator:
    def __call__(self, features: List[Dict]):
        return {
            key: _pad_1d_batch(features, key)
            for key in ("input_ids", "attention_mask", "loss_mask")
        }
