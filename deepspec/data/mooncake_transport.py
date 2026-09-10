"""Minimal raw-tensor transport used by realtime target inference."""

import os
import time
import uuid

import torch


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int64": torch.int64,
}
_OBJECT_NOT_FOUND = -704


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"realtime hidden states require environment variable {name}")
    return value


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class MooncakeTensorStore:
    """Put/get CPU tensors with shape and dtype carried in small HTTP metadata."""

    def __init__(self, *, writer: bool, store_id: str = None):
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except Exception as exc:
            raise RuntimeError(
                "realtime hidden states require mooncake-transfer-engine"
            ) from exc

        self.writer = bool(writer)
        self.store_id = store_id or os.environ.get(
            "DEEPSPEC_MOONCAKE_STORE_ID", uuid.uuid4().hex[:12]
        )
        self.store = MooncakeDistributedStore()
        if self.writer:
            global_segment_size = int(
                os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(32 << 30))
            )
            local_buffer_size = int(
                os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", str(1 << 30))
            )
        else:
            global_segment_size = int(
                os.environ.get("DEEPSPEC_MOONCAKE_CLIENT_SEGMENT_SIZE", "0")
            )
            local_buffer_size = int(
                os.environ.get(
                    "DEEPSPEC_MOONCAKE_CLIENT_BUFFER_SIZE", str(256 << 20)
                )
            )
        rc = self.store.setup(
            local_hostname=os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
            metadata_server=_required_env("MOONCAKE_METADATA_SERVER"),
            master_server_addr=_required_env("MOONCAKE_MASTER_SERVER_ADDR"),
            global_segment_size=global_segment_size,
            local_buffer_size=local_buffer_size,
            protocol=os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
            rdma_devices=os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
        )
        if rc is not None and int(rc) != 0:
            raise RuntimeError(f"Mooncake setup failed with status {rc}")

        self.put_config = None
        if self.writer:
            self.put_config = ReplicateConfig()
            self.put_config.replica_num = 1
            if hasattr(self.put_config, "with_hard_pin"):
                self.put_config.with_hard_pin = True
            elif hasattr(self.put_config, "with_soft_pin"):
                self.put_config.with_soft_pin = True

    def _put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        size = _nbytes(tensor)
        try:
            self.store.register_buffer(tensor.data_ptr(), size)
        except Exception:
            pass
        try:
            rc = self.store.put_from(
                key, tensor.data_ptr(), size, self.put_config
            )
        finally:
            try:
                self.store.unregister_buffer(tensor.data_ptr())
            except Exception:
                pass
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"Mooncake put_from failed for {key}: status {rc}")

    def put_batch(self, batch_id: str, tensors) -> dict:
        if not self.writer:
            raise RuntimeError("put_batch requires a writer store")
        metadata = {}
        try:
            for name, source in tensors.items():
                tensor = source.detach().to(device="cpu").contiguous()
                key = f"{self.store_id}/{batch_id}/g1/{name}"
                self._put_tensor(key, tensor)
                metadata[name] = {
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "nbytes": _nbytes(tensor),
                }
        except Exception:
            for item in metadata.values():
                try:
                    self.remove(item["key"])
                except Exception:
                    pass
            raise
        return metadata

    def get_tensor(self, metadata: dict) -> torch.Tensor:
        dtype_name = str(metadata["dtype"])
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported Mooncake tensor dtype: {dtype_name}")
        output = torch.empty(
            tuple(int(dim) for dim in metadata["shape"]),
            dtype=_DTYPES[dtype_name],
            device="cpu",
        ).contiguous()
        size = _nbytes(output)
        if size != int(metadata["nbytes"]):
            raise ValueError(
                f"Mooncake tensor metadata size mismatch: expected {size}, "
                f"got {metadata['nbytes']}"
            )
        try:
            self.store.register_buffer(output.data_ptr(), size)
        except Exception:
            pass
        try:
            rc = self.store.get_into(metadata["key"], output.data_ptr(), size)
        finally:
            try:
                self.store.unregister_buffer(output.data_ptr())
            except Exception:
                pass
        if rc is None or int(rc) != size:
            raise RuntimeError(
                f"Mooncake get_into failed for {metadata['key']}: "
                f"got {rc}, expected {size} bytes"
            )
        return output

    def remove(self, key: str, max_attempts: int = 20) -> None:
        for attempt in range(max_attempts):
            try:
                try:
                    rc = self.store.remove(key, force=True)
                except TypeError:
                    rc = self.store.remove(key)
            except Exception:
                rc = -1
            if rc is None or int(rc) in (0, _OBJECT_NOT_FOUND):
                return
            if attempt + 1 < max_attempts:
                time.sleep(0.1)
        raise RuntimeError(f"Mooncake remove failed for {key}: status {rc}")
