"""Training-side collator for realtime Transformers target inference."""

import json
import uuid
from urllib import error, request

import torch

from deepspec.data.mooncake_transport import MooncakeTensorStore
from deepspec.data.target_cache_dataset import ConversationCollator


class RealtimeCollator:
    def __init__(
        self,
        *,
        tokenizer,
        chat_template,
        max_length,
        min_loss_tokens,
        target_server_url,
        target_model_name_or_path,
        target_layer_ids,
        target_hidden_size,
        request_timeout_s=600,
    ):
        self.conversation_collator = ConversationCollator(
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=max_length,
            min_loss_tokens=min_loss_tokens,
        )
        self.generate_url = f"{str(target_server_url).rstrip('/')}/generate"
        self.target_model_name_or_path = str(target_model_name_or_path)
        self.target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
        self.target_hidden_size = int(target_hidden_size)
        self.request_timeout_s = float(request_timeout_s)
        self.store = MooncakeTensorStore(writer=False)

    def _request_hidden_states(self, batch_id: str, batch: dict) -> dict:
        payload = json.dumps(
            {
                "batch_id": batch_id,
                "input_ids": batch["input_ids"].tolist(),
                "attention_mask": batch["attention_mask"].tolist(),
            }
        ).encode("utf-8")
        http_request = request.Request(
            self.generate_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(
                http_request, timeout=self.request_timeout_s
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"target server returned HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"cannot reach target server {self.generate_url}: {exc}"
            ) from exc

    def _validate_response(self, response: dict, batch_id: str, batch: dict) -> None:
        if response.get("batch_id") != batch_id:
            raise RuntimeError("target server returned a mismatched batch_id")
        if (
            response.get("target_model_name_or_path")
            != self.target_model_name_or_path
        ):
            raise RuntimeError(
                "target server model does not match the training config"
            )
        if (
            [int(x) for x in response.get("target_layer_ids", [])]
            != self.target_layer_ids
        ):
            raise RuntimeError(
                "target server layer ids do not match the training config"
            )
        batch_size, seq_len = batch["input_ids"].shape
        expected_shapes = {
            "target_hidden_states": [
                batch_size,
                seq_len,
                len(self.target_layer_ids) * self.target_hidden_size,
            ],
            "target_last_hidden_states": [
                batch_size,
                seq_len,
                self.target_hidden_size,
            ],
        }
        features = response.get("features", {})
        for name, shape in expected_shapes.items():
            if (
                name not in features
                or list(features[name].get("shape", [])) != shape
            ):
                raise RuntimeError(
                    f"invalid {name} shape from target server: "
                    f"{features.get(name, {}).get('shape')}, expected {shape}"
                )
            if features[name].get("dtype") != "bfloat16":
                raise RuntimeError(
                    f"invalid {name} dtype from target server: "
                    f"{features[name].get('dtype')}"
                )

    def __call__(self, features):
        batch = self.conversation_collator(features)
        if batch is None or batch["input_ids"].shape[0] != len(features):
            raise RuntimeError(
                "realtime training cannot drop records inside a batch; "
                "prefilter records with fewer than min_loss_tokens"
            )
        batch_id = uuid.uuid4().hex
        response = self._request_hidden_states(batch_id, batch)
        self._validate_response(response, batch_id, batch)

        remote_features = response["features"]
        fetched = {}
        try:
            for name in ("target_hidden_states", "target_last_hidden_states"):
                fetched[name] = self.store.get_tensor(remote_features[name])
        finally:
            for name in fetched:
                self.store.remove(remote_features[name]["key"])

        padding = batch["attention_mask"].eq(0).unsqueeze(-1)
        for name, tensor in fetched.items():
            tensor.masked_fill_(padding, 0)
            batch[name] = tensor
        return batch
