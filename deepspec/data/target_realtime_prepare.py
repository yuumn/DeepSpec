"""Single-GPU Transformers service for realtime target hidden states."""

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import threading

import torch
from transformers import AutoModel

from deepspec.data.mooncake_transport import MooncakeTensorStore
from deepspec.utils import load_config, parse_opts_to_config


os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class TargetForwardResult:
    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


def _get_target_backbone(target_model):
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model") and hasattr(
            target_model.model, "language_model"
        ):
            return target_model.model.language_model
        raise RuntimeError("Gemma4 target model must expose a text language_model")
    return getattr(target_model, "model", target_model)


def _get_target_hidden_size(target_model) -> int:
    if str(target_model.config.model_type) in ("gemma4", "gemma4_unified"):
        return int(target_model.config.text_config.hidden_size)
    return int(target_model.config.hidden_size)


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if (
        isinstance(output, (tuple, list))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]
    raise TypeError(f"unsupported target hook output type: {type(output)!r}")


def run_target_forward_with_hooks(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids,
):
    backbone = _get_target_backbone(target_model)
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    captured = {}
    handles = []

    def capture_layer(layer_id):
        def hook(_module, _inputs, output):
            captured[layer_id] = _get_hook_tensor(output).detach()

        return hook

    try:
        if -1 in target_layer_ids:
            handles.append(
                backbone.embed_tokens.register_forward_hook(capture_layer(-1))
            )
        for layer_id in target_layer_ids:
            if layer_id >= 0:
                handles.append(
                    backbone.layers[layer_id].register_forward_hook(
                        capture_layer(layer_id)
                    )
                )
        with torch.inference_mode():
            output = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            return TargetForwardResult(
                target_hidden_states=torch.cat(
                    [captured[layer_id] for layer_id in target_layer_ids], dim=-1
                ).detach(),
                target_last_hidden_states=output.last_hidden_state.detach(),
            )
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()


class TargetHiddenStateService:
    def __init__(self, *, config, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the realtime target service requires a CUDA device")
        torch.cuda.set_device(self.device)
        self.model_name = str(config.model.target_model_name_or_path)
        self.layer_ids = [
            int(layer_id) for layer_id in config.model.target_layer_ids
        ]
        self.model = AutoModel.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device=self.device).eval()
        self.hidden_size = _get_target_hidden_size(self.model)
        self.store = MooncakeTensorStore(writer=True)
        self.lock = threading.Lock()

    def metadata(self):
        return {
            "target_model_name_or_path": self.model_name,
            "target_layer_ids": self.layer_ids,
            "hidden_size": self.hidden_size,
        }

    def generate(self, payload: dict) -> dict:
        batch_id = str(payload.get("batch_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", batch_id):
            raise ValueError(
                "batch_id must contain only letters, numbers, '_' or '-'"
            )
        input_ids = torch.tensor(payload["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(payload["attention_mask"], dtype=torch.long)
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must be equal-size 2D arrays"
            )
        if input_ids.numel() == 0:
            raise ValueError("empty target batch")

        with self.lock:
            result = run_target_forward_with_hooks(
                target_model=self.model,
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                target_layer_ids=self.layer_ids,
            )
            features = self.store.put_batch(
                batch_id,
                {
                    "target_hidden_states": result.target_hidden_states,
                    "target_last_hidden_states": result.target_last_hidden_states,
                },
            )
        return {"batch_id": batch_id, "features": features, **self.metadata()}


def _handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self._write_json(404, {"error": "not found"})
                return
            self._write_json(200, {"status": "ready", **service.metadata()})

        def do_POST(self):
            if self.path != "/generate":
                self._write_json(404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 64 * 1024**2:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                self._write_json(200, service.generate(payload))
            except Exception as exc:
                self._write_json(400, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format, *args):
            print(
                f"[target-server] {self.address_string()} {format % args}",
                flush=True,
            )

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.config_data = parse_opts_to_config(args.opts, load_config(args.config))
    return args


def main():
    args = parse_args()
    service = TargetHiddenStateService(config=args.config_data, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), _handler(service))
    print(
        f"target hidden-state service ready at http://{args.host}:{args.port}; "
        f"device={args.device}, model={service.model_name}, layers={service.layer_ids}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
