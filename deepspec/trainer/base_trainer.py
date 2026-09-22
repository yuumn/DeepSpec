from contextlib import nullcontext
import math
import os

import torch
from torch.profiler import profile, ProfilerActivity, record_function
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from deepspec.data import (
    TokenCacheCollator,
    TokenCacheDataset,
    run_target_forward_with_hooks,
    validate_train_token_cache,
)
from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.utils import (
    BF16Optimizer,
    StatelessResumableDistributedSampler,
    ensure_dir,
    init_dist,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    load_checkpoint_dataset_state,
    load_resume_draft_model,
    load_training_state,
    save_checkpoint,
)
import deepspec.utils.training_logger as training_logger
from deepspec.utils.hfai_suspend import SuspendController


_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_SHARDING_STRATEGIES = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}

_HYBRID_STRATEGIES = (
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
)

PROFILE_STEPS = os.environ.get("PROFILE_STEPS", None)
PROFILE_STEPS = int(PROFILE_STEPS) if PROFILE_STEPS is not None else None
IS_PROFILE = PROFILE_STEPS is not None

def _build_fsdp_kwargs(
    *, sharding_strategy_name: str, precision_dtype, world_size: int
) -> dict:
    sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    fsdp_kwargs = dict(
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=precision_dtype,
            buffer_dtype=precision_dtype,
        ),
        sharding_strategy=sharding_strategy,
    )
    if sharding_strategy in _HYBRID_STRATEGIES:
        devices_per_node = torch.cuda.device_count()
        fsdp_kwargs["device_mesh"] = init_device_mesh(
            "cuda",
            (world_size // devices_per_node, devices_per_node),
            mesh_dim_names=("replicate", "shard"),
        )
    return fsdp_kwargs


def _compute_gradient_accumulation_steps(
    *, world_size: int, local_batch_size: int, global_batch_size: int
) -> int:
    denom = world_size * local_batch_size
    assert global_batch_size % denom == 0, (
        "global_batch_size must be divisible by world_size * local_batch_size: "
        f"global_batch_size={global_batch_size}, world_size={world_size}, "
        f"local_batch_size={local_batch_size}"
    )
    return global_batch_size // denom


def _compute_samples_per_epoch(*, dataset_size: int, global_batch_size: int) -> int:
    samples_per_epoch = (dataset_size // global_batch_size) * global_batch_size
    assert samples_per_epoch > 0, (
        "train dataset is too small to form one full global batch: "
        f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
    )
    return samples_per_epoch


def _compute_training_schedule(
    *,
    world_size: int,
    dataset_size: int,
    local_batch_size: int,
    global_batch_size: int,
    num_train_epochs: int,
    max_train_steps=None,
) -> tuple[int, int, int, int, int, int, int]:
    gradient_accumulation_steps = _compute_gradient_accumulation_steps(
        world_size=world_size,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
    )
    samples_per_epoch = _compute_samples_per_epoch(
        dataset_size=dataset_size,
        global_batch_size=global_batch_size,
    )
    per_rank_samples_per_epoch = samples_per_epoch // world_size
    micro_batches_per_epoch = per_rank_samples_per_epoch // local_batch_size
    steps_per_epoch = micro_batches_per_epoch // gradient_accumulation_steps
    if max_train_steps is None:
        resolved_max_train_steps = int(num_train_epochs) * steps_per_epoch
        resolved_num_train_epochs = int(num_train_epochs)
    else:
        resolved_max_train_steps = int(max_train_steps)
        resolved_num_train_epochs = math.ceil(
            resolved_max_train_steps / steps_per_epoch
        )
    return (
        gradient_accumulation_steps,
        samples_per_epoch,
        per_rank_samples_per_epoch,
        micro_batches_per_epoch,
        steps_per_epoch,
        resolved_max_train_steps,
        resolved_num_train_epochs,
    )


def _launch_eval(
    *,
    target_model_name_or_path: str,
    checkpoint_dir: str,
    step: int,
    tensorboard_dir: str,
    exp_name: str,
) -> None:
    print("You can use this function to launch your auto eval script!")

class BaseTrainer:
    def __init__(self, local_rank, args):
        self.args = args
        self.device, self.global_rank, self.world_size = init_dist(local_rank)
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir
        self.resume_checkpoint_dir = discover_latest_checkpoint(
            # self.checkpoint_dir_root
            self.args.logging.resume_checkpoint_dir
        )
        print(f"self.checkpoint_dir_root: {self.checkpoint_dir_root}")
        print(f"self.resume_checkpoint_dir: {self.resume_checkpoint_dir }")
        self.suspend_controller = SuspendController(device=self.device)
        self.next_micro_step = 0
        self.sampler_seed = int(self.args.seed)

        if is_global_main_process(): ensure_dir(self.checkpoint_dir_root)
        training_logger.init(
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
        )
        torch.cuda.memory._record_memory_history(max_entries=100000)

        self.draft_model, self.tokenizer = self.build_models()
        if self.resume_checkpoint_dir is not None:
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                device=self.device,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
            )
        self.model = self.draft_model
        if self.args.train.torch_compile:
            print_on_local_main("Compiling training model with torch.compile...")
            self.model = torch.compile(self.model, dynamic=True)
        self.model = self._wrap_with_fsdp(self.model)

        checkpoint_dataset_state = None
        if self.resume_checkpoint_dir is not None:
            checkpoint_dataset_state = load_checkpoint_dataset_state(
                self.resume_checkpoint_dir
            )
        self.train_dataset = TokenCacheDataset(
            cache_dir=self.args.data.token_cache_path,
            checkpoint_state=checkpoint_dataset_state,
        )
        validate_train_token_cache(
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            target_model_name_or_path=self.args.model.target_model_name_or_path,
            chat_template=self.args.data.chat_template,
            max_length=int(self.args.data.max_length),
            min_loss_tokens=int(self.args.data.min_loss_tokens),
        )
        self.target_model = self._build_online_target_model()

        (
            self.gradient_accumulation_steps,
            self.samples_per_epoch,
            self.per_rank_samples_per_epoch,
            self.micro_batches_per_epoch,
            self.steps_per_epoch,
            self.max_train_steps,
            self.args.train.num_train_epochs,
        ) = _compute_training_schedule(
            world_size=self.world_size,
            dataset_size=len(self.train_dataset),
            local_batch_size=int(self.args.train.local_batch_size),
            global_batch_size=int(self.args.train.global_batch_size),
            num_train_epochs=int(self.args.train.num_train_epochs),
            max_train_steps=self.args.train.max_train_steps,
        )

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=float(self.args.train.lr),
            total_steps=self.max_train_steps,
            warmup_ratio=float(self.args.train.warmup_ratio),
            weight_decay=float(self.args.train.weight_decay),
        )
        if self.resume_checkpoint_dir is not None:
            resume_state = load_training_state(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                optimizer=self.optimizer,
                global_rank=self.global_rank,
                world_size=self.world_size,
                local_batch_size=int(self.args.train.local_batch_size),
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                micro_batches_per_epoch=self.micro_batches_per_epoch,
                samples_per_epoch=self.samples_per_epoch,
                max_train_steps=self.max_train_steps,
                sampler_seed=self.sampler_seed,
                dataset_signature=self.train_dataset.signature,
            )
            self.next_micro_step = resume_state.next_micro_step
        else:
            print_on_local_main("Training from scratch.")
        self.info_board()

    @property
    def global_step(self):
        return self.next_micro_step // self.gradient_accumulation_steps

    def info_board(self):
        print_on_local_main("***** Running training *****")
        print_on_local_main(f"  Train dataset size = {len(self.train_dataset)}")
        print_on_local_main(f"  Token cache path = {self.train_dataset.cache_dir}")
        print_on_local_main(
            "  Cached fields = input_ids, attention_mask, loss_mask"
        )
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(f"  Global batch size = {self.args.train.global_batch_size}")
        print_on_local_main(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        print_on_local_main(f"  Steps per epoch = {self.steps_per_epoch}")
        print_on_local_main(f"  Max train steps = {self.max_train_steps}")
        print_on_local_main("  Target hidden states = generated on local GPU")

    def build_models(self):
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        # This causal-LM instance is only used to initialize the frozen draft
        # embeddings and lm_head. The target backbone used during training is
        # loaded separately on the local GPU.
        target_model = AutoModelForCausalLM.from_pretrained(
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = target_model.get_input_embeddings()
        target_lm_head = target_model.get_output_embeddings()
        assert (target_lm_head is not None) and (target_embed_tokens is not None)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        del target_model
        return draft_model, tokenizer

    def _build_online_target_model(self):
        print_on_local_main("Loading target model on each local GPU...")
        target_model = AutoModel.from_pretrained(
            self.args.model.target_model_name_or_path,
            dtype=self.precision_dtype,
            attn_implementation="sdpa",
        )
        target_model.requires_grad_(False)
        return target_model.to(device=self.device).eval()

    def _generate_target_hidden_states(self, batch):
        target_result = run_target_forward_with_hooks(
            target_model=self.target_model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_layer_ids=self.draft_model.target_layer_ids,
        )

        # The offline preparation pipeline slices every sample to its true
        # length and pads hidden states with zeros. Match that behavior here so
        # padded query positions cannot leak target activations downstream.
        padding_mask = ~batch["attention_mask"].bool().unsqueeze(-1)
        batch["target_hidden_states"] = (
            target_result.target_hidden_states.masked_fill(padding_mask, 0)
        )
        batch["target_last_hidden_states"] = (
            target_result.target_last_hidden_states.masked_fill(padding_mask, 0)
        )
        return batch

    def _build_draft_model(self, *, target_config, model_args):
        raise NotImplementedError

    def _wrap_with_fsdp(self, model):
        fsdp_kwargs = _build_fsdp_kwargs(
            sharding_strategy_name=self.args.train.sharding_strategy,
            precision_dtype=self.precision_dtype,
            world_size=self.world_size,
        )
        return FSDP(model, **fsdp_kwargs)

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            total_size=self.samples_per_epoch,
            seed=self.sampler_seed,
            start_global_offset_samples=start_offset_samples,
            num_samples=num_samples,
        )
        num_workers = int(self.args.data.num_workers)
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(self.sampler_seed)
        dataloader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=TokenCacheCollator(),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            generator=dataloader_generator,
        )
        if num_workers > 0:
            dataloader_kwargs.update(persistent_workers=True, prefetch_factor=4)
        return DataLoader(**dataloader_kwargs)

    def run_batch(self, batch):
        raise NotImplementedError

    def _checkpoint_kwargs(self):
        return dict(
            model=self.model,
            draft_model=self.draft_model,
            optimizer=self.optimizer,
            checkpoint_dir_root=self.checkpoint_dir_root,
            train_config=self.args,
            next_micro_step=self.next_micro_step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            global_rank=self.global_rank,
            world_size=self.world_size,
            local_batch_size=int(self.args.train.local_batch_size),
            micro_batches_per_epoch=self.micro_batches_per_epoch,
            samples_per_epoch=self.samples_per_epoch,
            max_train_steps=self.max_train_steps,
            sampler_seed=self.sampler_seed,
            dataset_state=self.train_dataset.state_dict(),
        )

    def save_and_eval_checkpoint(self):
        checkpoint_dir = save_checkpoint(**self._checkpoint_kwargs())
        if is_global_main_process():
            _launch_eval(
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                checkpoint_dir=checkpoint_dir,
                step=self.global_step,
                tensorboard_dir=self.args.logging.tensorboard_dir,
                exp_name=self.args.exp_name,
            )
        dist.barrier()
        return checkpoint_dir

    def _save_and_suspend(self):
        print_on_global_main("Saving checkpoint before suspending...")
        save_checkpoint(**self._checkpoint_kwargs())
        dist.barrier()
        if is_global_main_process():
            print_on_global_main("Going to suspend...")
            self.suspend_controller.go_suspend()
        dist.barrier()

    def train(self):
        self.model.train()
        if self.global_step >= self.max_train_steps:
            return

        local_batch_size = int(self.args.train.local_batch_size)
        total_micro_steps = self.max_train_steps * self.gradient_accumulation_steps
        remaining_micro_steps = total_micro_steps - self.next_micro_step
        remaining_samples = remaining_micro_steps * local_batch_size

        dataloader = self._build_train_dataloader(
            start_offset_samples=self.next_micro_step * local_batch_size,
            num_samples=remaining_samples,
        )
        prefetcher = CUDAPrefetcher(dataloader, self.device)
        training_logger.start_session(global_step=self.global_step)

        with self.suspend_controller.monitoring():
            for batch in prefetcher:
                batch = self._generate_target_hidden_states(batch)
                should_sync = (
                    (self.next_micro_step + 1) % self.gradient_accumulation_steps == 0
                )
                sync_context = nullcontext() if should_sync else self.model.no_sync()
                with sync_context:
                    loss = self.run_batch(batch) / self.gradient_accumulation_steps
                    loss.backward()
                self.next_micro_step += 1

                if not should_sync:
                    continue

                grad_norm = FSDP.clip_grad_norm_(
                    self.model,
                    float(self.args.train.max_grad_norm),
                )
                self.optimizer.step()
                training_logger.on_optimizer_step(
                    global_step=self.global_step,
                    next_micro_step=self.next_micro_step,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    max_train_steps=self.max_train_steps,
                    learning_rate=self.optimizer.get_learning_rate(),
                    grad_norm=grad_norm.item(),
                )

                if (
                    self.global_step % int(self.args.logging.checkpointing_steps) == 0
                    or self.global_step % int(self.steps_per_epoch) == 0
                ):
                    self.save_and_eval_checkpoint()
                if IS_PROFILE and (self.global_step + 1) % PROFILE_STEPS == 0:
                    # torch.cuda.synchronize()
                    # torch.cuda.empty_cache()
                    # torch.cuda.reset_peak_memory_stats()
                    try:
                        file_prefix = os.environ.get(
                            "PROFILE_FILE_PREFIX",
                            f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/DeepSpec/scripts/train/profile/memory_{os.environ.get('TIMESTAMP', '')}_rank{self.global_rank}"
                        )
                        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
                    except Exception as e:
                        print_on_local_main(f"Failed to capture memory snapshot: {e}")

                    break

                if self.suspend_controller.requested():
                    self._save_and_suspend()
                    return
        if IS_PROFILE:
            return

        self.save_and_eval_checkpoint()

    def clean_up(self):
        self.train_dataset.close()
        del self.target_model
        torch.cuda.empty_cache()
        training_logger.close()
        torch.cuda.memory._record_memory_history(enabled=None)
        dist.barrier()
        dist.destroy_process_group()
