"""Pre-tokenize JSONL conversations into a token-only training cache.

The output stores only input_ids, attention_mask and loss_mask.  It never loads
the target model and never writes target hidden states.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deepspec.data import ConversationCollator
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.token_cache_dataset import (
    AsyncTokenCacheWriter,
    LocalTokenCacheWriteSummary,
    atomic_json_dump,
    build_global_token_cache_shard_map,
    build_token_cache_manifest,
    cleanup_token_cache_tmp_dir,
    compute_local_sample_range,
    finalize_token_cache_index,
    load_local_token_cache_summary,
    prepare_token_cache_output_dir,
    rename_local_token_cache_shards,
    tokenizer_signature,
    write_token_cache_manifest,
)
from deepspec.utils import (
    CustomJSONEncoder,
    get_git_diff,
    get_git_sha,
    init_dist,
    is_global_main_process,
    load_config,
    main_process_first,
    parse_opts_to_config,
    print_on_global_main,
    print_on_local_main,
    seed_all,
)


os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Training JSONL path. Repeat this argument to use multiple files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    cli_args = parser.parse_args()
    config = parse_opts_to_config(cli_args.opts, load_config(cli_args.config))
    return cli_args, config


def _source_jsonl_metadata(paths):
    metadata = []
    for path in paths:
        absolute_path = os.path.abspath(path)
        stat = os.stat(absolute_path)
        metadata.append(
            {
                "path": absolute_path,
                "size": int(stat.st_size),
                "mtime_ns": int(
                    getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
                ),
            }
        )
    return metadata


def _print_prepare_progress(*, global_rank: int, processed_samples: int, total_samples: int):
    print(
        f"[token-cache rank {global_rank}] "
        f"{processed_samples}/{total_samples} source samples",
        flush=True,
    )


def main(local_rank: int):
    cli_args, config = parse_args()
    train_data_paths = [os.path.abspath(path) for path in cli_args.train_data_path]
    min_loss_tokens = int(cli_args.min_loss_tokens)
    seed_all(int(config.seed))
    _device, global_rank, world_size = init_dist(local_rank)
    output_dir = os.path.abspath(cli_args.output_dir)

    print_on_local_main(json.dumps(config, indent=4, cls=CustomJSONEncoder), flush=True)
    print_on_local_main(
        json.dumps(
            {
                "train_data_path": train_data_paths,
                "output_dir": output_dir,
                "min_loss_tokens": min_loss_tokens,
                "max_shard_bytes": int(cli_args.max_shard_bytes),
                "local_batch_size": int(cli_args.local_batch_size),
                "num_workers": int(cli_args.num_workers),
                "cached_fields": ["input_ids", "attention_mask", "loss_mask"],
            },
            indent=4,
        ),
        flush=True,
    )

    if global_rank == 0:
        prepare_token_cache_output_dir(output_dir)
    dist.barrier()

    rank_dir = os.path.join(output_dir, "_tmp", f"rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)
    with main_process_first():
        dataset = JsonLineDataset(data_paths=train_data_paths)

    local_start, local_end = compute_local_sample_range(
        num_samples=len(dataset),
        rank=global_rank,
        world_size=world_size,
    )
    local_total_samples = local_end - local_start
    local_subset = Subset(dataset, range(local_start, local_end))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target_model_name_or_path,
    )
    train_collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template=config.data.chat_template,
        max_length=int(config.data.max_length),
        min_loss_tokens=min_loss_tokens,
    )
    num_workers = int(cli_args.num_workers)
    dataloader_kwargs = dict(
        dataset=local_subset,
        batch_size=int(cli_args.local_batch_size),
        collate_fn=train_collator,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    if num_workers > 0:
        dataloader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    dataloader = DataLoader(**dataloader_kwargs)
    writer = AsyncTokenCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(cli_args.max_shard_bytes),
        max_queue_size=max(int(cli_args.local_batch_size) * 4, 1),
    )

    processed_local_samples = 0
    last_progress_printed = 0
    try:
        for batch_idx, batch in enumerate(dataloader):
            processed_local_samples = min(
                (batch_idx + 1) * int(cli_args.local_batch_size),
                local_total_samples,
            )
            should_print_progress = (
                processed_local_samples - last_progress_printed >= 10_000
                or processed_local_samples == local_total_samples
            )
            if batch is not None:
                seq_lens = batch["attention_mask"].sum(dim=1).tolist()
                for sample_index, seq_len in enumerate(seq_lens):
                    seq_len = int(seq_len)
                    writer.write_sample(
                        input_ids=batch["input_ids"][sample_index, :seq_len],
                        attention_mask=batch["attention_mask"][sample_index, :seq_len],
                        loss_mask=batch["loss_mask"][sample_index, :seq_len],
                    )
            if should_print_progress:
                _print_prepare_progress(
                    global_rank=global_rank,
                    processed_samples=processed_local_samples,
                    total_samples=local_total_samples,
                )
                last_progress_printed = processed_local_samples
    finally:
        writer.close()

    dataset.close()
    summary = LocalTokenCacheWriteSummary(
        global_rank=global_rank,
        source_sample_start=local_start,
        source_sample_end=local_end,
        num_local_samples=writer.num_local_samples,
        local_shards=list(writer.local_shards),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))
    dist.barrier()

    shard_map = None
    shards = None
    summaries = None
    if is_global_main_process():
        summaries = [
            load_local_token_cache_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )
            for rank in range(world_size)
        ]
        shard_map, shards = build_global_token_cache_shard_map(summaries)
    broadcast_payload = [shard_map]
    dist.broadcast_object_list(broadcast_payload, src=0)
    shard_map = broadcast_payload[0]
    local_summary = load_local_token_cache_summary(rank_dir)
    rename_local_token_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=local_summary,
        shard_map=shard_map,
    )
    dist.barrier()

    if is_global_main_process():
        assert summaries is not None
        assert shards is not None
        index_metadata = finalize_token_cache_index(
            output_dir=output_dir,
            summaries=summaries,
            shard_map=shard_map,
        )
        manifest = build_token_cache_manifest(
            num_samples=int(index_metadata["num_samples"]),
            num_source_samples=len(dataset),
            shards=shards,
            index_metadata=index_metadata,
            extra_fields={
                "target_model_name_or_path": str(
                    config.model.target_model_name_or_path
                ),
                "source_jsonl_files": _source_jsonl_metadata(train_data_paths),
                "chat_template": str(config.data.chat_template),
                "max_length": int(config.data.max_length),
                "min_loss_tokens": min_loss_tokens,
                "tokenizer": tokenizer_signature(tokenizer),
                "cached_fields": ["input_ids", "attention_mask", "loss_mask"],
                "git_sha": str(get_git_sha()),
            },
        )
        write_token_cache_manifest(output_dir=output_dir, manifest=manifest)
        cleanup_token_cache_tmp_dir(output_dir)
        print_on_global_main(
            f"Prepared token cache at {output_dir} with "
            f"{manifest['num_samples']}/{manifest['num_source_samples']} valid "
            "samples. No hidden states were written."
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print("git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())
