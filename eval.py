from __future__ import annotations
import argparse
import json
from deepspec.eval.base_evaluator import load_selected_dataset, save_dataset_prompts
from deepspec.utils import CustomJSONEncoder

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, default=None)
    parser.add_argument("--draft_name_or_path", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument(
        "--dataset-prompt-dir",
        type=str,
        default=None,
        help=(
            "Directory in which to save the prompts selected for evaluation. "
            "Each dataset is written to <dataset_name>.jsonl."
        ),
    )
    parser.add_argument(
        "--save-dataset-only",
        action="store_true",
        help=(
            "Save the selected dataset prompts and exit without loading model "
            "configuration or weights. Requires --dataset-prompt-dir."
        ),
    )
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    args = parser.parse_args()
    if args.save_dataset_only:
        if args.dataset_prompt_dir is None:
            parser.error("--save-dataset-only requires --dataset-prompt-dir")
    elif args.target_name_or_path is None or args.draft_name_or_path is None:
        parser.error(
            "--target_name_or_path and --draft_name_or_path are required unless "
            "--save-dataset-only is used"
        )
    args.tasks = list(TASKS)
    return args


def save_datasets_only(args) -> None:
    print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    for dataset_name, max_samples in args.tasks:
        dataset = load_selected_dataset(
            dataset_name=dataset_name,
            max_samples=max_samples,
            seed=int(args.seed),
        )
        output_path = save_dataset_prompts(
            dataset_name=dataset_name,
            dataset=dataset,
            output_dir=args.dataset_prompt_dir,
        )
        print(
            f"Saved {len(dataset)} {dataset_name} prompts to {output_path}",
            flush=True,
        )


def main(local_rank: int, args):
    from transformers import AutoConfig

    from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
    from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
    from deepspec.eval.myspec import Qwen3MySpecEvaluator

    evaluators = {
        "Qwen3DSparkModel": Qwen3DSparkEvaluator,
        "Gemma4DSparkModel": Gemma4DSparkEvaluator,
        "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
        "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
        "Eagle3DraftModel": Qwen3Eagle3Evaluator,
        "Qwen3MySpecModel": Qwen3MySpecEvaluator,
    }
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = evaluators[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    if args.save_dataset_only:
        save_datasets_only(args)
    else:
        import torch

        torch.multiprocessing.spawn(
            main,
            args=(args,),
            nprocs=torch.cuda.device_count(),
        )
