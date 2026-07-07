from deepspec.data import CacheCollator
# from deepspec.modeling.myspec.gemma4 import Gemma4DSparkModel
# from deepspec.modeling.myspec.gemma4.config import (
#     build_draft_config as build_gemma4_draft_config,
# )
from deepspec.modeling.myspec.loss import compute_myspec_loss
from deepspec.modeling.myspec.qwen3 import Qwen3MySpecModel
from deepspec.modeling.myspec.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3MySpecTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3MySpecModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        """
        outputs: DSparkForwardOutput(
            draft_logits=draft_logits, # [bsz, num_blocks, block_size, vocab_size]
            target_ids=target_ids, # [bsz, num_blocks, block_size], [input_ids[:, anchor_pos + 1], input_ids[:, anchor_pos + 2], ..., input_ids[:, min(anchor_pos + block_size, seq_len - 1)]]
            eval_mask=eval_mask, # [bsz, num_blocks, block_size], bool
            block_keep_mask=block_keep_mask, # [bsz, num_anchors], num_anchors == num_blocks
            confidence_pred=confidence_pred, # [bsz, num_blocks, block_size, 1] -> [bsz, num_blocks, block_size]
            aligned_target_logits=aligned_target_logits, # [bsz, num_blocks, block_size, vocab_size]
        )
        """
        loss = compute_myspec_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        return loss


# class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
#     def _build_draft_model(self, *, target_config, model_args):
#         draft_config = build_gemma4_draft_config(
#             target_config=target_config,
#             model_args=model_args,
#         )
#         return Gemma4DSparkModel(draft_config)
