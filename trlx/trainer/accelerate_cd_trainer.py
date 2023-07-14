from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, PretrainedConfig

from trlx.data.configs import TRLConfig
from trlx.data.method_configs import MethodConfig, register_method
from trlx.pipeline.offline_pipeline import ContextDistillPipeline
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.data.cd_types import CDBatch


@dataclass
@register_method
class CDConfig(MethodConfig):
    """
    Config for context distillation training

    :param gen_kwargs: kwargs for generation
    :type gen_kwargs: Dict[str, Any]
    """

    gen_kwargs: dict


@register_trainer
class AccelerateCDTrainer(AccelerateRLTrainer):
    """
    Context Distillation trainer adapted from AccelerateSFTTrainer
    """

    def __init__(self, context: str, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self.loss_fct = nn.KLDivLoss(reduction=None, log_target=True)
        self.context = context
        self.logit_size = kwargs.get("logit_size", 50)
        self.ref_cache_path = kwargs.get("ref_cache_path", None)

    def get_arch(self, config):
        from_fn = AutoModelForCausalLM.from_pretrained
        if issubclass(type(config.model.model_path), PretrainedConfig):
            from_fn = AutoModelForCausalLM.from_config

        model = from_fn(config.model.model_path)

        if config.model.peft_config is not None:
            # Initialize the peft adapter
            import peft

            peft_config = config.model.peft_config
            if not isinstance(peft_config, peft.PeftConfig):
                if isinstance(peft_config, dict):
                    peft_config = peft.get_peft_config(peft_config)
                else:
                    raise ValueError("`peft_config` should be an instance of `peft.PeftConfig` or a dict.")
            model = peft.get_peft_model(model, peft_config)
            if self.accelerator.is_main_process:
                model.print_trainable_parameters()

        return model

    def loss(self, batch: CDBatch):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        ref_logprobs = batch.logprobs
        vocab_ixs = batch.vocab_ixs

        logits = self.model(input_ids, attention_mask).logits
        top_logits = logits.gather(-1, vocab_ixs)
        stu_logprobs = ContextDistillPipeline.squash_other_logits(logits, top_logits)

        # N.B.: our current impl computes KL-divergence at every token position, and
        #       it's not clear whether we would like to include the last token logits.
        loss = self.loss_fct(stu_logprobs, ref_logprobs) # (bsz, seq_len, logit_size+1)
        kl_mask = attention_mask.detach().clone().unsqueeze(-1)
        loss = (loss * kl_mask).sum().div(attention_mask.sum())
        stats = {"loss": loss.item()}
        return loss, stats

    def prepare_learning(self):
        # TODO: need to check this for context distill compatibility
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(self.model, self.opt, train_dataloader, eval_dataloader)

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def make_experience(self, samples, seq_length):
        if isinstance(samples[0], str):
            self.store = ContextDistillPipeline(
                self.context,
                samples,
                seq_length,
                self.model,
                self.tokenizer,
                logit_size=self.logit_size,
                ref_cache_path=self.ref_cache_path,
            )
        else:
            raise NotImplementedError(
                "The vanilla Anthropic context distillation does not consider"
                "tuning exclusively on assistant responses, but this is something"
                "we may experiment with in the future."
            )
