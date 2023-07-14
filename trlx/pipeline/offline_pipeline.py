import math
from os.path import isfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Union

from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from trlx.data.ilql_types import (
    ILQLBatch,
    ILQLElement,
    ILQLSeq2SeqBatch,
    ILQLSeq2SeqElement,
)
from trlx.data.cd_types import (
    CDBatch,
    CDElement,
)
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


@dataclass
class DialogMessage:
    is_output: bool
    tokens: Tuple[int]


def tokenize_dialogue(  # noqa: C901
    dialogue: Union[str, Iterable[str]], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_length=2048
) -> List[DialogMessage]:
    """
    Tokenize sample with the interleaved form of (prompt_1, output_1, prompt_2, output_2...)
    """
    if isinstance(dialogue, str):
        bos_token = tokenizer.bos_token or tokenizer.eos_token
        dialogue = [bos_token, dialogue]
    elif isinstance(dialogue, Iterable):
        if len(dialogue) % 2 != 0:
            raise ValueError("Dialogue must have an even number of phrases, alternating prompt and output")
        dialogue = list(dialogue)

    if not dialogue[-1].endswith(tokenizer.eos_token):
        dialogue[-1] = dialogue[-1] + tokenizer.eos_token

    tokenized = [
        DialogMessage(is_output=i % 2 == 1, tokens=tuple(tokenizer(dialogue[i], add_special_tokens=False).input_ids))
        for i in range(len(dialogue))
    ]

    # flip to truncate from the left
    if tokenizer.truncation_side == "left":
        tokenized = [DialogMessage(is_output=m.is_output, tokens=m.tokens[::-1]) for m in tokenized[::-1]]

    # truncate if necessary
    lengths = [len(t.tokens) for t in tokenized]
    cumsum_lengths = [sum(lengths[:i]) for i in range(len(lengths))]
    truncated = [
        DialogMessage(is_output=t.is_output, tokens=t.tokens[: max(max_length - cl, 0)])
        for t, cl in zip(tokenized, cumsum_lengths)
    ]

    # flip back if was fliped to left truncate
    if tokenizer.truncation_side == "left":
        truncated = [DialogMessage(is_output=m.is_output, tokens=m.tokens[::-1]) for m in truncated[::-1]]

    # remove empty messages
    out = [t for t in truncated if len(t.tokens) > 0]

    if out[0].is_output:
        if sum(map(lambda msg: len(msg.tokens), out)) == max_length:
            if tokenizer.truncation_side == "left":
                out[0].tokens = out[0].tokens[1:]
            else:
                out[-1].tokens = out[-1].tokens[:-1]

        out.insert(0, DialogMessage(False, (tokenizer.bos_token_id,)))
    return out


class DialogStore(BaseRolloutStore):
    def __init__(self, dialogs: List[List[DialogMessage]], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        attention_masks = [torch.ones(sum(len(m.tokens) for m in d), dtype=torch.bool) for d in dialogs]
        input_ids = [torch.tensor([t for m in d for t in m.tokens], dtype=torch.long) for d in dialogs]
        # -100 is the ignore index for CrossEntropyLoss
        labels = [
            torch.tensor([t if m.is_output else -100 for m in d for t in m.tokens], dtype=torch.long) for d in dialogs
        ]
        self.history = [
            dict(input_ids=i, attention_mask=a, labels=l) for i, a, l in zip(input_ids, attention_masks, labels)
        ]

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        hf_collate_fn = DataCollatorWithPadding(self.tokenizer)

        def collate_fn(elems: Iterable[dict]):
            batch = hf_collate_fn(
                {"input_ids": [e["input_ids"] for e in elems], "attention_mask": [e["attention_mask"] for e in elems]}
            )
            labels = hf_collate_fn([{"input_ids": e["labels"]} for e in elems])["input_ids"]
            batch["labels"] = labels
            return batch

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)


@register_datapipeline
class PromptPipeline(BasePipeline):
    """
    Dataloader which is used to supply prompts for either training or evaluation

    Args:
        prompts (`List[str]` or `List[Dict[str, Any]]`): list of raw text prompts or a dictionary with a required
            key `"prompt"` and extra information, that would be passed along the generation for that prompt as a
            keyword argument to a reward function.
        max_prompt_length (`int`): max length of the prompt, if exceeded the prompt will be truncated according to
            tokenizer's truncation setting.
        tokenizer (`transformers.PreTrainedTokenizer`): a tokenizer to tokenize prompts with.
        add_special_tokens (`bool`): whether to encode prompts with tokenizer's special tokens (passed directly
            into `tokenizer.encode`)
    """

    def __init__(
        self,
        prompts: Union[List[Dict[str, Any]], List[str]],
        max_prompt_length: int,
        tokenizer: PreTrainedTokenizer,
        add_special_tokens: bool = False,
    ):
        super().__init__()

        if isinstance(prompts[0], dict):
            metadata = prompts
            prompts = [x.pop("prompt") for x in metadata]
        else:
            metadata = [{}] * len(prompts)

        model_inputs = tokenizer(
            prompts, truncation=True, padding=False, max_length=max_prompt_length, add_special_tokens=add_special_tokens
        )

        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.tokenizer = tokenizer
        self.prompts = [
            {"input_ids": tokens, "attention_mask": mask} for tokens, mask in zip(prompts_tokens, attention_mask)
        ]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False, sampler=None, drop_last=False) -> DataLoader:
        def collate_fn(xs):
            out = self.tokenizer.pad([{"input_ids": x["input_ids"]} for x in xs], return_tensors="pt")

            for key in xs[0]:
                if key != "input_ids" and key != "attention_mask":
                    out[key] = [x[key] for x in xs]

            return out

        # Since all data is already pre-processed, no need to have
        # multi-process data loading
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            drop_last=drop_last,
        )


def cd_collate_fn(elems: Iterable[CDElement]):
    return CDBatch(
        pad_sequence([torch.LongTensor(x.input_ids) for x in elems], batch_first=True, padding_value=0),
        pad_sequence([torch.LongTensor(x.attention_mask) for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.logprobs for x in elems], batch_first=True, padding_value=-math.inf),
        pad_sequence([x.vocab_ixs for x in elems], batch_first=True, padding_value=-100),
    )


@register_datapipeline
class ContextDistillPipeline(BasePipeline):
    """
    Dataloader which is used to supply prompts for either training or evaluation

    Args:
        prompts (`List[str]` or `List[Dict[str, Any]]`): list of raw text prompts or a dictionary with a required
            key `"prompt"` and extra information, that would be passed along the generation for that prompt as a
            keyword argument to a reward function.
        max_prompt_length (`int`): max length of the prompt, if exceeded the prompt will be truncated according to
            tokenizer's truncation setting.
        tokenizer (`transformers.PreTrainedTokenizer`): a tokenizer to tokenize prompts with.
        add_special_tokens (`bool`): whether to encode prompts with tokenizer's special tokens (passed directly
            into `tokenizer.encode`)
    """

    def __init__(
        self,
        context: str,
        prompts: Union[List[Dict[str, Any]], List[str]],
        max_prompt_length: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        add_special_tokens: bool = False,
        logit_size: int = 50,
        ref_cache_path: Union[str, None] = None,
    ):
        super().__init__()

        if isinstance(prompts[0], dict):
            metadata = prompts
            prompts = [x.pop("prompt") for x in metadata]
        else:
            metadata = [{}] * len(prompts)

        tokenizer_kwargs = {
            "truncation": True,
            "padding": False,
            "max_length": max_prompt_length,
            "add_special_tokens": add_special_tokens,
        }
        ctx_len = self._set_context_length(context, tokenizer, **tokenizer_kwargs)

        if ref_cache_path is None or not isfile(ref_cache_path):
            print(f"Caching teacher distribution to {ref_cache_path}...")
            self._cache_distribution(
                context,
                prompts,
                model,
                tokenizer,
                logit_size,
                ref_cache_path,
                **tokenizer_kwargs,
            )
        else:
            print(f"Loading teacher distribution from {ref_cache_path}...")
            self.ref_dist = torch.load(ref_cache_path)

        # truncate max_length by ctx_len
        tokenizer_kwargs.update({"max_length": max_prompt_length - ctx_len})
        model_inputs = tokenizer(
            prompts,
            **tokenizer_kwargs,
        )

        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.tokenizer = tokenizer
        self.prompts = [
            {"input_ids": tokens, "attention_mask": mask, **metadata}
            for tokens, mask, metadata in zip(prompts_tokens, attention_mask, metadata)
        ]

    def __getitem__(self, ix: int):
        return CDElement(
            input_ids=self.prompts[ix]["input_ids"],
            attention_mask=self.prompts[ix]["attention_mask"],
            logprobs=self.ref_dist[ix]["logprobs"],
            vocab_ixs=self.ref_dist[ix]["vocab_ixs"],
        )

    def __len__(self) -> int:
        return len(self.prompts)

    @staticmethod
    def squash_other_logits(
        logits: torch.FloatTensor,
        logit_size: int,
    ) -> torch.FloatTensor:
        """Select all logits outside of top-k and squash them together"""
        vocab_size = logits.shape[-1]

        other_logits = torch.topk(
            logits,
            vocab_size - logit_size,
            dim=-1,
            largest=False,
        )[
            0
        ].sum(dim=-1, keepdim=True)
        return other_logits

    def _set_context_length(self, context: str, tokenizer: PreTrainedTokenizer, **tokenizer_kwargs):
        if context.endswith(" "):
            _context = context.rstrip()
            if len(_context) < len(context) - 1:
                raise ValueError(f"{context} has too many trailing whitespaces!")
            context = _context
        self.ctx_len = len(
            tokenizer(
                context,
                **tokenizer_kwargs,
            )["input_ids"]
        )
        return self.ctx_len

    def _process_logits(
        self,
        logits: torch.FloatTensor,
        logit_size: int = 50,
    ) -> Dict[str, torch.Tensor]:
        top_logits, vocab_ixs = torch.topk(logits, logit_size, dim=-1)
        other_logits = self.squash_other_logits(logits, logit_size)
        logprobs = torch.log_softmax(torch.cat((top_logits, other_logits), dim=-1), dim=-1).cpu()
        vocab_ixs = vocab_ixs.cpu()
        return {
            "logprobs": logprobs,
            "vocab_ixs": vocab_ixs,
        }

    @torch.no_grad()
    def _cache_distribution(
        self,
        context: str,
        prompts: Union[List[Dict[str, Any]], List[str]],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        logit_size: int = 50,
        ref_cache_path: Union[str, None] = None,
        **tokenizer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        self.ref_dist = []
        model.eval()
        ctx_prompts = [context + prompt for prompt in prompts]
        for sample in tqdm(ctx_prompts):
            input_ids = tokenizer(sample, return_tensors="pt", **tokenizer_kwargs).input_ids
            input_ids = input_ids.to(model.device)
            logits = model(input_ids).logits.squeeze()[self.ctx_len :]  # (seq_len, vocab_size)

            self.ref_dist.append(self._process_logits(logits, logit_size))

        if ref_cache_path is not None:
            torch.save(self.ref_dist, ref_cache_path)
        return self.ref_dist

    def create_loader(self, batch_size: int, shuffle=False, sampler=None, drop_last=False) -> DataLoader:
        # Since all data is already pre-processed, no need to have
        # multi-process data loading
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=cd_collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            drop_last=drop_last,
        )


def ilql_collate_fn(elems: Iterable[ILQLElement]):
    return ILQLBatch(
        pad_sequence([x.input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.attention_mask for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.rewards for x in elems], batch_first=True, padding_value=0.0),
        pad_sequence([x.states_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.actions_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.dones for x in elems], batch_first=True, padding_value=0),
    )


class ILQLRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(self, input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_collate_fn,
            drop_last=torch.distributed.is_initialized(),
        )


def ilql_seq2seq_collate_fn(elems: Iterable[ILQLElement]):
    return ILQLSeq2SeqBatch(
        pad_sequence([x.input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.attention_mask for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.decoder_input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.rewards for x in elems], batch_first=True, padding_value=0.0),
        pad_sequence([x.states_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.actions_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.dones for x in elems], batch_first=True, padding_value=0),
    )


class ILQLSeq2SeqRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(self, input_ids, attention_mask, decoder_input_ids, rewards, states_ixs, actions_ixs, dones):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLSeq2SeqElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.decoder_input_ids[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_seq2seq_collate_fn,
            drop_last=torch.distributed.is_initialized(),
        )
