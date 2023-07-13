# %%
from os.path import isfile
from typing import List, Tuple, Union

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, PreTrainedModel


class ContextDistillDataset(Dataset):
    """
    A re-implementation to Anthropic's context distillation (https://arxiv.org/abs/2209.15189).

    N.B: this is qualitatively different from https://arxiv.org/abs/2209.15189, which samples
    model response from unlabeled inputs.
    """

    def __init__(
        self,
        dataset: List[str],
        **cache_config,
    ):
        super().__init__()
        cache_path = cache_config.get("cache_path", None)

        if cache_path is None or not isfile(cache_path):
            print(f"Cache not found. Building cache to {cache_path}...")
            self._cache_distribution(dataset, **cache_config)
        else:
            self.ref_dist = torch.load(cache_path)

        self.dataset = dataset

    @staticmethod
    def squash_other_logits(
        logits: torch.FloatTensor,
        k: int,
    ) -> torch.FloatTensor:
        """Select all logits outside of top-k and squash them together"""
        vocab_size = logits.shape[-1]

        other_logits = torch.topk(logits, vocab_size - k, dim=-1, largest=False,)[
            0
        ].sum(dim=-1, keepdim=True)
        return other_logits

    def _get_prompt_len(
        self,
        prompt: str,
        tokenizer: PreTrainedTokenizer,
    ) -> int:
        """Determine the correct prompt length"""
        if prompt.endswith(" "):
            prompt = prompt.strip()
        return len(tokenizer(prompt).input_ids)

    def _process_logits(
        self,
        logits: torch.FloatTensor,
        k: int = 50,
    ) -> Tuple[torch.Tensor]:

        topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
        other_logits = self.squash_other_logits(logits, k)
        log_probs = torch.log_softmax(torch.cat((topk_logits, other_logits), dim=-1), dim=-1).cpu()
        topk_idx = topk_idx.cpu()
        return {
            "log_probs": log_probs,
            "topk_idx": topk_idx,
        }

    @torch.no_grad()
    def _cache_distribution(
        self,
        dataset: Dataset,
        prompt: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        k: int = 50,
        cache_path: Union[str, None] = None,
    ) -> Tuple[torch.Tensor]:
        start = self._get_prompt_len(prompt, tokenizer)
        self.ref_dist = []
        model.eval()
        for sample in tqdm(dataset):
            input_ids = tokenizer(prompt + sample, return_tensors="pt").input_ids
            input_ids = input_ids.to(model.device)
            logits = model(input_ids).logits.squeeze()[start:-1]  # (seq_len, vocab_size)
            self.ref_dist.append(self._process_logits(logits, k))

        if cache_path is not None:
            torch.save(self.ref_dist, cache_path)
        return self.ref_dist

    def __getitem__(self, index):
        return self.dataset[index], self.ref_dist[index]

    def __len__(self):
        return len(self.dataset)
