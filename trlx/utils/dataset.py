# %%
from os.path import isfile
from typing import List, Tuple, Union

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, PreTrainedModel


class ContextDistillDataset(Dataset):
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
            self.teacher_distribution = torch.load(cache_path)
        
        self.dataset = dataset

    @staticmethod
    def squash_other_logits(
        logits: torch.FloatTensor,
        k: int,
    ) -> torch.FloatTensor:
        """Select all logits outside of top-k and squash them together"""
        vocab_size = logits.shape[-1]

        other_logits = torch.topk(
            logits, vocab_size - k, dim=-1, largest=False,
        )[0].sum(dim=-1, keepdim=True)
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
        k: Union[int, None] = 50,
    ) -> torch.FloatTensor:
        if k is not None:
            topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
            other_logits = self.squash_other_logits(logits, k)
            log_probs = torch.log_softmax(
                torch.concat((topk_logits, other_logits), dim=-1),
                dim=-1
            ).cpu()
            topk_idx = topk_idx.cpu()
            return log_probs, topk_idx
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _cache_distribution(
        self,
        dataset: Dataset,
        prompt: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        k: Union[int, None] = 50,
        cache_path: Union[str, None] = None,
    ) -> Tuple[torch.Tensor]:
        start = self._get_prompt_len(prompt, tokenizer)
        self.teacher_distribution = []
        model.eval()
        for sample in tqdm(dataset):
            input_ids = tokenizer(prompt + sample, return_tensors="pt").input_ids
            input_ids = input_ids.to(model.device)
            # N.B.: we may want to only distill on logits b/w [start:-1]
            logits = model(input_ids).logits.squeeze()[start:]  # (seq_len, vocab_size)
            self.teacher_distribution.append(self._process_logits(logits, k))

        if cache_path is not None:
            torch.save(self.teacher_distribution, cache_path)
        return self.teacher_distribution

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# %%
if __name__ == "__main__":
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    
    prompt = "Human: How's it going?\nAssistant: I'm doing well. How about you?\n\nHuman: "
    dataset = [
        "Hello World! My name is Oliver.",
        # "I like to study deep learning and NLP. I used to research on optimization, too."
    ]
    
    prompt = ' '.join([str(i) for i in range(1, 10)]) + ' '
    dataset = [
        "10 11 12 13 14 15 16 17",
    ]

    prompt = (
        'A robot may not injure a human being or, through inaction, allow a human being to come to harm.\n'
        'A robot must obey the orders given it by human beings except where such orders would conflict with the First Law.\n'
    )
    dataset = ['A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.']
    
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")


    cddataset = ContextDistillDataset(
        dataset, 
        prompt=prompt, 
        model = model,
        tokenizer = tokenizer,
        k=50,
    )

    start = cddataset._get_prompt_len(prompt, tokenizer)
    for i in range(len(dataset)):
        sample = prompt + dataset[i]
        sample_input_ids = tokenizer(sample).input_ids[start:]
        print(f'sample {i}: \n{tokenizer.decode(sample_input_ids)}')
        dist, idx = cddataset.teacher_distribution[i]
        idx = idx[:,0]
        for pos, tok in enumerate(idx):
            print(f"[{pos}] original: {tokenizer.decode(sample_input_ids[pos])}; top-1: {tokenizer.decode(tok)}")
        



