from dataclasses import dataclass

from torchtyping import TensorType  # type: ignore

@dataclass
class CDElement:
    """
    Data element for context distillation

    :param input_ids: Input tokens. Should be a long tensor.
    :type input_ids: torch.Tensor

    :param attention_mask: Attention mask. Should be a long tensor.
    :type attention_mask: torch.Tensor

    :param logprobs: Teacher distribution log probabilities. Should be a float tensor
    :type logprobs: torch.Tensor

    :param vocab_ix: Vocabulary index associated with teacher distribution probabilities. Should be a long tensor
    :type vocab_ix: torch.Tensor
    """
    input_ids: TensorType["query_size"]
    attention_mask: TensorType["query_size"]
    logprobs: TensorType["query_size", "logit_size_plus_1"]
    vocab_ixs: TensorType["query_size", "logit_size"]

@dataclass
class CDBatch:
    """
    Batched ILQL data elements

    :param input_ids: Batch of input tokens.
    :type input_ids: torch.Tensor

    :param attention_mask: Batch of attention masks.
    :type attention_mask: torch.Tensor

    :param rewards: Batch of rewards for each token in each token batch.
    :type rewards: torch.Tensor
    """

    input_ids: TensorType["batch_size", "query_size"]
    attention_mask: TensorType["batch_size", "query_size"]
    logprobs: TensorType["batch_size", "query_size", "logit_size_plus_1"]
    vocab_ixs: TensorType["batch_size", "query_size", "logit_size"]


