import gc
from typing import Dict, List, Optional, Tuple, Union
import transformers
import torch
import math
from pathlib import Path

from .config import load_config
from .custom_cache import DynamicCacheWithQuery
from .custom_modeling_llama import LlamaForCausalLM, repeat_kv
from .custom_modeling_qwen2 import Qwen2ForCausalLM
from .custom_modeling_qwen3 import Qwen3ForCausalLM

PACKAGE_DIR = Path(__file__).parent
CONFIG_DIR = PACKAGE_DIR / 'configs'

# Everything that differs between supported models, keyed by model_base_class (lowercased).
# To support a new model, add one entry here and its yaml files under configs/.
LLAMA_TEMPLATE = dict(
    prompt_prefix='<|start_header_id|>user<|end_header_id|>',
    prompt_suffix='<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
    prompt_separator=' \n\n',
    generation_prefix='\n\n', # what the chat template puts after the assistant header; get_prompt() leaves it out
)
QWEN_TEMPLATE = dict(
    prompt_prefix='<|im_start|>user',
    prompt_suffix='<|im_end|>\n<|im_start|>assistant',
    prompt_separator='\n\n',
    generation_prefix='\n',
)
MODEL_SPECS = {
    # config_name = prefix of the model's default configs, e.g. configs/Llama-3.1-8B-Instruct_full_head.yaml
    'llama-3.1-8b-instruct': dict(model_class=LlamaForCausalLM, config_name='Llama-3.1-8B-Instruct', torch_dtype=torch.float16, **LLAMA_TEMPLATE),
    'llama-3.1-70b-instruct': dict(model_class=LlamaForCausalLM, config_name='Llama-3.1-70B-Instruct', torch_dtype=torch.float16, **LLAMA_TEMPLATE),
    'llama-3.2-3b-instruct': dict(model_class=LlamaForCausalLM, config_name='Llama-3.2-3B-Instruct', torch_dtype=torch.float16, **LLAMA_TEMPLATE),
    'llama-3.2-1b-instruct': dict(model_class=LlamaForCausalLM, config_name='Llama-3.2-1B-Instruct', torch_dtype=torch.float16, **LLAMA_TEMPLATE),
    'qwen2.5-7b-instruct': dict(model_class=Qwen2ForCausalLM, config_name='Qwen2.5-7B-Instruct', torch_dtype=torch.float16, **QWEN_TEMPLATE),
    # Qwen3 is trained in bf16; fp16 risks overflowing its activation outliers
    'qwen3-4b-instruct-2507': dict(model_class=Qwen3ForCausalLM, config_name='Qwen3-4B-Instruct-2507', torch_dtype=torch.bfloat16, **QWEN_TEMPLATE),
    # the Thinking chat template opens the reply with '<think>\n', so a generated CoT is the model's own reasoning
    'qwen3-4b-thinking-2507': dict(model_class=Qwen3ForCausalLM, config_name='Qwen3-4B-Thinking-2507', torch_dtype=torch.bfloat16,
                                   **{**QWEN_TEMPLATE, 'generation_prefix': '\n<think>\n'}),
}


def get_model_spec(model_base_class: str) -> Dict:
    if model_base_class.lower() not in MODEL_SPECS:
        raise ValueError(f"Unsupported model class: {model_base_class}")
    return MODEL_SPECS[model_base_class.lower()]


def infer_config(model_base_class: Optional[str], model_name_or_path: Optional[str], head_set: str) -> Dict:
    """Load configs/{config_name}_{head_set}.yaml, finding the model by model_base_class, else by its name inside model_name_or_path."""
    if model_base_class is not None:
        if model_base_class.lower() not in MODEL_SPECS:
            raise NotImplementedError(f"Config inference for model_base_class {model_base_class} is not implemented.")
        spec = MODEL_SPECS[model_base_class.lower()]
    elif model_name_or_path is not None:
        # first match in MODEL_SPECS order, e.g. 'meta-llama/Llama-3.1-8B-Instruct' -> 'llama-3.1-8b-instruct'
        matches = [s for name, s in MODEL_SPECS.items() if name in model_name_or_path.lower()]
        if not matches:
            raise NotImplementedError(f"Config inference for model_name_or_path {model_name_or_path} is not implemented.")
        spec = matches[0]
    else:
        raise ValueError("model_name_or_path or model_base_class is required to use default config")
    return load_config(CONFIG_DIR / f"{spec['config_name']}_{head_set}.yaml")

class SPEC_HEAD_SET:
    """
    A set of fixed attention heads.
    """
    FULL_SET = "full_heads"

# attention based retriever
class AttnBasedRetriever:
    query_chunk_size = 64 # query tokens per chunk in _attn_scores_from_cache

    # core needs: model_name_or_path, model_base_class, attention_head_subset
    def __init__(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
        attn_head_set: Optional[str] = None, # 1-2,3-4,5-6
        device: Optional[str] = None,
        # controling retriever
        null_query: Optional[str] = "N/A",
    ):
        config = self.setup_config(config_or_config_path, model_name_or_path, model_base_class)
        print("Using config:", config)
        self.config = config
        # override config with kwargs
        for k, v in config.items():
            setattr(self, k, v)

        # init model etc
        spec = get_model_spec(self.model_base_class)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.llm = spec['model_class'].from_pretrained(
            self.model_name_or_path,
            torch_dtype=spec['torch_dtype'],
            attn_implementation="flash_attention_2",
            device_map='auto'
        )
        self.llm.config.pad_token_id = self.llm.config.eos_token_id

        self.start_layer = 0
        self.end_layer = self.llm.config.num_hidden_layers - 1

        # override config if attn_head_set is provided
        if attn_head_set is not None:
            self.attn_head_set = attn_head_set

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.null_query = null_query

    def setup_config(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
    ):
        if config_or_config_path is not None:
            assert isinstance(config_or_config_path, (str, Dict))
            if isinstance(config_or_config_path, str):
                config = load_config(config_or_config_path)
            else:
                config = config_or_config_path
        else:
            # infer from model information
            if model_base_class is not None:
                # TODO: infer config from model_base_class
                raise NotImplementedError()
            elif model_name_or_path is not None:
                # TODO: infer config from model_name_or_path
                raise NotImplementedError()
            else:
                raise ValueError("model_name_or_path or model_base_class is required to use default config")
        return config

    def get_content_span(self, prompt: str, char_offset_to_token_idx: dict, content: str):
        if content not in prompt:
            raise ValueError(f"Content not found in the prompt.")
        
        content_char_start = prompt.index(content)
        content_char_end = content_char_start + len(content) - 1 # inclusive end index

        # find the token indices that correspond to the character indices
        start_idx = char_offset_to_token_idx[content_char_start]
        end_idx = char_offset_to_token_idx[content_char_end]
        return start_idx, end_idx
    
    def get_prompt(self, query: str, docs: List[Dict]):
        spec = get_model_spec(self.model_base_class)
        self.prompt_prefix = spec['prompt_prefix']
        self.prompt_suffix = spec['prompt_suffix']
        self.prompt_separator = spec['prompt_separator']

        self.retrieval_instruction = ' Here are some paragraphs:'
        self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.'
        
        llm_prompt = self.prompt_prefix + self.retrieval_instruction
        for i, doc in enumerate(docs):

            paragraph_text = doc['paragraph_text']
            if doc.get('title', None) is not None:
                paragraph_text = doc['title'] + '\n' + paragraph_text

            doc = f'[{i+1}] {paragraph_text}'
            llm_prompt += self.prompt_separator + doc

        llm_prompt += self.prompt_separator + self.retrieval_instruction_late + self.prompt_separator + 'Query:'
        query_prompt = f' {query}' + self.prompt_suffix
        llm_prompt += query_prompt
        return llm_prompt
    
    def compose_scoring_prompt(self, query: str, docs: List[Dict]) -> Tuple:
        # encode query and docs into a prompt
        # return a tuple of (prompt, tokenized_prompt, query_span, doc_spans)
        # use the function I give you to find the spans, do not hard code according to tokenizer
        llm_prompt = self.get_prompt(query, docs)

        prompt_tokenization_output = self.tokenizer(llm_prompt, return_offsets_mapping=True)
        prompt_token_ids = prompt_tokenization_output["input_ids"]
        char_offset_to_token_idx = self._char_offset_to_token_idx(prompt_tokenization_output["offset_mapping"])

        document_span_intervals = []
        for i, doc in enumerate(docs):
            # get the token span for this doc

            paragraph_text = doc['paragraph_text']
            if doc.get('title', None) is not None:
                paragraph_text = doc['title'] + '\n' + paragraph_text

            #################################################### 
            # TODO: should doc_content include the index, e.g. '[1] '?
            doc_content = f'[{i+1}] {paragraph_text}'
            ####################################################

            start_idx, end_idx = self.get_content_span(llm_prompt, char_offset_to_token_idx, doc_content)
            document_span_intervals.append((start_idx, end_idx))

        ################################################ 
        # TODO: should query_content includes 1) instruction, 2) suffix: '<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>'?
        # If we just use the query only, we don't have to do set self.xxx as class attributes in get_prompt() ??
        query_content = self.retrieval_instruction_late + self.prompt_separator + 'Query:' + f' {query}' + self.prompt_suffix
        query_start_idx, query_end_idx = self.get_content_span(llm_prompt, char_offset_to_token_idx, query_content)
        ################################################

        # query_start_idx, query_end_idx = self.get_content_span(llm_prompt, char_offset_to_token_idx, query)
        query_span = (query_start_idx, query_end_idx)

        return llm_prompt, prompt_token_ids, query_span, document_span_intervals

    @staticmethod
    def _char_offset_to_token_idx(offset_mapping: List[Tuple[int, int]]) -> Dict[int, int]:
        """Map every character offset covered by a token to that token's index."""
        char_offset_to_token_idx = {}
        for i, (start, end) in enumerate(offset_mapping):
            for j in range(start, end):
                char_offset_to_token_idx[j] = i
        return char_offset_to_token_idx


    def score_docs(self, query: str, docs: List[Dict]) -> Dict[str, Dict]:
        """
        score docs for a given query. return a dict of doc_id to scoring info.
        each doc is a dict with the following keys:
            - doc_id: str
            - text: str
            - title: optional str
        return a dict of doc_id to scoring info.
        """
        prompt, tokenized_prompt, query_span, doc_spans = self.compose_scoring_prompt(query, docs)
        null_prompt, tokenized_null_prompt, null_query_span, _ = self.compose_scoring_prompt(self.null_query, docs)

        # up to query prompt the tokenized_prompt is the same
        assert tokenized_null_prompt[:query_span[0]] == tokenized_prompt[:query_span[0]]
        assert query_span[0] == null_query_span[0], "Query start indices do not match between query and null query."

        # scoring with actual query
        per_token_scores, kv_cache = self.score_per_token_attention_to_query(prompt, query_span, None, 0)

        # use kv_cache from first query to speed up forward() for the calibration query.
        self._crop_cache(kv_cache, query_span[0])
        start_idx = query_span[0]

        null_per_token_scores, _ = self.score_per_token_attention_to_query(null_prompt, null_query_span, kv_cache, start_idx)

        return self._aggregate_doc_scores(self._calibrate(per_token_scores, null_per_token_scores), doc_spans, docs)

    @staticmethod
    def _calibrate(per_token_scores: torch.Tensor, null_per_token_scores: torch.Tensor) -> torch.Tensor:
        """Subtract the null-query scores over the tokens both passes share (the null tail is usually shorter)."""
        min_length = min(per_token_scores.shape[-1], null_per_token_scores.shape[-1])
        return per_token_scores[:,:,:min_length] - null_per_token_scores[:,:,:min_length]

    def _crop_cache(self, kv_cache, length: int):
        """Drop everything after the first `length` tokens so the cache can be extended with a new tail."""
        for i in range(len(kv_cache.key_cache)):
            kv_cache.key_cache[i] = kv_cache.key_cache[i][:,:,:length,:]
            kv_cache.value_cache[i] = kv_cache.value_cache[i][:,:,:length,:]
        kv_cache._seen_tokens = length

    def _prefill(self, prefix_ids: List[int]) -> DynamicCacheWithQuery:
        """Forward the shared prefix once; no query vectors are needed from it. Tails go on top with _score_tail()."""
        kv_cache = DynamicCacheWithQuery(query_indices=[])
        with torch.no_grad():
            self.llm(input_ids=torch.tensor([prefix_ids], device=self.device), use_cache=True, past_key_values=kv_cache, compute_logits=False)
        return kv_cache

    def _score_tail(self, kv_cache: DynamicCacheWithQuery, prefix_len: int, tail_ids: List[int], query_span: Tuple[int, int]) -> torch.Tensor:
        """
        Crop the cache back to the prefix, forward tail_ids on top of it, and return the mean attention from
        tail_ids[query_span[0] : query_span[1]+1] (relative to the tail start, inclusive) to every token,
        shape (num_layers, num_heads, prefix_len + len(tail_ids)).
        """
        self._crop_cache(kv_cache, prefix_len)
        kv_cache.query_cache = []
        kv_cache._query_indices = list(range(query_span[0], query_span[1] + 1))
        with torch.no_grad():
            self.llm(input_ids=torch.tensor([tail_ids], device=self.device), use_cache=True, past_key_values=kv_cache, compute_logits=False)
        return self._attn_scores_from_cache(kv_cache, prefix_len + query_span[0])

    def _attn_scores_from_cache(self, kv_cache: DynamicCacheWithQuery, query_start: int) -> torch.Tensor:
        """
        Same as the loop in score_per_token_attention_to_query, except that it
          - only computes the layers the head set uses (other layers stay zero),
          - processes the query tokens in chunks: a CoT is ~10x longer than the original query span, and
            _get_attn_weights materializes (num_heads, q_len, kv_len) plus a causal mask of the same size.
        Each chunk only sees keys up to its own last token, so the bottom-right causal mask in _get_attn_weights
        stays correct even when the query tokens are not at the end of the sequence.
        query_start: absolute position of the first cached query token.
        """
        num_layers = self.llm.config.num_hidden_layers
        num_heads = self.llm.config.num_attention_heads
        kv_len = kv_cache.key_cache[0].shape[-2]
        q_len = kv_cache.query_cache[0].shape[-2]

        if self.attn_head_set == SPEC_HEAD_SET.FULL_SET:
            layers = range(num_layers)
        else:
            layers = sorted({int(h.split('-')[0]) for h in self.attn_head_set.split(',')})

        tok_w = torch.full((q_len,), 1 / q_len) # mean over query tokens
        scores = torch.zeros(num_layers, num_heads, kv_len, dtype=torch.float32, device=self.device)
        for i in layers:
            for c0 in range(0, q_len, self.query_chunk_size):
                c1 = min(c0 + self.query_chunk_size, q_len)
                end = query_start + c1  # keys visible to the last query token of this chunk
                attn_weights = self._get_attn_weights(kv_cache.key_cache[i][:, :, :end, :], kv_cache.query_cache[i][:, :, c0:c1, :])
                scores[i, :, :end] += (attn_weights[0].float() * tok_w[c0:c1, None].to(attn_weights.device)).sum(1).to(self.device)
        return scores

    def _aggregate_doc_scores(self, per_token_scores_CAL: torch.Tensor, doc_spans: List[Tuple[int, int]], docs: List[Dict]) -> Dict[str, float]:
        """per_token_scores_CAL: (num_layers, num_heads, num_tokens). Returns doc_id -> score."""
        # Steps:
        # 1. select heads if head_set is not full
        # 2. aggregate scores across heads
        # 3. run remove abnormal scores and aggregate across tokens

        # aggregate scores across layers and heads
        if self.attn_head_set == SPEC_HEAD_SET.FULL_SET:
            per_token_scores_CAL = per_token_scores_CAL.sum(0) # sum per-token scores across layers            
            per_token_scores_CAL = per_token_scores_CAL.sum(0) # sum per-token scores across attention heads
        else:
            # convert self.attn_head_set to a list of tuples
            head_set = self.attn_head_set.split(',')
            head_set = [tuple(map(int, h.split('-'))) for h in head_set]
            # select heads if head_set is not full
            indices = torch.tensor(head_set).to(self.device)
            layers = indices[:, 0]
            heads = indices[:, 1]
            per_token_scores_CAL = per_token_scores_CAL[layers, heads]  # Shape: (num_selected_heads, num_tokens)
            per_token_scores_CAL = per_token_scores_CAL.sum(0)

        # remove abnormally calibrated scores and aggregate scores across tokens to get per-document scores
        per_doc_scores = []
        for i, doc_span in enumerate(doc_spans): 
            curr_doc_per_tok_scores_CAL = per_token_scores_CAL[doc_span[0] : doc_span[1]+1]

            threshold = curr_doc_per_tok_scores_CAL.mean() - 2*curr_doc_per_tok_scores_CAL.std()
            tok_mask = (curr_doc_per_tok_scores_CAL > threshold)

            per_doc_scores.append((curr_doc_per_tok_scores_CAL * tok_mask).sum())  # sum scores for each document

        assert len(per_doc_scores) == len(docs), "Number of per-document scores does not match number of documents."

        results = {} # doc_id -> score
        for i, doc in enumerate(docs):
            doc_id = doc['idx']
            results[doc_id] = per_doc_scores[i].item()
        return results


    def score_docs_per_head_for_detection(self, query: str, docs: List[Dict]) -> Dict[str, Dict]:
        """
        This function is used for QRHead detection.

        similar to score_docs, but return per-head scores for each document.
        each doc is a dict with the following keys:
            - doc_id: str
            - text: str
            - title: optional str
        return a dict of doc_id to scoring info. such as
        {
           "id": Tensor(num_layer, num_heads)
        }
        """
        prompt, tokenized_prompt, query_span, doc_spans = self.compose_scoring_prompt(query, docs)
        null_prompt, tokenized_null_prompt, null_query_span, _ = self.compose_scoring_prompt(self.null_query, docs)

        # up to query prompt the tokenized_prompt is the same
        assert tokenized_null_prompt[:query_span[0]] == tokenized_prompt[:query_span[0]]
        assert query_span[0] == null_query_span[0], "Query start indices do not match between query and null query."

        # scoring with actual query
        per_token_scores, kv_cache = self.score_per_token_attention_to_query(prompt, query_span, None, 0)

        # use kv_cache from first query to speed up forward() for the calibration query.
        self._crop_cache(kv_cache, query_span[0])
        start_idx = query_span[0]

        null_per_token_scores, _ = self.score_per_token_attention_to_query(null_prompt, null_query_span, kv_cache, start_idx)

        per_token_scores_CAL = self._calibrate(per_token_scores, null_per_token_scores) # shape: (n_layers, n_heads, n_tok)

        # Steps:
        # 1. each doc has a (n_layers, n_heads, n_tok) per_token_scores_CAL Tensor
        # 2. run the calibration to remove abnormal scores and aggregate across tokens
        # 3. for each doc, get a (n_layers, n_heads) score tensor

        # remove abnormally calibrated scores and aggregate scores across tokens to get per-document scores tensor (n_layers, n_heads)
        per_doc_score_tensors = [] # a list of (n_layers, n_heads) tensors
        for i, doc_span in enumerate(doc_spans):
            curr_doc_per_tok_scores_CAL = per_token_scores_CAL[:, :, doc_span[0] : doc_span[1]+1] # shape: (n_layers, n_heads, n_tok)

            threshold = curr_doc_per_tok_scores_CAL.mean(dim=-1) - 2*curr_doc_per_tok_scores_CAL.std(dim=-1) # shape: (n_layers, n_heads)

            # broadcast threshold over tokens: compare (n_layers, n_heads, n_tok) > (n_layers, n_heads, 1)
            tok_mask = curr_doc_per_tok_scores_CAL > threshold.unsqueeze(-1)  # shape (n_layers, n_heads, n_tok)
            
            # zero out tokens below threshold and sum over tokens -> (n_layers, n_heads)
            masked_scores = curr_doc_per_tok_scores_CAL.masked_fill(~tok_mask, 0.0)
            masked_scores = masked_scores.sum(dim=-1)  # (n_layers, n_heads)
            per_doc_score_tensors.append(masked_scores)

        assert len(per_doc_score_tensors) == len(docs), "Number of per-document scores does not match number of documents."

        results = {} # doc_id -> score tensor
        for i, doc in enumerate(docs):
            doc_id = doc['idx']
            results[doc_id] = per_doc_score_tensors[i]  # shape: (n_layers, n_heads)
        return results


    def score_per_token_attention_to_query(self, prompt, query_span, kv_cache=None, start_idx=0):
        # return num_layers * num_heads * num_tokens, up to query_span
        tokenized_input = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = tokenized_input.input_ids[:, start_idx:]
        query_indices = list(range(query_span[0]-start_idx, query_span[1]-start_idx+1))
        
        if kv_cache is None:
            kv_cache = DynamicCacheWithQuery(query_indices=query_indices)
        else:
            kv_cache.query_cache = []
            kv_cache._query_indices = query_indices

        with torch.no_grad():
            output = self.llm(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True,
                compute_logits=False,
            )
        kv_cache = output.past_key_values
        
        per_token_scores = []
        # loop through all layers and compute attention scores
        for i in range(self.start_layer, self.end_layer+1):                     
            attn_weights = self._get_attn_weights(kv_cache.key_cache[i], kv_cache.query_cache[i]).to(self.device).squeeze(0)  ######## TODO: [:,:,:query_span[0]+1] OR [:,:,:query_span[1]+1] OR [:,:,:] ???
            attn_weights = attn_weights.mean(1) # average over query tokens
            per_token_scores.append(attn_weights.squeeze(0))

        per_token_scores = torch.stack(per_token_scores, dim=0) # (num_layers, num_heads, num_tokens)
        return per_token_scores, kv_cache

    def _get_attn_weights(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = repeat_kv(key_states, num_key_value_groups)
    
        # Scale before multiplication to prevent overflow
        scale = 1.0 / math.sqrt(head_dim)
        scaled_queries = query_states * scale
        attn_weights = torch.matmul(scaled_queries, key_states.transpose(2,3))

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")
        
        # make causal mask and add it to attention weights.
        causal_mask = self._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(0)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True) # Log-sum-exp of attention weights for numerical stability in softmax.
        attn_weights = torch.exp(attn_weights - attn_lses) # softmax
        return attn_weights
    
    def _get_causal_mask(self, attn_weights):
        query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2).squeeze(0))
        causal_mask = torch.triu(causal_mask, diagonal=-(seq_len-query_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        return causal_mask






class FullHeadRetriever(AttnBasedRetriever):
    def __init__(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
        attn_head_set: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(config_or_config_path, model_name_or_path, model_base_class, attn_head_set, device)
        assert self.attn_head_set == SPEC_HEAD_SET.FULL_SET

    def setup_config(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
    ):
        if config_or_config_path is not None:
            assert isinstance(config_or_config_path, (str, Dict))
            if isinstance(config_or_config_path, str):
                config = load_config(config_or_config_path)
            else:
                config = config_or_config_path
        else:
            # infer from model information
            config = infer_config(model_base_class, model_name_or_path, 'full_head')
        return config




class QRRetriever(AttnBasedRetriever):
    def __init__(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
        attn_head_set: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(config_or_config_path, model_name_or_path, model_base_class, attn_head_set, device)

    def setup_config(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
    ):
        if config_or_config_path is not None:
            assert isinstance(config_or_config_path, (str, Dict))
            if isinstance(config_or_config_path, str):
                config = load_config(config_or_config_path)
            else:
                config = config_or_config_path
        else:
            print("config_or_config_path is not provided. Use default config: LME for qr-head.", flush=True)
            # infer from model information, default qr-head config is LME
            config = infer_config(model_base_class, model_name_or_path, 'qr_head_LME')
        return config


class CoTQRRetriever(QRRetriever):
    """
    QRRetriever that uses a chain of thought (CoT) the model generates over QRRetriever's own retrieval prompt as the
    query. Not to be confused with QRCoTRetriever below (the QR-CoT repo's method), which scores the task's own prompt
    with an output generated elsewhere.

    The docs prefix is prefilled once; every later step crops the KV cache back to it and forwards a new tail:
        prefix = [BOS, retrieval instruction, doc1..docN]          (identical to QRRetriever)
        tail   = [CoT instruction, Query: q, assistant header]
        1) generate:  prefix | tail          -> greedy-decode the CoT
        2) score:     prefix | tail + CoT    with the instruction + question + CoT tokens as the query
        3) calibrate: prefix | null tail
    The query span starts at the CoT instruction, as QRRetriever's starts at its instruction; starting at the question
    instead (before 2026-10-24) cost 7-15 Recall@|gold| points on oolong test64k.
    """
    def __init__(self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
        attn_head_set: Optional[str] = None,
        device: Optional[str] = None,
        cot_instruction: str = 'Think step by step about the following query using the paragraphs above, then answer it. If no relevant information is found, just say so briefly.',
        max_new_tokens: int = 512,
    ):
        super().__init__(config_or_config_path, model_name_or_path, model_base_class, attn_head_set, device)
        self.cot_instruction = cot_instruction
        self.max_new_tokens = max_new_tokens
        self.generation_prefix = get_model_spec(self.model_base_class)['generation_prefix']

    def _build_tail(self, instruction: str, query: str) -> Tuple[str, List[int]]:
        """Text and token ids of the part after the docs prefix. Needs the template attributes set by get_prompt()."""
        tail_text = instruction + self.prompt_separator + 'Query:' + f' {query}' + self.prompt_suffix + self.generation_prefix
        return tail_text, self.tokenizer(tail_text, add_special_tokens=False).input_ids

    def compose_cot_prompt(self, query: str, docs: List[Dict]) -> Tuple:
        """
        Keep QRRetriever's prompt up to the query (so doc spans and what the QRHeads see over the docs are unchanged)
        and replace everything after it with the CoT tail.
        Returns (cot_prompt, prefix_ids, tail_ids, doc_spans). cot_prompt is the plain-text prompt, e.g. for
        generating the CoT offline with vLLM.
        """
        llm_prompt, prompt_ids, query_span, doc_spans = self.compose_scoring_prompt(query, docs)
        prefix_ids = prompt_ids[:query_span[0]]
        tail_text, tail_ids = self._build_tail(self.cot_instruction, query)

        cot_prompt = llm_prompt[:llm_prompt.rindex(self.retrieval_instruction_late)] + tail_text
        # prefix and tail are tokenized separately; that must match tokenizing the whole prompt at once
        assert self.tokenizer(cot_prompt).input_ids == prefix_ids + tail_ids, "Tokenization differs at the prefix/tail boundary."
        return cot_prompt, prefix_ids, tail_ids, doc_spans

    def score_docs(self, query: str, docs: List[Dict], cot_ids: Optional[List[int]] = None, return_cot: bool = False):
        """
        cot_ids: optional pre-generated CoT (e.g. from vLLM on compose_cot_prompt()[0]), tokenized with
                 add_special_tokens=False and without EOS. Generated greedily here if not given.
        Returns doc_id -> score, or (doc_id -> score, cot_ids) if return_cot.
        """
        _, prefix_ids, tail_ids, doc_spans = self.compose_cot_prompt(query, docs)
        prefix_len = len(prefix_ids)
        kv_cache = self._prefill(prefix_ids)

        # with max_new_tokens=0 the CoT is empty and the query is just the question (no-CoT control)
        if cot_ids is None:
            cot_ids = self.generate_cot(kv_cache, prefix_len, tail_ids)

        # score with the whole tail + CoT as the query: from the instruction's first token to the last CoT token
        # (span is relative to the tail start, inclusive; it also covers the question and the assistant header)
        per_token_scores = self._score_tail(kv_cache, prefix_len, tail_ids + cot_ids, (0, len(tail_ids) + len(cot_ids) - 1))

        # calibrate with a null query under the same CoT instruction (mirrors QRRetriever's null-query calibration)
        _, null_tail_ids = self._build_tail(self.cot_instruction, self.null_query)
        null_per_token_scores = self._score_tail(kv_cache, prefix_len, null_tail_ids, (0, len(null_tail_ids) - 1))

        results = self._aggregate_doc_scores(self._calibrate(per_token_scores, null_per_token_scores), doc_spans, docs)
        return (results, cot_ids) if return_cot else results

    def generate_cot(self, kv_cache: DynamicCacheWithQuery, prefix_len: int, tail_ids: List[int]) -> List[int]:
        """Greedy-decode a CoT on top of the prefilled docs prefix. Returns token ids without EOS."""
        self._crop_cache(kv_cache, prefix_len)
        # the model slices query_states[:, :, _query_indices] on every forward; while decoding q_len is 1,
        # so any real index would be out of range. Cache no query vectors during generation.
        kv_cache._query_indices = []
        kv_cache.query_cache = []

        eos = self.llm.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}

        cot_ids = []
        input_ids = torch.tensor([tail_ids], device=self.device)
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                logits = self.llm(input_ids=input_ids, use_cache=True, past_key_values=kv_cache).logits
                next_id = logits[0, -1].argmax().item()
                if next_id in eos_ids:
                    break
                cot_ids.append(next_id)
                input_ids = torch.tensor([[next_id]], device=self.device)
        return cot_ids


class QRCoTRetriever(QRRetriever):
    """
    The QR-CoT repo's method (src/qrcotretriever/attn_cot_retriever.py): score docs by the attention from an output
    generated elsewhere (e.g. a Thinking model's reasoning + answer, sampled by vLLM over the full task prompt) to the
    docs inside that same prompt. Not to be confused with CoTQRRetriever above, which wraps the docs in QRRetriever's
    retrieval prompt and generates its own CoT.

        prefix = formatted_input   the chat-template-applied prompt exactly as the generator saw it; holds the docs
        tail   = output            every output token is in the query (plain mean)
        null   = null_query ("N/A") as the output
    Differences from the QR-CoT repo, none of which change the method: the model loads in its MODEL_SPECS dtype
    (bf16 for Qwen3, the repo uses fp16), attention is accumulated in float32 over the head set's layers only, and
    docs are searched in order (the repo searches each from the start, so a repeated line maps to its first copy).
    """

    def compose_output_prompt(self, formatted_input: str, output: str, docs: List[Dict]) -> Tuple:
        """
        Returns (prefix_ids, out_ids, null_out_ids, doc_spans): the tokens of formatted_input, of output and of the null
        output after it, and each doc's (start, end) token span inside formatted_input (inclusive).
        docs must be in the order they appear in formatted_input; each is searched as its stripped paragraph_text
        (preceded by its title and a newline, if it has one).
        """
        if not output:
            raise ValueError("empty output: there are no output tokens to score with")
        # formatted_input already holds the template's special tokens, so add none
        tokenized = self.tokenizer(formatted_input + output, return_offsets_mapping=True, add_special_tokens=False)
        out_start = self._output_start(tokenized["offset_mapping"], len(formatted_input))
        prefix_ids, out_ids = tokenized["input_ids"][:out_start], tokenized["input_ids"][out_start:]

        # the null pass reuses the prefix's KV cache, so the prefix must tokenize the same in front of either output
        tokenized_null = self.tokenizer(formatted_input + self.null_query, return_offsets_mapping=True, add_special_tokens=False)
        null_out_start = self._output_start(tokenized_null["offset_mapping"], len(formatted_input))
        assert tokenized_null["input_ids"][:null_out_start] == prefix_ids, "formatted_input tokenizes differently in front of the null output."
        null_out_ids = tokenized_null["input_ids"][null_out_start:]

        char_offset_to_token_idx = self._char_offset_to_token_idx(tokenized["offset_mapping"][:out_start])
        doc_spans = []
        search_from = 0
        for doc in docs:
            paragraph_text = doc['paragraph_text'].strip()
            if doc.get('title', None) is not None:
                paragraph_text = doc['title'].strip() + '\n' + paragraph_text
            char_start = formatted_input.find(paragraph_text, search_from)
            if char_start < 0:
                raise ValueError(f"Doc {doc['idx']} not found in formatted_input after character {search_from}.")
            char_end = char_start + len(paragraph_text) - 1 # inclusive
            doc_spans.append((char_offset_to_token_idx[char_start], char_offset_to_token_idx[char_end]))
            search_from = char_end + 1
        return prefix_ids, out_ids, null_out_ids, doc_spans

    @staticmethod
    def _output_start(offset_mapping: List[Tuple[int, int]], output_char_start: int) -> int:
        """Index of the first token that starts at or after output_char_start; a token straddling it stays in the prefix."""
        for i, (start, _) in enumerate(offset_mapping):
            if start >= output_char_start:
                return i
        raise ValueError("No token starts inside the output; it may tokenize into the end of formatted_input.")

    def score_docs(self, formatted_input: str, output: str, docs: List[Dict]) -> Dict[str, float]:
        """
        formatted_input: tokenizer.apply_chat_template([user prompt], tokenize=False, add_generation_prompt=True), as
                         saved by the generation script; formatted_input + output is exactly what the generator saw.
        output: the generated text, without EOS.
        Returns doc_id -> score.
        """
        prefix_ids, out_ids, null_out_ids, doc_spans = self.compose_output_prompt(formatted_input, output, docs)
        prefix_len = len(prefix_ids)
        kv_cache = self._prefill(prefix_ids)
        per_token_scores = self._score_tail(kv_cache, prefix_len, out_ids, (0, len(out_ids) - 1))
        null_per_token_scores = self._score_tail(kv_cache, prefix_len, null_out_ids, (0, len(null_out_ids) - 1))
        return self._aggregate_doc_scores(self._calibrate(per_token_scores, null_per_token_scores), doc_spans, docs)
