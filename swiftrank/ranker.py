import json
from pathlib import Path
from collections import OrderedDict
from typing import (
    overload, Any, Optional, Iterable, Callable, TypeVar
)

import numpy as np
import onnxruntime as ort
from tokenizers import AddedToken, Tokenizer as TokenizerLoader

from . import settings


_T = TypeVar("_T")


class Ranker:
    """Load Ranker from available models."""
    def __init__(
        self, model_id: str = settings.DEFAULT_MODEL
    ) -> None:
        self.model_id = model_id
        model_file = settings.MODEL_MAP.get(self.model_id)
        if model_file is None:
            raise LookupError(f"{self.model_id!r} model not available.")
        self.instance = ort.InferenceSession(
            settings.get_model_path(model_id=self.model_id) / model_file
        )


class Tokenizer:
    """Load Tokenizer from available models."""
    def __init__(
        self, model_id: str = settings.DEFAULT_MODEL, max_length: int = 512
    ) -> None:
        self.model_id = model_id
        self.model_dir = settings.get_model_path(model_id=self.model_id) 
        self.max_length = max_length
        self.instance = self.__load()
    
    def __file_handler(self, filename: str, read_json: bool = True) -> dict[str, Any] | Path:
        """Json file handler. If read_json is true, returns loaded object else path is returned"""
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{filename!r} file missing from {self.model_dir!r}")
        if read_json:
            return json.loads(path.read_bytes())
        return path

    def __load_vocab(self, vocab_file: Path):
        """Load vocab file"""
        vocab, ids_to_tokens = OrderedDict(), OrderedDict() 
        with vocab_file.open(encoding="utf-8") as handler:
            tokens = handler.readlines()

        for idx, tok in enumerate(tokens):
            tok = tok.rstrip("\n")
            vocab[tok], ids_to_tokens[idx] = idx, tok

        return vocab, ids_to_tokens

    def __load(self):
        """Load tokenizer"""
        config = self.__file_handler("config.json")
        tokenizer_config = self.__file_handler("tokenizer_config.json")
        tokens_map = self.__file_handler("special_tokens_map.json")

        tokenizer: TokenizerLoader = TokenizerLoader.from_file(str(
            self.__file_handler("tokenizer.json", read_json=False)
        ))
        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], self.max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])
        
        vocab_file = self.model_dir / "vocab.txt"
        if vocab_file.exists():
            tokenizer.vocab, tokenizer.ids_to_tokens = self.__load_vocab(vocab_file)

        return tokenizer
    
    
class ReRankPipeline:
    """
    Pipeline for reranking task.
    :param ranker: `Ranker` class instance
    :param tokenizer: `Tokenizer` class instance

    >>> from flashrank import ReRankPipeline
    >>> pipeline = ReRankPipeline(ranker=ranker, tokenizer=tokenizer)
    >>> pipeline.invoke(
    ...     query="<query>", contexts=["<context1>", "<context2>", ...]    
    ... )
    """
    def __init__(self, ranker: Ranker, tokenizer: Tokenizer) -> None:
        self.ranker = ranker.instance
        self.tokenizer = tokenizer.instance

    @classmethod
    def from_model_id(cls, __id: str, tk_max_length: int = 512):
        """
        Create Reranker from model ID
        @param __id: Model ID
        @param tk_max_length: Max length for tokenizer
        """
        return cls(
            ranker=Ranker(model_id=__id), 
            tokenizer=Tokenizer(model_id=__id, max_length=tk_max_length)
        )

    def __create_attr_array(self, tokenized, attr: str):
        """Create array of tokenized attribute values."""
        return np.array([getattr(_, attr) for _ in tokenized], dtype=np.int64)

    @overload
    def invoke_with_score(
        self, query: str, contexts: Iterable[str], threshold: Optional[float] = None
    ) -> list[tuple[float, str]]:
        """
        Rerank contexts based on query.
        :param query: The query to use for reranking evaluation.
        :param contexts: The contexts to rerank.
        :param threshold: Get contexts that are equal or higher than threshold value.
        """
    
    @overload
    def invoke_with_score(
        self, query: str, contexts: Iterable[_T], threshold: Optional[float] = None, *, key: Callable[[_T], str]
    ) -> list[tuple[float, _T]]:
        """
        Rerank contexts based on query.
        :param query: The query to use for reranking evaluation.
        :param contexts: The contexts object.
        :param threshold: Get contexts that are equal or higher than threshold value.
        :param key: callback to use for getting fields from contexts object.
        """

    def invoke_with_score(
        self, 
        query: str, 
        contexts: Iterable, 
        threshold: Optional[float] = None, 
        *, 
        key: Callable = None
    ) -> list[tuple]:

        processor = (lambda _:_) if key is None else key
        tokenized = self.tokenizer.encode_batch(
            [(query, processor(context)) for context in contexts])

        onnx_input = {
            "input_ids": self.__create_attr_array(tokenized, 'ids'),
            "attention_mask": self.__create_attr_array(tokenized, 'attention_mask')}
        token_type_ids = self.__create_attr_array(tokenized, 'type_ids')
        use_type_ids = not np.all(token_type_ids == 0)
        if use_type_ids:
            onnx_input = onnx_input | {'token_type_ids': token_type_ids}

        output = self.ranker.run(None, onnx_input)[0]
        scores = list(1 / (1 + np.exp(
            -(output[:, 1] if output.shape[1] > 1 else output.flatten()))))
        combined = sorted(zip(scores, contexts), key=lambda x: x[0], reverse=True)

        if threshold is None:
            return [(sc, ctx) for sc, ctx in combined]
        return [(sc, ctx) for sc, ctx in combined if sc >= threshold]

    @overload
    def invoke(
        self, query: str, contexts: Iterable[str], threshold: Optional[float] = None
    ) -> list[str]:
        """
        Rerank contexts based on query.
        :param query: The query to use for reranking evaluation.
        :param contexts: The contexts to rerank.
        :param threshold: Get contexts that are equal or higher than threshold value.
        """
    
    @overload
    def invoke(
        self, query: str, contexts: Iterable[_T], threshold: Optional[float] = None, *, key: Callable[[_T], str]
    ) -> list[_T]:
        """
        Rerank contexts based on query.
        :param query: The query to use for reranking evaluation.
        :param contexts: The contexts object.
        :param threshold: Get contexts that are equal or higher than threshold value.
        :param key: callback to use for getting fields from contexts object.
        """

    def invoke(
        self, 
        query: str, 
        contexts: Iterable, 
        threshold: Optional[float] = None, 
        *, 
        key: Callable = None
    ) -> list:

        return [context for _, context in self.invoke_with_score(
            query=query, contexts=contexts, threshold=threshold, key=key)]