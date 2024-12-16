from typing import Dict, Optional
import importlib

import torch
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings, MIN_VERSION
from langchain_core.utils import pre_init
import onnxruntime as ort


assert torch.cuda.is_available(), "CUDA is not availible"
assert (
    "CUDAExecutionProvider" in ort.get_available_providers()
), "Could not find CUDAExecutionProvider"


class FastEmbedEmbeddingsGPU(FastEmbedEmbeddings):
    mtype: Optional[str] = "text"

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that FastEmbed has been installed."""
        model_name = values.get("model_name")
        max_length = values.get("max_length")
        cache_dir = values.get("cache_dir")
        threads = values.get("threads")
        mtype = values.get("mtype", "text")

        try:
            fastembed = importlib.import_module("fastembed")

        except ModuleNotFoundError:
            raise ImportError(
                "Could not import 'fastembed' Python package. "
                "Please install it with `pip install fastembed`."
            )

        if importlib.metadata.version("fastembed-gpu") < MIN_VERSION:
            raise ImportError(
                'FastEmbedEmbeddings requires `pip install -U "fastembed>=0.2.0"`.'
            )

        if mtype == "image":
            values["model"] = fastembed.ImageEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads,
                cuda=True,
            )
        else:
            values["model"] = fastembed.TextEmbedding(
                model_name=model_name,
                max_length=max_length,
                cache_dir=cache_dir,
                threads=threads,
                cuda=True,
            )
        return values
