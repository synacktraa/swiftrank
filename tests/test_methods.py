from swiftrank import ReRankPipeline

PIPELINE = ReRankPipeline.from_model_id("ms-marco-TinyBERT-L-2-v2")

QUERY = "Tricks to accelerate LLM inference"

CONTEXTS = [
    "Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
    "LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
    "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second.  - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint.  - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
    "Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
    "vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels"
]

RERANKED = [
    (0.9977508, 'Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.',),
    (0.9415497, "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second.  - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint.  - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",),
    (0.47455463, 'vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels',),
    (0.43783104, 'LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper',),
    (0.043041725, 'Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.',)
]

def test_invoke_with_score():
    output = PIPELINE.invoke_with_score(query=QUERY, contexts=CONTEXTS)
    for idx in range(len(output)):
        assert (f"{output[idx][0]:.5f}" == f"{RERANKED[idx][0]:.5f}")

def test_invoke():
    output = PIPELINE.invoke(query=QUERY, contexts=CONTEXTS)
    for idx in range(len(output)):
        assert (output[idx] == RERANKED[idx][1])

def test_invoke_with_threshold_parameter():
    output = PIPELINE.invoke(query=QUERY, contexts=CONTEXTS, threshold=0.8)
    for idx in range(len(output)):
        assert (output[idx] == RERANKED[idx][1])

def test_invoke_with_key_parameter():
    context_map = [{'id': idx, 'content': content} for idx, content in enumerate(CONTEXTS)]
    output = PIPELINE.invoke(query=QUERY, contexts=context_map, key=lambda x: x['content'])
    for idx in range(len(output)):
        assert (output[idx]['content'] == RERANKED[idx][1])
