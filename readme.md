<div align="center">
  <img src="https://i.imgur.com/MYThQ5c.gif" alt="SwiftRank GIF">
</div>

---

<p align="center">Streamlined, Light-Weight, Ultra-Fast State-of-the-Art Reranker, Engineered for Both Retrieval Pipelines and Terminal Applications.</p>

> Re-write version of [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) with additional features, more flexibility and optimizations.
---

### Features 🌀

🌟 **Light Weight**:
- **No Torch or Transformers**: Operable solely on CPU.
- Boasts the **tiniest reranking model in the world, ~4MB**.

⚡ **Ultra Fast**:
- Reranking efficiency depends on the **total token count in contexts and queries, plus the depth of the model (number of layers)**.
- For illustration, the duration for the process using the standard model is exemplified in the following test:
  <img src="https://i.imgur.com/YUlDnD8.jpg" width=600/>

🎯 **Based on SoTA Cross-encoders and other models**:
- How good are Zero-shot rerankers? => [Reference](https://github.com/PrithivirajDamodaran/FlashRank/blob/main/README.md#references).
- Supported Models :-
  * `ms-marco-TinyBERT-L-2-v2` (default)
  * `ms-marco-MiniLM-L-12-v2`
  * `ms-marco-MultiBERT-L-12`  (Multi-lingual, [supports 100+ languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages))
  * `rank-T5-flan` (Best non cross-encoder reranker)
- Why only sleeker models? Reranking is the final leg of larger retrieval pipelines, idea is to avoid any extra overhead especially for user-facing scenarios. To that end models with really small footprint that doesn't need any specialised hardware and yet offer competitive performance are chosen. Feel free to raise issues to add support for a new models as you see fit.

🔧 **Versatile Configuration**:
- Implements a structured pipeline for the reranking process. `Ranker` and `Tokenizer` instances are passed to create the pipeline.
- Supports complex dictionary objects handling.
- Includes a customizable threshold parameter to filter contexts, ensuring only those with a value equal to or exceeding the threshold are selected.

⌨️ **Terminal Integration**:
- Pipe your output into `swiftrank` cli tool and get reranked output

---

### 🚀 Installation 

```sh
pip install swiftrank
```

### Library Usage 🤗

- Create `Ranker` and `Tokenizer` instance.
  ```py
  from swiftrank import Ranker, Tokenizer
  ranker = Ranker(model_id="ms-marco-TinyBERT-L-2-v2")
  tokenizer = Tokenizer(model_id="ms-marco-TinyBERT-L-2-v2")
  ```

- Build a `ReRankPipeline` instance
  ```py
  from swiftrank import ReRankPipeline
  reranker = ReRankPipeline(ranker=ranker, tokenizer=tokenizer)
  ```

- Evaluate the pipeline
  ```py
  contexts = [
      "Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
      "LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
      "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second.  - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint.  - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
      "Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
      "vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels"
  ]
  for mapping in reranker.invoke(
      query="Tricks to accelerate LLM inference", contexts=contexts
  ):
      print(mapping)
  ```
  ```
  {'score': 0.9977508, 'context': 'Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.'}
  {'score': 0.9415497, 'context': "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second.  - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint.  - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run."}
  {'score': 0.47455463, 'context': 'vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels'}
  {'score': 0.43783104, 'context': 'LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper'}
  {'score': 0.043041725, 'context': 'Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.'}
  ```

- Want to filter contexts? Utilize `threshold` parameter.
  ```py
  for mapping in reranker.invoke(
      query="Tricks to accelerate LLM inference", contexts=contexts, threshold=0.8
  ):
      print(mapping)
  ```
  ```
  {'score': 0.9977508, 'context': 'Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.'}
  {'score': 0.9415497, 'context': "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second.  - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint.  - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run."}

  ```

- Have complex dictionary object? Utilize `key` parameter.
  ```py
  contexts = [
      {"id": 1, "content": "Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step."},
      {"id": 2, "content": "LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper"},
      {"id": 3, "content": "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second.  - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint.  - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run."},
      {"id": 4, "content": "Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup."},
      {"id": 5, "content": "vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels"}
  ]
  for mapping in reranker.invoke(
      query="Tricks to accelerate LLM inference", contexts=contexts, key=lambda x: x['content']
  ):
      print(mapping)
  ```
  ```
  {'score': 0.9977508, 'context': {'id': 1, 'content': 'Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.'}}
  {'score': 0.9415497, 'context': {'id': 3, 'content': "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second.  - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint.  - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run."}}
  {'score': 0.47455463, 'context': {'id': 5, 'content': 'vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels'}}
  {'score': 0.43783104, 'context': {'id': 2, 'content': 'LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper'}}
  {'score': 0.043041725, 'context': {'id': 4, 'content': 'Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.'}}
  ```

### CLI Usage 🤗

```
Usage: swiftrank COMMAND

Rerank contexts provided on stdin.

╭─ Session Parameters ───────────────────────────────────────────╮
│ *  --query      -q  query for reranking evaluation. [required] │
│    --threshold  -t  filter contexts using threshold.           │
│    --first      -f  get most relevant context.                 │
│    --version        Display application version.               │
│    --help       -h  Display this message and exit.             │
╰────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────╮
│ process  STDIN processor. [ json | jsonl | yaml ]              │
╰────────────────────────────────────────────────────────────────╯
```

- Print most relevant context
  ```sh
  cat files/contexts | swiftrank -q "Jujutsu Kaisen: Season 2" -f
  ```
  ```
  Jujutsu Kaisen 2nd Season
  ```

- Filtering using threshold
  > piping the output to `fzf` provides with a selection menu
  ```sh
  cat files/contexts | swiftrank -q "Jujutsu Kaisen: Season 2" -t 0.98 | fzf
  ```
  ```
  Jujutsu Kaisen 2nd Season
  Jujutsu Kaisen 2nd Season Recaps
  ```

- Using different model by setting `SWIFTRANK_MODEL` environment variable
  - Shell
    ```sh
    export SWIFTRANK_MODEL="ms-marco-MiniLM-L-12-v2"
    ```
  - Powershell
    ```powershell
    $env:SWIFTRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
    ```
  ```sh
  cat files/contexts | swiftrank -q "Jujutsu Kaisen: Season 2" 
  ```
  ```
  Jujutsu Kaisen 2nd Season
  Jujutsu Kaisen 2nd Season Recaps
  Jujutsu Kaisen
  Jujutsu Kaisen 0 Movie
  Jujutsu Kaisen Official PV
  Shingeki no Kyojin Season 2
  Shingeki no Kyojin Season 3 Part 2
  Shingeki no Kyojin Season 3
  Shingeki no Kyojin: The Final Season
  Kimi ni Todoke 2nd Season
  ```

#### Handling Complex Data

> Note: The schema closely resembles that of JQ, but employs a custom parser to avoid the hassle of installing JQ.

```
Usage: swiftrank process [OPTIONS]

STDIN processor. [ json | jsonl | yaml ]

╭─ Parameters ──────────────────────────────────────────────╮
│ --pre   -r  schema for pre-processing input.              │
│ --ctx   -c  schema for extracting context.                │
│ --post  -p  schema for extracting field after reranking.  │
╰───────────────────────────────────────────────────────────╯
```


- `json`
  ```sh
  cat files/contexts.json | swiftrank -q "Jujutsu Kaisen: Season 2" process -r ".categories[].items" -c '.name' -t 0.9  
  ```
  ```
  Jujutsu Kaisen 2nd Season
  Jujutsu Kaisen 2nd Season Recaps
  Jujutsu Kaisen
  Jujutsu Kaisen Official PV
  Jujutsu Kaisen 0 Movie
  ```

  > Provide one field for reranking and retrieve a different field as the output using `--post/-p` option
  ```sh
  cat files/contexts.json | swiftrank -q "Jujutsu Kaisen: Season 2" process -r ".categories[].items" -c '.name' -p '.url' -f
  ```
  ```
  https://myanimelist.net/anime/51009/Jujutsu_Kaisen_2nd_Season
  ```

- `yaml`
  ```sh
  cat files/contexts.yaml | swiftrank -q "Monogatari Series: Season 2" process -r ".categories[].items" -c '.name' -f
  ```
  ```
  Monogatari Series: Second Season
  ```

  > Provide one field for reranking and receive a different field as the output using `--post/-p` option
  ```sh
  cat files/contexts.yaml | swiftrank -q "Monogatari Series: Season 2" process -r ".categories[].items" -c '.name' -p '.payload.status' -f
  ```
  ```
  Finished Airing
  ```

> Json and Yaml lines doesn't require `--pre/-r` option, as they're by default loaded into an array object.

- `jsonlines`
  ```sh
  cat files/contexts.jsonl | swiftrank -q "Monogatari Series: Season 2" process -c '.name' -p '.payload.aired' -f
  ```
  ```
  Jul 7, 2013 to Dec 29, 2013
  ```

- `yamllines`
  ```sh
  cat files/contextlines.yaml | swiftrank -q "Monogatari Series: Season 2" process -c '.name' -f  
  ```
  ```
  Monogatari Series: Second Season
  ```

---

#### Acknowledgment of Original Repository

This project is derived from [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank), which is licensed under the Apache License 2.0. We extend our gratitude to the original authors and contributors for their work. The original repository provided a foundational framework for the development of our project, and we have built upon it with additional features and improvements.