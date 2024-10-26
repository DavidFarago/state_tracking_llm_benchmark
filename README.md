# state_tracking_llm_benchmark

Benchmark to test state tracking capabilities of LLMs (via permutation group S_5).

## Details

The paper [The Illusion of State in State-Space Models](https://arxiv.org/pdf/2404.08819) has shown that 
even the simple tracking of the order of 5 elements through a sequence of swaps (state tracking in the permutation group $S_5$) 
cannot be handled by a fixed size transformer/S4/S6 model. 
Instead, the model must be larger with longer swap sequences to handle the state tracking.

The paper [The Expressive Power of Transformers with Chain of Thought](https://arxiv.org/pdf/2310.07923), however, 
has shown that state-tracking and reasoning improves with CoT / scratchpad, 
but that the size of CoT / scratchpad is relevant for determining the expressiveness. 
So you can trade-off model-size for CoT size, i.e. context length (and correspondingly runtime) for state tracking tasks.

Besides state tracking resp. CoT, the model must also have sufficient NLU resp. instruction following capabilities 
to be able to understand resp. follow the task of this benchmark.

## Results

| MODEL_NAME                 | max correct swaps | comments             |
| -------------------------- | ----------------- | -------------------- |
| gemma2-9b-it               | 0                 | served at groq       |
| llama-3.1-70b-versatile    | 2                 | served at groq       |
| llama-3.2-11b-text-preview | 2                 | served at groq       |
| lfm-40b                    | 0                 | Liquid AI's LFM 40.3B MoE served at lambdalab  |
| Qwen2.5-72B-Instruct       | 2                 | served at hyperbolic |
| gpt-4o-mini                | 2                 | served at OpenAI     |
| gpt-4o                     | 3                 | served at OpenAI     |
| o1-mini                    | 100               | served at OpenAI, expensive and slow|
| o1-preview                 | 101               | served at OpenAI, expensive and slow|
