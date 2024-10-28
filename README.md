# state_tracking_llm_benchmark

Benchmark to test state tracking capabilities of LLMs (via permutation group S_5).

## Motivation

The paper [The Illusion of State in State-Space Models](https://arxiv.org/pdf/2404.08819) has shown that 
even the simple tracking of the order of 5 elements through a sequence of swaps (state tracking in the permutation group $S_5$) 
cannot be handled by a fixed size transformer/S4/S6 model. 
Instead, the model must be larger with longer swap sequences to handle the state tracking.

The paper [The Expressive Power of Transformers with Chain of Thought](https://arxiv.org/pdf/2310.07923), however, 
has shown that state-tracking and reasoning improves with CoT / scratchpad, 
but that the size of CoT / scratchpad is relevant for determining the expressiveness. 
So you can trade-off model-size for CoT size, i.e. context length (and correspondingly runtime) for state tracking tasks.

## Results

| MODEL_NAME                 | max correct swaps |max correct swaps with CoT| infrastructure       |
| -------------------------- | ----------------- | -------------------------|--------------------- |
| gemma2-9b-it               | 0                 | 2 (144 tokens, 0.6')   |served at groq       |
| llama-3.1-70b-versatile    | 2                 | 2 (140 tokens, 0.9')   |served at groq       |
| llama-3.2-11b-text-preview | 2                 | 2 (152 tokens, 0.7')   |served at groq       |
| lfm-40b                    | 0                 | 2 (53 tokens, 1.1')    |Liquid AI's LFM 40.3B MoE served at lambdalab  |
| jamba-1.5-large            | 1                 | 2 (242 tokens, 4.9')   |Mamba served at a21  |
| Qwen2.5-72B-Instruct       | 2                 | 2 (4.2')               |served at hyperbolic |
| gpt-4o-mini                | 2                 | 11 (1164 tokens, 17.2')|served at OpenAI     |
| gpt-4o                     | 3                 | 2 (99 tokens, 2.1')    |served at OpenAI     |
| o1-mini                    | 100 (15k tokens, 90')| 98 (12970 tokens, 73.9')|served at OpenAI|
| o1-preview                 | 101 (9k tokens, 60') | 74 (12562 tokens, 94.4')|served at OpenAI|

## Error analyses

Besides state tracking resp. CoT, the model must also have sufficient NLU resp. instruction following capabilities 
to be able to understand resp. follow the task of this benchmark.

One of the main root causes is [Lost In The Middle](https://arxiv.org/abs/2307.03172): For instance, with GPT-4o-mini and activated CoT for given 51 swaps, the CoT shows that given swap 49 is left out. When increasing the input further, Lost In The Middle can become more severe: For given 101 swaps, the CoT tracks the state correctly for the first 46 swaps, but the following 52 given swaps are replaced by hallucinated 25 swaps (leaving out some of those 52 given swaps never leads to those hallucinated 25 swaps), with the last 3 swaps in the CoT being also the last 3 given swaps. However, Lost In The Middle is not always in the Middle: For given 16 swaps, the CoT left out the last swap. Interestingly, within the next 5 problem instances (given 17 swaps up to given 21 swaps), three times swaps are lost, and always exactly one swap: swap number 16. Furthermore, Lost In The Middle does not monotonically get more severe: For instance, given 31 swaps has stronger Lost In The Middle compared to 51 swaps: the given swap 16 plus the given swaps 25 to 31 are left out. Lost swaps cannot be suppresed by adding `Never miss any given swap.` to the instruction.

However, often the left out swaps make no difference because the CoT makes a swapping mistake earlier, often mixing up balls and slots, e.g. for given 31 swaps already at the 10th swap:
```
   - Output: `5, 3, 2, 1, 4`
10. Swap ball 5 and ball 4:
    - Slot 1: 5
    - Slot 2: 3
    - Slot 3: 2
    - Slot 4: 4
    - Slot 5: 1
    - Output: `5, 3, 2, 4, 1`
```
The same error occurs for given 13 swaps. However, it does not occur for given 11 swaps (where it reproducibly computes the correct permutation, even though the CoT is different on different runs -- despite temperature 0).

Sometimes, only a single ball is mixed up with a slot, e.g. for gpt-4o with activated CoT for given 3 swaps:
```
   - New order: 5, 3, 2, 4, 1
4. Swap ball 2 and ball 5:
   - Ball 2 moves to slot 5, and ball 1 moves to slot 3.
   - New order: 5, 3, 1, 4, 2

```
These mixups cannot be suppresed by adding `Never mix up balls and slots.` to the instruction.

Unfortunately, there are even worse CoT mistakes, e.g. for GPT-4o-mini with activated CoT for 19 given swaps, ball 3 is replaced by a second ball 1, upon which is gets even more confused:
```
...
   - **State:** `2, 3, 5, 1, 4`
7. **Swap ball 1 and ball 2:**
   - Slot 1: Ball 2
   - Slot 2: Ball 1
   - **State:** `1, 2, 5, 1, 4`
8. **Swap ball 2 and ball 1:**
   - No change (same balls swapped).
   - **State:** `1, 2, 5, 1, 4`
9. **Swap ball 5 and ball 2:**
   - Slot 2: Ball 5
   - Slot 5: Ball 2
   - **State:** `1, 5, 2, 1, 4`
```
This kind of error also happens to the Liquid S4 MoE lfm-40b with CoT with :
```
...
5, 2, 3, 4, 1
2. Swap ball 3 and ball 2:
5, 2, 2, 4, 1
```

The long tails contains even stranger errors:
For gemma2-9b-it with CoT for 3 given swaps, ball 2 is misinterpreted as ball 22:
```
* **Swap 2 (ball 3 and ball 2):** 5, 3, 2, 4, 1
* **Swap 3 (ball 22 and ball 5):**  This swap is impossible since there's no ball 22.
```
For o1-preview with CoT for 100 given swaps, at the 29th swap more than 2 balls change places:
```
4, 3, 2, 1, 5
1, 4, 2, 3, 5
```
For o1-preview with CoT for 83 given swaps, at the 4th swap all balls change places:
```
2, 3, 5, 4, 1
5, 1, 2, 3, 4
```
For o1-preview with CoT for 80 given swaps, at swap 79, no balls change places:
```
    1, 4, 5, 3, 2
79. Swap ball 1 and ball 2:
    1, 4, 5, 3, 2
```
For o1-preview with CoT for 80 given swaps, there was one time the following answer after 9 swaps in the CoT (you cannot set temperature to other values than 1.0 for 01 models):
````
*(Continue the process similarly for each of the swaps, updating the `ball_in_slot` list and outputting the positions after each swap.)*

**Final Swap: Swap ball 5 and ball 2**
- **Before Swap:**
  - *(Assuming we've updated `ball_in_slot` appropriately up to this point)*
- **Swap Positions:**
  - Swap positions of ball 5 and ball 2.
- **After Swap:**
  - *(Final positions after the last swap)*
- **Output:**
  ```
  *(Final positions of balls in slots)*
  ```
````

## Conclusion

The theory of [The Illusion of State in State-Space Models](https://arxiv.org/pdf/2404.08819) is not reflected well in the benchmark results. This is due to the high number of error causes besides state tracking: Lost In The Middle causes swaps to be left out, weak instruction following causes slots and balls to be mixed up. Trying to forbid these kind of mistakes through prompts has little effect.

The theory of [The Expressive Power of Transformers with Chain of Thought](https://arxiv.org/pdf/2310.07923) is reflected in the internal CoT of the o1 models, but not in explicit CoTs. Consequently, explicit CoT makes o1 perform worse, and less deterministic/robust.
