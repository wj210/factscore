# factscore

Improved version FactScore from https://github.com/shmsw25/FActScore/tree/main

1. Updated the calling of chatgpt function to be compatible with both chat completion and regular prompt completion in the latest openai update. The `openai` version is `1.12.0`.
2. Batching is allowed by passing in `batch_size` in `get_score` function. The batch size is only used during the fact-checking portion and within each generation. Ie, if a generation is split into 20 facts and batch size is set to 8, it processes 8 in parallel, if the generation contains < batch size facts, it will be completed in one pass. Each generation is still processed sequentially. If you use Llama, batch size of 8 uses roughly 30+GB, but if ChatGPT is used, you can just increase the batch size to as many as you want without exceeding the API limits as multithreading is used to perform fast inference and does not require any RAM.
3. Multi-threading is also used for fact-generation, which makes it almost instantaneous.
4. `get_score` has the additional arguments: `n` which is the number of few-shot examples to use for atomic fact-generation, default is set to 7 as per the original, `batch_size` as explained in 2. `k` which is the number of passages to retrieve, set to default 5 as original. Lowering both `n` and `k` can reduce cost if required.
5. Fix the error in setting generation_length for fact-checking, which was not corrected in the original code, which causes the length to be 128 instead, if llama is used, this will be very slow, or waste cost if ChatGPT is used.

The remaining are the same, and the implementation is followed from the original github. Please cite the original authors similarly.
