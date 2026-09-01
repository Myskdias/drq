import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
from openai import AsyncOpenAI
import openai
import backoff

# MAX_NUM_TOKENS = 4096

class GPT:
    def __init__(self, model, system_prompt, temperature=1., seed=0, base_url=None):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.seed = seed

        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is not None and api_key is None:
            api_key = "local"
        client_kwargs = dict(api_key=api_key)
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError, openai.PermissionDeniedError))
    async def get_completion_async(self, prompt, n_responses=1, seed=None):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            # max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=self.seed if seed is None else seed,
        )
        return [r.message.content for r in response.choices]

    async def get_multiple_completions_async(self, prompts, n_responses=1, seed=None):
        base_seed = self.seed if seed is None else seed
        results = await asyncio.gather(*(
            self.get_completion_async(prompt, n_responses, seed=base_seed + i)
            for i, prompt in enumerate(prompts)
        ))
        return results
    
    def get_completion(self, prompt, n_responses=1, seed=None):
        return asyncio.run(self.get_completion_async(prompt, n_responses, seed=seed))

    def get_multiple_completions(self, prompts, n_responses=1, seed=None):
        return asyncio.run(self.get_multiple_completions_async(prompts, n_responses, seed=seed))
        
if __name__ == "__main__":
    gpt = GPT(model="gpt-4o-mini", system_prompt="You are a helpful assistant.")
    prompts = ["Hello", "Tell me a joke", "What's 2+2?", "What's the capital of France?"]

    for p in prompts:
        c = gpt.get_completion(p)
        print(c)

    results = gpt.get_multiple_completions(prompts, n_responses=2)
    print(results)
    
