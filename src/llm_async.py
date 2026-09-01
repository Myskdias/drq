import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
from openai import AsyncOpenAI
import openai
import backoff

# MAX_NUM_TOKENS = 4096

class GPT:
    def __init__(self, model, system_prompt, temperature=1., seed=0, base_url=None,
                 backend="openai", max_new_tokens=4096):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.seed = seed
        self.backend = backend
        self.max_new_tokens = max_new_tokens

        if backend == "huggingface":
            self._init_huggingface()
            return
        if backend != "openai":
            raise ValueError(f"Unsupported LLM backend: {backend}")

        self._init_openai(base_url)

    def _init_openai(self, base_url):
        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is not None and api_key is None:
            api_key = "local"
        client_kwargs = dict(api_key=api_key)
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)

    def _init_huggingface(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("The Hugging Face backend requires an available CUDA GPU.")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            local_files_only=True,
        )
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.bfloat16,
            device_map=0,
            local_files_only=True,
        )
        self.hf_model.eval()

    def _get_huggingface_completion(self, prompt, n_responses, seed):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.hf_model.device)
        input_length = inputs["input_ids"].shape[-1]

        generation_kwargs = {
            "do_sample": self.temperature > 0,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        responses = []
        with self.torch.random.fork_rng(devices=[self.hf_model.device.index]):
            self.torch.manual_seed(seed)
            for _ in range(n_responses):
                with self.torch.inference_mode():
                    output = self.hf_model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
                responses.append(self.tokenizer.decode(
                    output[0, input_length:],
                    skip_special_tokens=True,
                ))
        return responses

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError, openai.PermissionDeniedError))
    async def get_completion_async(self, prompt, n_responses=1, seed=None):
        request_seed = self.seed if seed is None else seed
        if self.backend == "huggingface":
            return self._get_huggingface_completion(prompt, n_responses, request_seed)

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
            seed=request_seed,
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
    
