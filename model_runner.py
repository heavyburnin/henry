from llama_cpp import Llama
import os
from functools import lru_cache
from threading import Lock

class model_runner:
    def __init__(self, model_path="models/zephyr-7b-alpha.Q4_K_M.gguf"):
        n_threads = os.cpu_count() or 4
        self.lock = Lock()

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=n_threads,
            n_batch=64,      # Batch size for internal eval speedup
            use_mlock=True,  # Optional: lock model in RAM to reduce page-outs
            verbose=False
        )

        # Warm-up call to reduce cold-start latency
        self.llm.create_completion(
            prompt="<|system|>\nHello\n<|user|>\nHi\n<|assistant|>\n",
            max_tokens=1
        )

    @lru_cache(maxsize=128)
    def infer(self, user_prompt: str) -> str:
        prompt = (
            "<|system|>\nYou are a helpful, honest, and intelligent assistant.\n"
            f"<|user|>\n{user_prompt}\n"
            "<|assistant|>\n"
        )

        with self.lock:
            result = self.llm.create_completion(
                prompt=prompt,
                max_tokens=256,
                stop=["<|user|>"],
                temperature=0.7
            )

        return result["choices"][0]["text"].strip()

