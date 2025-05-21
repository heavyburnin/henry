from llama_cpp import Llama

class model_runner:
    def __init__(self, model_path="models/zephyr-7b-alpha.Q2_K.gguf"):
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)

    def infer(self, user_prompt):
        prompt = (
            "<|system|>\nYou are a helpful, honest, and intelligent assistant.\n"
            "<|user|>\n" + user_prompt + "\n"
            "<|assistant|>\n"
        )

        result = self.llm(prompt, max_tokens=256, stop=["<|user|>"], echo=False)
        return result["choices"][0]["text"].strip()

