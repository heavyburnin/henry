from llama_cpp import Llama

class HermesRunner:
    def __init__(self, model_path="models/tinyllama-1.1b-chat-v0.3.Q2_K.gguf"):
        print("🔄 Loading TinyLLaMA model...")
        self.llm = Llama(model_path=model_path, n_ctx=1024)
        print("✅ Model loaded.")

    def infer(self, prompt):
        print("🧠 Prompt:", prompt)
        result = self.llm(prompt, max_tokens=128, stop=["</s>"], echo=False)
        print("✅ Response generated.")
        return result["choices"][0]["text"].strip()

    def tokenize(self, text):
        return self.llm.tokenize(text.encode("utf-8"))

    def detokenize(self, tokens):
        return self.llm.detokenize(tokens).decode("utf-8")

