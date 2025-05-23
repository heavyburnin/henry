from llama_cpp import Llama
import os
from functools import lru_cache
from threading import Lock
import spacy

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model for the spaCy 'en_core_web_sm'...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class model_runner:
    def __init__(self, model_path="models/zephyr-7b-alpha.Q4_K_M.gguf"):
        n_threads = os.cpu_count() or 4
        self.lock = Lock()

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=n_threads,
            n_batch=512,
            use_mlock=True,
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

    def infer_stream(self, user_prompt: str):
        prompt = (
            "<|system|>\nYou are a helpful, honest, and intelligent assistant.\n"
            f"<|user|>\n{user_prompt}\n"
            "<|assistant|>\n"
        )

        buffer = ""
        first_token = True
        with self.lock:
            for token in self.llm.create_completion(
                prompt=prompt,
                max_tokens=256,
                stop=["<|user|>"],
                temperature=0.7,
                stream=True
            ):
                buffer += token["choices"][0]["text"]

                # Process the buffer using spaCy
                doc = nlp(buffer)
                sentences = [sent.text for sent in doc.sents]

                if sentences:
                    last_sentence = sentences[-1]
                    if last_sentence and last_sentence[-1] in ".!?":
                        # If the last sentence ends with punctuation, yield it
                        yield buffer.strip()
                        buffer = ""
                    else:
                        # If not, yield the text without the last incomplete sentence
                        if len(sentences) > 1:
                            yield " ".join(sentences[:-1]).strip()
                else:
                    # If no sentences are detected, yield the buffer as is
                    if first_token:
                        first_token = False
                        yield buffer.strip().capitalize()
                    else:
                        yield buffer.strip()

            # Yield any remaining text
            if buffer.strip():
                yield buffer.strip()
