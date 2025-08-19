# app/core/llm.py
import ollama

DEFAULT_OPTIONS = {"num_ctx": 256, "num_predict": 120, "temperature": 0.2}

def call_llm(prompt: str,
             model: str = "qwen2.5:3b-instruct",
             system: str = "You are a helpful linguist assistant. Write clearly and concisely.",
             options: dict | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    resp = ollama.chat(model=model, messages=messages, options=opts)
    return resp["message"]["content"]


def translate_word(word: str, from_lang: str = "EN", to_lang: str = "UK"):
    """
    Translate a word using Ollama with specific translation prompt.
    
    Args:
        word: The word to translate
        from_lang: Source language (default: "EN")
        to_lang: Target language (default: "UK")
    
    Returns:
        The translated word/content
    """
    prompt = f"Word: {word}. {from_lang}->{to_lang}"
    resp = ollama.chat(
        model="qwen2.5:3b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 256, "num_predict": 120, "temperature": 0.2}
    )
    return resp["message"]["content"]

# Example usage (uncomment to test):
# if __name__ == "__main__":
#     result = translate_word("serendipity")
#     print(result)

