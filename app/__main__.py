# app/__main__.py
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes the warning

import argparse, json, re, subprocess
from typing import Dict, Any

def _contains_whole_word(s: str, w: str) -> bool:
    return re.search(rf"(?i)\b{re.escape(w)}\b", s) is not None

from app.core.llm import call_llm
from app.core.rag import ask_with_rag_def

DEFAULT_MODEL = "qwen2.5:3b-instruct"  # lighter and more stable for 8 GB

# --- strict spell-check + 3 examples (JSON response) ---
SPELL_SYSTEM = (
    "You are a precise spell checker and example-sentence generator.\n"
    "Given ONE input word and its language, decide if it is correctly spelled.\n"
    "If correct: keep it as is. If misspelled: correct ONLY the spelling of the same intended lemma.\n"
    "Return STRICT JSON: {\"is_correct\": bool, \"input\": str, \"final\": str, \"sentences\": [str, str, str]}\n"
    "- final = input if correct; else final = corrected spelling\n"
    "- sentences: exactly 3 short, natural sentences in the SAME language as the input; each MUST contain the final word\n"
    "- No commentary/markdown outside JSON."
)
SPELL_USER_TMPL = "language: {lang}\nword: {word}\nRespond with JSON only."

def _extract_json(s: str) -> Dict[str, Any]:
    """Extracts the first JSON block and validates the schema."""
    m = re.search(r"\{.*\}", s, flags=re.S)
    data = json.loads(m.group(0) if m else s)
    assert isinstance(data.get("is_correct"), bool)
    assert isinstance(data.get("input"), str)
    assert isinstance(data.get("final"), str)
    ss = data.get("sentences")
    assert isinstance(ss, list) and len(ss) == 3 and all(isinstance(x, str) for x in ss)
    return data

def check_spelling_and_examples(word: str, lang: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    prompt = SPELL_USER_TMPL.format(lang=lang, word=word.strip())
    # economical options for Air M2 8 GB (without Metal)
    opts = {"num_ctx": 256, "num_predict": 180, "temperature": 0.2, "num_thread": 4, "num_gpu": 0}
    raw = call_llm(prompt, model=model, system=SPELL_SYSTEM, options=opts)
    return _extract_json(raw)

def main():
    ap = argparse.ArgumentParser(description="Spell-check (+ 3 examples) + optional RAG info")
    ap.add_argument("word", help="Word to check")
    ap.add_argument("-l", "--lang", default="en", help="Input language (e.g., en/uk/pl)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Ollama model tag")
    ap.add_argument("--rag", action="store_true", help="Use local RAG for extra info")
    ap.add_argument("--k", type=int, default=3, help="Top-k RAG chunks")
    ap.add_argument("--stop-after", action="store_true", help="Stop the model in Ollama after run")
    args = ap.parse_args()

    print(f"Checking: {args.word}  | language: {args.lang}  | model: {args.model}  | RAG: {args.rag}")

    try:
        # 1) Spelling check + 3 examples (always)
        data = check_spelling_and_examples(args.word, lang=args.lang, model=args.model)
        if data["is_correct"]:
            print(f"✅ Word does not need modification: «{data['input']}»")
        else:
            print(f"✍️ Corrected: «{data['input']}» → «{data['final']}»")

        # 1.5) Post-check sentences (check if they contain the word as a whole word)
        final_w = data["final"].strip()
        bad = [i for i,s in enumerate(data["sentences"]) if not _contains_whole_word(s, final_w)]
        if bad:
            print(f"⚠️  Sentences {[i+1 for i in bad]} do not contain the word «{final_w}» as a whole word. Regenerating...")
            # regenerate only sentences (light "repair" request)
            repair_system = 'Return STRICT JSON: {"sentences": [str, str, str]} — exactly 3 short sentences, each must contain the word "{w}". No commentary.'
            repair_user = f'word: {final_w}'
            raw = call_llm(repair_user, model=args.model,
                           system=repair_system.format(w=final_w),
                           options={"num_ctx": 128, "num_predict": 120, "temperature": 0.2, "num_gpu": 0})
            try:
                m = re.search(r"\{.*\}", raw, flags=re.S)
                data["sentences"] = json.loads(m.group(0))["sentences"]
                print("✅ Sentences regenerated")
            except Exception:
                print("❌ Failed to regenerate sentences")

        print("\nExamples:")
        for i, s in enumerate(data["sentences"], 1):
            print(f"{i}. {s}")

        # 2) Additional information (RAG or simple LLM) — take the final word
        final_word = data["final"].strip()
        print(f"\nAdditional information about «{final_word}»:")
        if args.rag:
            answer = ask_with_rag_def(final_word, k=args.k, model=args.model, max_context_chars=800)
        else:
            prompt = (
                f"Give a concise definition, common usages, and 2 collocations for the word “{final_word}”. "
                f"Structure in markdown."
            )
            answer = call_llm(prompt, model=args.model)
        print(answer)
        
        # 3) Small post-check of sentences (check if they contain the word)
        print(f"\nSentence verification:")
        for i, sentence in enumerate(data["sentences"], 1):
            if final_word.lower() in sentence.lower():
                print(f"  {i}. ✅ Contains «{final_word}»")
            else:
                print(f"  {i}. ❌ Does NOT contain «{final_word}»")

    finally:
        # optionally — free RAM after run
        if args.stop_after:
            try:
                subprocess.run(["ollama", "stop", args.model], check=False)
            except Exception:
                pass

if __name__ == "__main__":
    main()
