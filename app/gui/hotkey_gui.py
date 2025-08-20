from __future__ import annotations
import os, re, json, subprocess, threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# приглушити попередження HF
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# обмежити потоки (щоб MAC AIR 8GB не «висів»)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from pynput import keyboard  # глобальна гаряча клавіша
from app.core.llm import call_llm

# Import with fallback for RAG
try:
    from app.core.rag import ask_with_rag_def as ask_rag
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False
    ask_rag = None

DEFAULT_MODEL = "qwen2.5:3b-instruct"

# Use the same system prompt as the CLI for consistency
SPELL_SYSTEM = (
    "You are a precise spell checker and example-sentence generator.\n"
    "Given ONE input word and its language, decide if it is correctly spelled.\n"
    "If correct: keep it as is. If misspelled: correct ONLY the spelling of the same intended lemma.\n"
    "Return STRICT JSON: {\"is_correct\": bool, \"input\": str, \"final\": str, \"sentences\": [str, str, str]}\n"
    "- final = input if correct; else final = corrected spelling\n"
    "- sentences: exactly 3 short, natural sentences in the SAME language as the input; each MUST contain the final word\n"
    "- No commentary/markdown outside JSON."
)

def contains_whole_word(s: str, w: str) -> bool:
    return re.search(rf"(?i)\b{re.escape(w)}\b", s) is not None

def extract_json(s: str) -> dict:
    """Extracts the first JSON block and validates the schema."""
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response")
    
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}")
    
    if not isinstance(data, dict):
        raise ValueError("Response is not a JSON object")
    
    if not isinstance(data.get("is_correct"), bool):
        raise ValueError("Missing or invalid 'is_correct' field")
    
    if not isinstance(data.get("input"), str):
        raise ValueError("Missing or invalid 'input' field")
    
    if not isinstance(data.get("final"), str):
        raise ValueError("Missing or invalid 'final' field")
    
    ss = data.get("sentences")
    if not isinstance(ss, list) or len(ss) != 3 or not all(isinstance(x, str) for x in ss):
        raise ValueError("Missing or invalid 'sentences' field (must be list of 3 strings)")
    
    return data

def spell_check(word: str, lang: str, model: str) -> dict:
    user = f"language: {lang}\nword: {word}\nRespond with JSON only."
    opts = {"num_ctx": 256, "num_predict": 180, "temperature": 0.2, "num_thread": 4, "num_gpu": 0}
    raw = call_llm(user, model=model, system=SPELL_SYSTEM, options=opts)
    return extract_json(raw)

def repair_sentences(final_word: str, model: str) -> list[str] | None:
    system = f'Return STRICT JSON: {{"sentences": [str, str, str]}} — exactly 3 short sentences, each must contain the word "{final_word}". No commentary.'
    user = f'word: {final_word}'
    opts = {"num_ctx": 128, "num_predict": 120, "temperature": 0.2, "num_gpu": 0}
    raw = call_llm(user, model=model, system=system, options=opts)
    try:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            repair_data = json.loads(m.group(0))
            if isinstance(repair_data, dict) and "sentences" in repair_data:
                sentences = repair_data["sentences"]
                if isinstance(sentences, list) and len(sentences) == 3 and all(isinstance(x, str) for x in sentences):
                    return [s.strip() for s in sentences]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None

# ---------------- GUI ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Language Helper — mini GUI")
        self.geometry("640x520")
        self._build_ui()
        self.withdraw()  # стартуємо захованим

    def _build_ui(self):
        pad = {"padx": 6, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)
        
        # Add warning for RAG on macOS
        if os.name == 'posix' and hasattr(os, 'uname') and 'Darwin' in os.uname().sysname:
            warning_label = ttk.Label(frm, text="⚠️ RAG may crash on macOS with Python 3.13", foreground="orange")
            warning_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        ttk.Label(frm, text="Word:").grid(row=1, column=0, sticky="w")
        self.ent_word = ttk.Entry(frm)
        self.ent_word.grid(row=1, column=1, columnspan=3, sticky="ew", **pad)
        frm.columnconfigure(3, weight=1)

        self.lang = tk.StringVar(value="en")
        ttk.Label(frm, text="Lang:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.lang, values=["en","uk","pl"], width=6, state="readonly").grid(row=2, column=1, sticky="w", **pad)

        self.model = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Label(frm, text="Model:").grid(row=2, column=2, sticky="e")
        ttk.Combobox(frm, textvariable=self.model, values=["qwen2.5:3b-instruct","qwen2.5:7b-instruct"], width=22, state="readonly").grid(row=2, column=3, sticky="ew", **pad)

        self.use_rag = tk.BooleanVar(value=False)  # Disable RAG by default to prevent crashes
        ttk.Checkbutton(frm, text="Use RAG", variable=self.use_rag).grid(row=3, column=0, sticky="w", **pad)

        self.k_var = tk.IntVar(value=3)
        ttk.Label(frm, text="k:").grid(row=3, column=1, sticky="e")
        ttk.Spinbox(frm, from_=1, to=6, textvariable=self.k_var, width=5).grid(row=3, column=2, sticky="w", **pad)

        self.max_ctx = tk.IntVar(value=800)
        ttk.Label(frm, text="ctx chars:").grid(row=3, column=3, sticky="e")
        ttk.Spinbox(frm, from_=400, to=1600, increment=100, textvariable=self.max_ctx, width=8).grid(row=3, column=3, sticky="w", padx=(80,6))

        btns = ttk.Frame(self); btns.pack(fill="x", **pad)
        self.btn_run = ttk.Button(btns, text="Check (Enter)", command=self.on_run)
        self.btn_run.pack(side="left")
        ttk.Button(btns, text="Stop model", command=self.stop_model).pack(side="left", padx=6)
        ttk.Button(btns, text="Hide (Esc)", command=self.hide).pack(side="left", padx=6)

        self.out = scrolledtext.ScrolledText(self, height=20)
        self.out.pack(fill="both", expand=True, **pad)
        self.bind("<Return>", lambda e: self.on_run())
        self.bind("<Escape>", lambda e: self.hide())

    def show(self):
        self.deiconify()
        self.lift()
        self.focus_force()
        self.ent_word.focus_set()

    def hide(self):
        self.withdraw()

    def log(self, text: str):
        self.out.insert("end", text + "\n")
        self.out.see("end")

    def clear(self):
        self.out.delete("1.0", "end")

    def stop_model(self):
        try:
            subprocess.run(["ollama", "stop", self.model.get()], check=False)
            self.log("🧹 Model stopped (ollama).")
        except Exception as e:
            messagebox.showerror("Error", f"ollama stop failed: {e}")

    def on_run(self):
        word = self.ent_word.get().strip()
        if not word:
            return
        self.btn_run.config(state="disabled")
        self.clear()
        self.log(f"Checking: {word} | lang={self.lang.get()} | model={self.model.get()} | RAG={self.use_rag.get()}")
        threading.Thread(target=self._run_task, args=(word,), daemon=True).start()

    def _run_task(self, word: str):
        try:
            self._ui(lambda: self.log(f"🔄 Processing: {word}..."))
            
            # Spell check
            data = spell_check(word, self.lang.get(), self.model.get())
            final_word = data["final"].strip()
            if data["is_correct"]:
                status = f"✅ Word does not need modification: «{data['input']}»"
            else:
                status = f"✍️ Fixed: «{data['input']}» → «{final_word}»"

            # Check sentence integrity
            bad = [i for i,s in enumerate(data["sentences"]) if not contains_whole_word(s, final_word)]
            if bad:
                self._ui(lambda: self.log(f"⚠️ Sentences {[i+1 for i in bad]} do not contain «{final_word}». Regenerating..."))
                rep = repair_sentences(final_word, self.model.get())
                if rep:
                    data["sentences"] = rep

            # Output results
            self._ui(lambda: self.log(status))
            self._ui(lambda: self.log("\nExamples:"))
            for i,s in enumerate(data["sentences"],1):
                self._ui(lambda s=s: self.log(f"{i}. {s}"))

            # Additional information
            self._ui(lambda: self.log("\n---\nExtra info:"))
            if self.use_rag.get():
                try:
                    if RAG_AVAILABLE and ask_rag:
                        self._ui(lambda: self.log("🔄 Using RAG (may take a moment)..."))
                        ans = ask_rag(final_word, k=self.k_var.get(), model=self.model.get(), max_context_chars=self.max_ctx.get())
                    else:
                        raise Exception("RAG not available")
                except Exception as e:
                    self._ui(lambda: self.log(f"❌ RAG failed: {e}"))
                    self._ui(lambda: self.log("Falling back to simple LLM response..."))
                    self.use_rag.set(False)  # Disable RAG for future use
                    prompt = (f"Give a concise definition, common usages, and 2 collocations for the word \"{final_word}\". "
                              f"Structure in markdown.")
                    ans = call_llm(prompt, model=self.model.get())
            else:
                prompt = (f"Give a concise definition, common usages, and 2 collocations for the word \"{final_word}\". "
                          f"Structure in markdown.")
                ans = call_llm(prompt, model=self.model.get())
            self._ui(lambda: self.log(ans))

        except Exception as e:
            self._ui(lambda: self.log(f"❌ Error: {e}"))
            self._ui(lambda: self.log("💡 Tip: Try disabling RAG if you experience crashes"))
        finally:
            self._ui(lambda: self.btn_run.config(state="normal"))

    def _ui(self, fn):  # безпечно оновлюємо UI з бекґраунду
        self.after(0, fn)

# ---------- глобальна гаряча клавіша ----------
def start_hotkey(app: App):
    # Ctrl+Option+L → показати/сховати
    def toggle():
        if app.state() == "withdrawn":
            app.show()
        else:
            app.hide()
    hk = keyboard.GlobalHotKeys({ "<ctrl>+<alt>+l": toggle })
    hk.start()
    return hk

if __name__ == "__main__":
    # НА MAC: надай «Accessibility» для Terminal/iTerm (System Settings → Privacy & Security → Accessibility)
    app = App()
    hk = start_hotkey(app)
    app.mainloop()
