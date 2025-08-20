# Language-helper

A language learning assistant that provides spell checking, example sentences, and additional information about words using local LLMs and optional RAG (Retrieval-Augmented Generation).

## Features

- **Spell Checking**: Automatically detects and corrects spelling errors
- **Example Sentences**: Generates 3 natural example sentences for each word
- **Additional Information**: Provides definitions, usage examples, and collocations
- **RAG Support**: Optional retrieval-augmented generation for enhanced information
- **Multi-language Support**: Works with English, Ukrainian, Polish, and other languages

## Installation

1. Install dependencies:
```bash
poetry install
```

2. Install and start Ollama:
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:3b-instruct
```

## Usage

### Basic Usage
```bash
poetry run python -m app "word" -l en
```

### With RAG (Retrieval-Augmented Generation)
```bash
poetry run python -m app "word" -l en --rag --k 3
```

### Force Disable RAG (if experiencing crashes)
```bash
poetry run python -m app "word" -l en --no-rag
```

### Available Options
- `-l, --lang`: Input language (default: en)
- `-m, --model`: Ollama model to use (default: qwen2.5:3b-instruct)
- `--rag`: Enable RAG for enhanced information
- `--no-rag`: Force disable RAG (useful if RAG causes crashes)
- `--k`: Number of RAG chunks to retrieve (default: 3)
- `--stop-after`: Stop the Ollama model after completion

## GUI Interface

### Desktop GUI with Hotkey Support
```bash
poetry run python -m app.gui.hotkey_gui
```

**Features:**
- **Global Hotkey**: `Ctrl+Option+L` to show/hide the GUI instantly
- **Desktop Integration**: Native desktop application
- **Quick Access**: Perfect for quick word lookups while working
- **Comprehensive Features**: Spell checking, examples, and word information
- **Non-blocking UI**: Background processing with threading
- **Stable Operation**: RAG disabled by default to prevent crashes

**Setup on macOS:**
1. **Install tkinter** (if not available):
   ```bash
   brew install python-tk
   ```
   Or use the system Python that includes tkinter by default.

2. **Grant Accessibility permissions** to Terminal/iTerm in System Settings → Privacy & Security → Accessibility
3. **Run the GUI application**: `poetry run python -m app.gui.hotkey_gui`
4. **Use `Ctrl+Option+L`** to toggle the interface

**Important Notes:**
- **RAG is disabled by default** to prevent segmentation faults on macOS with Python 3.13
- **GUI will auto-disable RAG** if it crashes once
- **Use CLI with `--no-rag`** for the most stable experience
- **If hotkey doesn't work**: Check accessibility permissions

### Streamlit Web Interface
```bash
poetry run streamlit run app/ui/app.py
```

**Features:**
- Web-based interface accessible via browser
- Translation-focused functionality
- Easy to deploy and share

## Known Issues

### RAG Segmentation Fault on macOS with Python 3.13

**Issue**: The RAG functionality may cause segmentation faults on macOS when using Python 3.13 due to compatibility issues with the sentence-transformers library.

**Symptoms**: 
- Application crashes with "segmentation fault" error
- Multiprocessing resource tracker warnings
- GUI crashes when RAG is enabled

**Solutions**: 
1. **CLI with RAG disabled** (recommended):
   ```bash
   poetry run python -m app "word" -l en --no-rag
   ```

2. **GUI with RAG disabled** (default):
   ```bash
   poetry run python -m app.gui.hotkey_gui
   ```
   RAG is disabled by default in the GUI to prevent crashes.

3. **Streamlit interface** (stable):
   ```bash
   poetry run streamlit run app/ui/app.py
   ```

**Workaround**: The application automatically falls back to simple LLM responses when RAG is disabled, still providing comprehensive word information including definitions, usage examples, and collocations.

## Current Status

### ✅ **Project is Fully Functional Without RAG**

The language helper project works perfectly without RAG functionality, providing:
- **Spell checking and correction**
- **Example sentence generation**
- **Word definitions and usage**
- **Collocations and language patterns**
- **Multi-language support**

All three interfaces (CLI, GUI, Streamlit) are stable and ready for use.

## Example Output
Checking: commend  | language: en  | model: qwen2.5:3b-instruct  | RAG: False
✅ Word does not need modification: «commend»

Examples:
1. I commend your dedication to the project.
2. Your efforts commendably exceeded expectations.
3. Commendation is due for your outstanding performance.

Additional information about «commend»:
```markdown
- **Definition**: To praise or recommend someone or something highly.

- **Common Usages**:
  - Praising an individual's work: The jury commended her dedication to her research.
  - Recommending a product/service: The hotel is well-known for its excellent service.

- **Collocations**:
  - Commend (someone) for something
  - Commend (something) as the best
```

Sentence verification:
  1. ✅ Contains «commend»
  2. ✅ Contains «commend»
  3. ✅ Contains «commend»
```

## Development

### Project Structure
```
Language-helper/
├── app/
│   ├── __main__.py      # Main CLI entry point
│   ├── core/
│   │   ├── llm.py       # LLM integration
│   │   ├── rag.py       # RAG functionality
│   │   └── cache.py     # Caching utilities
│   └── ui/
│       └── app.py       # Streamlit UI
├── data/                # Data files
├── docs/               # Documentation and RAG index
└── tests/              # Test files
```

### Running Tests
```bash
poetry run pytest
```

## License

This project is licensed under the MIT License.