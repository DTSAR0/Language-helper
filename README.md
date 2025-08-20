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

## Known Issues

### RAG Segmentation Fault on macOS with Python 3.13

**Issue**: The RAG functionality may cause segmentation faults on macOS when using Python 3.13 due to compatibility issues with the sentence-transformers library.

**Symptoms**: 
- Application crashes with "segmentation fault" error
- Multiprocessing resource tracker warnings

**Solution**: Use the `--no-rag` flag to disable RAG functionality:
```bash
poetry run python -m app "word" -l en --no-rag
```

**Workaround**: The application will automatically fall back to simple LLM responses when RAG is disabled, still providing useful information about words.

## Example Output

```
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