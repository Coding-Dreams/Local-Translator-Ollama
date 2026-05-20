# PDF/MOBI Book Translator (Local Ollama)

Translate a PDF or MOBI book into one or more target languages using local compute and a local Ollama model.
NOTE: IT IS IMPORTANT YOU USE LLM MODELS THAT ARE GOOD FOR THE WANTED LANGUAGE GIGO!!!!

## What this does

- Extracts text from each page of a source PDF
- Extracts text from PDF or MOBI input (MOBI is unpacked locally)
- Splits long page text into chunks for safer LLM translation
- Adds overlap context between chunks to improve coherence
- Uses tagged segment translation to reduce dropped lines
- Optionally auto-generates a glossary from the book text
- Optionally runs a second-pass consistency edit
- Runs QA checks and retries low-quality chunks automatically
- Checks numeric fidelity (prevents quantity drift)
- Runs optional back-translation QA (target -> English) to detect meaning drift
- Optionally runs model-based fidelity review/repair per chunk
- Calls local Ollama API for translation
- Produces:
  - A translated `.txt` file (with page separators)
  - A translated `.pdf` file per target language
- Produces a per-language QA report (`.qa_report.json`)
- Saves progress so interrupted runs can resume

## Setup

1. Install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start Ollama locally (native install or container) and pull a model:

```bash
ollama serve
ollama pull [MODEL NAME] (e.g. gemma3:12b)
```

If you run Ollama in Docker, expose `11434` and pass `--ollama-url` if needed.

## Usage

Default run (Spanish + Mandarin):

```bash
python3 translate_pdf.py "NAME.pdf"
```

High-quality run with all improvements enabled:

```bash
python3 translate_pdf.py "NAME.mobi" \
  --languages spanish,mandarin \
  --auto-glossary \
  --second-pass \
  --backtranslate-qa \
  --fidelity-review \
  --strict-number-preservation \
  --qa-retries 2 \
  --overlap-sentences 2 \
  --force-restart
```

Use a different model:

```bash
python3 translate_pdf.py "NAME.mobi" --model qwen2.5:14b
```

Use your own glossary JSON:

```bash
python3 translate_pdf.py "NAME.mobi" --glossary-file glossary.json
```

Outputs are written to `./output`:

- `output/NAME.spanish.txt`
- `output/NAME.spanish.pdf`
- `output/NAME.mandarin.txt`
- `output/NAME.mandarin.pdf`
- `output/NAME.spanish.qa_report.json`
- `output/NAME.mandarin.qa_report.json`
- `output/NAME.<lang>.glossary.json` (when `--auto-glossary` is used)

## Glossary automation

If you do not have a glossary, use `--auto-glossary`:

- The script extracts high-value terms from the source text
- It asks the local Ollama model to translate those terms into each target language
- It writes the merged glossary to output and reuses it during translation

If both `--auto-glossary` and `--glossary-file` are used, your glossary file wins on conflicts.

## Quality tuning

If translation is still too literal or misses facts:

- Use `--fidelity-review` (enabled by default) for an extra repair pass
- Use `--backtranslate-qa` (enabled by default) for round-trip semantic checks
- Keep `--strict-number-preservation` enabled to prevent numeric drift
- Increase context continuity with `--overlap-sentences 3`
- Reduce per-call load with `--chunk-chars 1800` and `--segment-chars 550`
- If you changed strategies/models, use `--force-restart` to ignore cached progress

## Notes

- This is translation-focused, not layout-preserving. The output PDF is reflowed text.
- If the source PDF is scanned images, `pypdf` text extraction may be empty (OCR would be required first).
- Some MOBI files are DRM-protected or non-standard; local extraction may fail in those cases.
- For very large books, use a smaller `--chunk-chars` value if your model struggles with long passages.
