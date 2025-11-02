# JosefGPT Local

JosefGPT is a Streamlit-based local AI assistant for knowledge retrieval, coaching, and B2B productivity workflows. It combines hybrid retrieval with OpenAI reasoning, runs fully on your machine, and can optionally mirror Google Drive folders so everything stays in sync.

> This repository merges the previous `ai-book-agent`, `Stealth-mode-OFF/JosefGPT-Local`, and `JosefGPT-App` projects into a single maintained codebase.

## Features
- Chunk and embed PDFs, EPUBs, Markdown, and text files from the `books/`, `texts/`, or `data/` directories.
- Streamlit UI with chat history, configurable retrieval/generation settings, and source previews.
- Typer-based CLI for ingestion, terminal chat, and launching the UI.
- Fully configurable through `.env`, including an offline heuristic fallback when no OpenAI key is provided.

## Quickstart
1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure environment**
   - Copy `.env.example` to `.env`.
   - Fill in `OPENAI_API_KEY` (and any Drive credentials if you plan to sync Google Drive).
3. **Add knowledge sources**
   - Drop PDFs, EPUBs, Markdown, or text files into `books/`, `texts/`, or `data/`.
   - Or configure Google Drive folders (see below) and run `python -m app.cli sync-drive`.
4. **Ingest embeddings**
   ```bash
   python -m app.cli ingest --sync-drive
   ```
5. **Launch the UI**
   ```bash
   streamlit run app/ui.py
   ```

## CLI Commands
Run `python -m app.cli --help` for the full menu.

- `python -m app.cli ingest`  
  Rebuilds the Chroma database. Use `-s path/to/dir` to ingest a custom folder. Pass `--sync-drive` to refresh Google Drive sources first.

- `python -m app.cli chat`  
  Starts an interactive terminal chat. Flags such as `--top-k`, `--temperature`, and `--max-tokens` override defaults, and `--hide-sources` suppresses source summaries.

- `python -m app.cli serve`  
  Convenience wrapper around `streamlit run app/ui.py`.

- `python -m app.cli sync-drive`  
  Downloads the configured Google Drive folders into the local cache. Runs automatically when you use `ingest --sync-drive`.

## Streamlit UI
- Adjust retrieval/generation parameters from the sidebar; settings persist during the session.
- Every reply lists supporting source excerpts with similarity scores and chunk identifiers for quick verification.

## Configuration
Environment variables (see `.env.example`):

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_MODEL` | Chat completion model. | `gpt-4o-mini` |
| `DEFAULT_MODEL` | Fallback model if `OPENAI_MODEL` is unset. | `gpt-4o` |
| `USE_OPENAI_EMBEDDINGS` | Switch between OpenAI and local SentenceTransformer embeddings. | `false` |
| `EMBEDDING_MODEL` | OpenAI embedding model name. | `text-embedding-3-large` |
| `SOURCE_DIRS` | Comma-separated list of directories to scan. | `books,texts,data` |
| `EMBEDDINGS_PATH` | Location for the ChromaDB store. | `embeddings/` |
| `TOP_K` | Default retrieved chunks per query. | `6` |
| `MAX_TOKENS` | Default completion max tokens. | `900` |
| `TEMPERATURE` | Default completion temperature. | `0.3` |
| `LLM_MODE` | `openai`/`online`, `offline`, or `auto` (falls back to offline when no key). | `auto` |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | Path to a service-account JSON with Drive read access. | `None` |
| `GOOGLE_DRIVE_FOLDER_IDS` | Comma-separated Drive folder IDs to mirror locally. | `None` |
| `GOOGLE_DRIVE_CACHE` | Local cache location for mirrored Drive files. | `drive_cache/` |

### Google Drive ingestion
1. Create a Google Cloud service account and grant it Reader access to the Drive folders you want to ingest.
2. Download the service account JSON and set `GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/credentials.json` in your `.env`.
3. Add the target folder IDs to `GOOGLE_DRIVE_FOLDER_IDS` (comma-separated).
4. Run `python -m app.cli sync-drive` to download the ebooks into the local cache, or `python -m app.cli ingest --sync-drive` to sync and ingest in a single step.

## Development & Testing
- Format/compile checks:
  ```bash
  python3 -m py_compile ingest_books.py app/*.py
  ```
- Run the automated test suite:
  ```bash
  pytest
  ```
- Clean embeddings/data quickly by removing the `embeddings/` directory (listed in `.gitignore`).

## Project Layout
```
├── app/
│   ├── cli.py
│   ├── config.py
│   ├── drive_sync.py
│   ├── llm.py
│   ├── query_engine.py
│   └── ui.py
├── books/              # user-provided documents (empty placeholder)
├── data/               # optional structured data (empty placeholder)
├── drive_cache/        # synced Google Drive files (empty placeholder)
├── embeddings/         # Chroma/FAISS stores (empty placeholder)
├── models/             # optional local models (empty placeholder)
├── texts/              # lightweight notes (empty placeholder)
├── tests/              # pytest-based E2E tests
├── agent_ingest_index.py
├── agent_retriever_http_fix.py
├── ingest_books.py
├── main.py
├── render.yaml
├── requirements.txt
└── README.md
```

## Offline Mode
- Set `LLM_MODE=offline` (or omit an `OPENAI_API_KEY`) to run without external API calls.
- In offline mode, JosefGPT synthesises heuristic guidance from retrieved context; the UI and CLI clearly flag offline responses.
