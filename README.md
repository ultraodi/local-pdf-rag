# Local PDF RAG

Upload a PDF, ask questions about it, and get answers with page citations. Everything runs locally with [Ollama](https://ollama.com), so your documents never leave your machine.

Built with Gradio, LangChain, ChromaDB, and Docker.

## How it works

```
PDF → text extraction → chunking → embeddings → vector DB → retrieval → LLM → cited answer
```

1. The PDF is read page by page and split into overlapping chunks.
2. Each chunk is embedded with `nomic-embed-text` and stored in ChromaDB, along with its page number.
3. When you ask a question, the most similar chunks are retrieved.
4. `llama3.2` answers using only those chunks and cites them as `[1]`, `[2]`, etc. The sources list shows the file and page for each one.

If the document doesn't contain the answer, the model is instructed to say it doesn't know.

## Features

- Upload a PDF through the browser and index it with a progress bar
- Answers cite the source passages, with page numbers
- Reset button clears the document and the vector database
- Runs in Docker, with Ollama either on your machine or in a container
- Run several independent copies side by side

## Requirements

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose (for the Docker setup)
- [Ollama](https://ollama.com/download) (unless you use the bundled Ollama container)
- Roughly 3 GB of free disk space for the models

## Quick start with Docker

### Option A: use Ollama running on your machine (recommended on Mac)

Pull the models once:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Then start the app:

```bash
docker compose up --build
```

Open <http://localhost:7860>.

### Option B: Ollama in a container

```bash
OLLAMA_BASE_URL=http://ollama:11434 docker compose --profile ollama up --build -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
```

The models are stored in a Docker volume, so you only download them once.

> Docker on macOS cannot use Apple GPU acceleration, so Ollama in a container is much slower on a Mac than native Ollama. On Linux with an NVIDIA GPU, the container option works well.

### Running multiple copies

Give each copy its own project name and port. Each gets separate storage and its own document:

```bash
APP_PORT=7861 docker compose -p rag2 up -d
```

## Run without Docker

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull llama3.2

python ui.py
```

Open <http://localhost:7860>.

## Usage

1. Upload a PDF and click **Index document**. Wait for the status to show the number of chunks.
2. Type a question and press Enter or click **Ask**.
3. Read the answer and check the **Sources** box for file and page references.
4. Click **Reset** to clear the document and start over. Uploading a new PDF also replaces the old one.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` (`http://host.docker.internal:11434` in Docker) | Where the app finds Ollama |
| `CHROMA_DIR` | `./chroma_langchain_db` (`/data/chroma` in Docker) | Where the vector database is stored |
| `APP_PORT` | `7860` | Host port (Docker Compose only) |

To use different models, change the model names near the top of `ui.py`. If you change the embedding model, reset and re-index, because vectors from different models aren't comparable.

## Project structure

```
.
├── ui.py                  # Gradio app: upload, index, ask, reset
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

## Limitations

- One document at a time per running app
- One shared document store per container, so it's designed for a single user or team
- Scanned PDFs (images with no text layer) aren't supported, since there's no OCR
- Only PDFs are supported for now
- Each question is independent, with no conversation memory

## Troubleshooting

**The first indexing attempt fails with a connection reset.** Ollama may still be loading the model. Wait a few seconds and try again.

**The container can't reach Ollama.** On Linux, Ollama listens only on localhost by default. Set `OLLAMA_HOST=0.0.0.0` for the Ollama service, or use Option B.

**Answers say "I don't know" for things that are in the PDF.** Check that the document finished indexing (the status box shows the chunk count), and try rephrasing the question.

**I edited `ui.py` but nothing changed in Docker.** The code is baked into the image, so run `docker compose up --build` again.

**Logs:** `docker compose logs -f app`

## License

MIT. See [LICENSE](LICENSE).