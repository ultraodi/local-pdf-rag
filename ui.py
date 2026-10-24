import os
import pypdf
import gradio as gr
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ---------- Config (environment variables) ----------
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_langchain_db")
COLLECTION = "docs"
BATCH_SIZE = 50

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url=OLLAMA_URL, keep_alive=30
)
llm = ChatOllama(model="llama3.2", base_url=OLLAMA_URL, temperature=0)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)


def new_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


vector_store = new_store()


# ---------- Indexing ----------
def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": os.path.basename(file_path), "page": i},
        )
        for i, page in enumerate(reader.pages, start=1)
    ]


def status_text() -> str:
    n = vector_store._collection.count()
    return f"{n} chunks indexed." if n else "No document loaded."


def index_pdf(file_path, progress=gr.Progress()):
    global vector_store
    if file_path is None:
        return "Please upload a PDF first."

    progress(0, desc="Reading PDF")
    docs = [d for d in load_pdf_pages(file_path) if d.page_content.strip()]
    if not docs:
        return "No text found. This may be a scanned PDF (images only)."

    # Replace whatever document was loaded before
    vector_store.delete_collection()
    vector_store = new_store()

    splits = text_splitter.split_documents(docs)
    for i in range(0, len(splits), BATCH_SIZE):
        vector_store.add_documents(documents=splits[i : i + BATCH_SIZE])
        progress(
            (min(i + BATCH_SIZE, len(splits)), len(splits)),
            desc="Embedding chunks",
        )

    name = os.path.basename(file_path)
    return f"Indexed {name}: {len(docs)} pages, {len(splits)} chunks."


def reset():
    global vector_store
    vector_store.delete_collection()
    vector_store = new_store()
    # file, status, question, answer, sources
    return None, "Document cleared.", "", "", ""


# ---------- Asking ----------
def ask(question: str, k: int = 4):
    docs = vector_store.similarity_search(question, k=k)

    context = "\n\n".join(
        f"[{i}] (page {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    messages = [
        (
            "system",
            "You answer questions using ONLY the numbered context passages provided. "
            "After each claim, cite the passage number in square brackets, like [1] or [2][3]. "
            "If the context does not contain the answer, say you don't know. "
            "Do not use outside knowledge.",
        ),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    return llm.invoke(messages).content, docs


def respond(question: str):
    question = (question or "").strip()
    if not question:
        return "Please type a question.", ""
    if vector_store._collection.count() == 0:
        return "Upload a PDF first.", ""

    answer, docs = ask(question)
    sources = "\n".join(
        f"[{i}] {d.metadata['source']}, page {d.metadata['page']}"
        for i, d in enumerate(docs, start=1)
    )
    return answer, sources


# ---------- UI ----------
with gr.Blocks(title="PDF Q&A (RAG)") as demo:
    gr.Markdown("# PDF Q&A\nUpload a PDF, then ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
            with gr.Row():
                index_btn = gr.Button("Index document", variant="primary")
                reset_btn = gr.Button("Reset")
            status = gr.Textbox(label="Status", value=status_text(), interactive=False)

        with gr.Column(scale=2):
            question = gr.Textbox(label="Question", placeholder="What is phishing?")
            ask_btn = gr.Button("Ask", variant="primary")
            answer = gr.Textbox(label="Answer", lines=6, interactive=False)
            sources = gr.Textbox(label="Sources", lines=4, interactive=False)

    index_btn.click(index_pdf, inputs=file_in, outputs=status)
    reset_btn.click(reset, outputs=[file_in, status, question, answer, sources])
    ask_btn.click(respond, inputs=question, outputs=[answer, sources])
    question.submit(respond, inputs=question, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)