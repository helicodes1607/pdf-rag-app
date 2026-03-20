import os
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)

    # Join ALL pages into one text first, then chunk
    full_text = "\n\n".join(d.text for d in docs if getattr(d, "text", None))

    chunks = splitter.split_text(full_text)  # split_text, not split_texts
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        raise ValueError("No valid text chunks to embed.")

    all_embeddings = []
    batch_size = 200  # safely below the array-size limit

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings
