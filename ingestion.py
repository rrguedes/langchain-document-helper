import asyncio
import os
import ssl
from typing import List, Any, Dict
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import log_info, log_success, log_error, log_warning, log_header, Colors

load_dotenv()


def configure_ssl_context() -> ssl.SSLContext:
    """Configura o contexto SSL para usar o certificado CA do certifi."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    return ssl_context


def get_embeddings() -> OpenAIEmbeddings:
    """Retorna uma instância de OpenAIEmbeddings."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=50,
        retry_min_seconds=10,
        show_progress_bar=False,
    )


def main():
    # Your main logic here
    ssl_context = configure_ssl_context()
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )
    tv_extract = TavilyExtract()
    tv_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
    tv_crawl = TavilyCrawl(extract=tv_extract)


if __name__ == "__main__":
    asyncio.run(main())
