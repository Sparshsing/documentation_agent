INPUT_DIR = "./data/crawl4ai"
MODEL = "models/gemini-2.0-flash-exp"# "llama-3.3-70b-versatile"
RATE_LIMIT = -1  # LLM req/min, -1 if no limit
COLLECTION = 'simplemdvectors'

import os
import time
import logging
import math
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from IPython.display import display, Markdown


from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core import Settings
from llama_index.core.extractors import TitleExtractor, SummaryExtractor
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from vertexai.preview import tokenization
import chromadb
from tqdm import tqdm

load_dotenv()

# llm = Groq(model=MODEL, api_key=os.environ['GROQ_API_KEY'], max_retries=3,  # Number of retry attempts
#     retry_on_rate_limit=True)
llm = Gemini(
    model=MODEL,
    api_key=os.environ['GEMINI_API_KEY'], 
    max_retries=3,  # Number of retry attempts
    retry_on_rate_limit=False
)
summary_extractor = SummaryExtractor(llm = llm)
node_parser = MarkdownNodeParser(chunk_size=512, chunk_overlap=32)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = llm
Settings.context_window = 32000
Settings.embed_model = embed_model

logger = logging.getLogger('LLMEventsLogger')
handler = logging.FileHandler('logs/llm_events.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Debugging - Observability - LLM calls
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import (
    LLMCompletionEndEvent,
    LLMChatEndEvent,
)
from llama_index.core.instrumentation.events.embedding import EmbeddingEndEvent


class ModelEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ModelEventHandler"

    def handle(self, event) -> None:
        """Logic for handling event."""
        if isinstance(event, LLMCompletionEndEvent):
            logger.info(f"LLM Prompt length: {len(Settings.tokenizer(event.prompt))}")
            logger.info(f"LLM Completion length: {len(Settings.tokenizer(event.response.text))}")
        elif isinstance(event, LLMChatEndEvent):
            messages_str = "\n".join([str(x) for x in event.messages])
            logger.info(f"LLM Input Messages length: {len(Settings.tokenizer(messages_str))}")
            logger.info(f"LLM Input Messages : {messages_str[:50]} ... {messages_str[:50]}")
            logger.info(f"LLM Response Length: {len(Settings.tokenizer(str(event.response.message)))}")
            logger.info(f"LLM Response: {str(event.response.message)[:50]}...")
            logger.info('---------------------')
            
        elif isinstance(event, EmbeddingEndEvent):
            pass
            # logger.info(f"Embedding {len(event.chunks)} text chunks")

from llama_index.core.instrumentation import get_dispatcher

# root dispatcher
root_dispatcher = get_dispatcher()

# register event handler
root_dispatcher.add_event_handler(ModelEventHandler())

print('index loading')
logger.info('index loading')

db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection(COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

print('index loaded')

# Query Data from the persisted index
query_engine = index.as_query_engine()

print('staring query')
while(True):
    query = input('Enter query. press q to quit')
    if query =='q':
        break
    response = query_engine.query(query)
    print(response)


print('finished')
print('yo')
logger.info('hey')
print('mo')
print('mo')
print('finish')
