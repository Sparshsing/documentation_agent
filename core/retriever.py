import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

import tiktoken
import chromadb
import Stemmer

from langfuse import get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, CodeSplitter, SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor, DocumentContextExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

# import custom components
sys.path.append(str(Path(__file__).parent.absolute()))  # add the parent directory to the path
from custom_components.custom_google_genai import CustomGoogleGenAI
from utilities import GoogleGenAIDummyTokensizer, HuggingfaceTokenizer


load_dotenv()


PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_PATH = os.environ.get('PROCESSED_DATA_PATH', 'processed_data')

# INDEX_MAPPPING = {
#     'google_genai-api': 'data/google_genai/api',
#     'google_genai-docs': 'data/google_genai/docs',
# }

# Configuration
LLM_MODEL_PROVIDER = 'litellm'  # choose from ['litellm', 'ollama', 'gemini', 'groq']
LLM_MODEL = "gemini/gemini-2.5-flash" # "cerebras/llama-3.3-70b"  #  # "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo" # "cerebras/llama-3.3-70b"  # "groq/llama-3.3-70b-versatile"  # "cerebras/llama-3.3-70b"
USE_DEFAULT_TOKENIZER = True
RETRIEVER_VECTOR_STORE = os.environ.get('RETRIEVER_VECTOR_STORE', 'chromadb')
PINECONE_INDEX_NAME = 'documentation-agent'

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY', '')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')

GRAPH_AVAILABLE = False


def verify_config(config):
    """
    Verifies that the crucial fields in the config are present.
    """
    fields_to_verify = [
        'index_name', 'vector_store', 
        'chroma_collection', 'docstore_path', 
        'embedding_provider', 'embedding_model', 'tokenizer_provider', 
        'tokenizer_model_name'
    ]
    
    for field in fields_to_verify:
        if field not in config:
            print(f"Field '{field}' not found in config")
            return False

    return True


def get_config(index):
    # load config file from processed dir
    config_file = Path(PROCESSED_DATA_PATH) / index / 'config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def get_llm(llm_model_provider, llm_model):
    if llm_model_provider == 'groq':
        from llama_index.llms.groq import Groq
        return Groq(model=llm_model, api_key=GROQ_API_KEY, max_retries=2, retry_on_rate_limit=True) # Number of retry attempts
    elif llm_model_provider == 'gemini':
        return CustomGoogleGenAI(
            model=llm_model,
            api_key=GEMINI_API_KEY, 
            max_retries=2,  # Number of retry attempts
            retry_on_rate_limit=True
        )
    elif llm_model_provider == 'ollama':
        from llama_index.llms.ollama import Ollama
        return Ollama(model=llm_model, request_timeout=120.0, context_window=8192, )
    elif llm_model_provider == 'litellm':
        import litellm
        from llama_index.llms.litellm import LiteLLM
        litellm.suppress_debug_info = True
        return LiteLLM(model=llm_model, max_tokens=8192, max_retries=6)
    else:
        raise NotImplementedError(f"LLM provider {llm_model_provider} invalid or not implemented")


def get_embed_model(embedding_provider, embedding_model):
    if embedding_provider == 'FastEmbedEmbedding':
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        return FastEmbedEmbedding(model_name=embedding_model)
    elif embedding_provider == 'GoogleGenAIEmbedding':
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        return GoogleGenAIEmbedding(model_name=embedding_model, api_key=GEMINI_API_KEY)
    else:
        raise NotImplementedError(f"Embedding provider {embedding_provider} invalid or not implemented")
    

def get_tokenizer(tokenizer_provider, tokenizer_model_name, llm):
    if tokenizer_provider == 'gemini':
        return GoogleGenAIDummyTokensizer(llm=llm).encode
    elif tokenizer_provider == 'huggingface':
        return HuggingfaceTokenizer(model=tokenizer_model_name).encode
    elif tokenizer_provider == 'tiktoken':
        return tiktoken.get_encoding(encoding_name=tokenizer_model_name).encode
    else:
        raise NotImplementedError(f"{tokenizer_provider} invalid or not implemented")


# setup observability
def initialize_langfuse():
    langfuse = get_client()
    langfuse_available = False

    # Verify langfuse connection
    if langfuse.auth_check():
        langfuse_available = True
        LlamaIndexInstrumentor().instrument()
        print("Langfuse client is authenticated and ready!")
        return langfuse, langfuse_available
    else:
        print("Authentication failed. Please check your credentials and host.")
        return langfuse, langfuse_available


async def retrieve_nodes(query, index, top_k=5, mode='hybrid', rerank=True, use_graph=False, config=None):
    # modes: vector, keyword, hybrid (default)
    
    print('retrieve_nodes is runnig')
    print("current directory", os.getcwd())
    print("PROCESSED_DATA_PATH", PROCESSED_DATA_PATH)

    # check if we are able to write to files inside PROCESSED_DATA_PATH
    # test_file = Path(PROCESSED_DATA_PATH) / 'test.txt'
    # try:
    #     with open(test_file, 'w') as f:
    #         f.write('test')
    #     print('test file written')
    # except Exception as e:
    #     print(f"Error writing to test file: {e}")

    # test_file_chromadb = Path(PROCESSED_DATA_PATH) / 'chromadb/test.txt'
    # try:
    #     with open(test_file_chromadb, 'w') as f:
    #         f.write('test')
    #     print('test file written to chromadb')
    # except Exception as e:
    #     print(f"Error writing to test file in chromadb: {e}")

    start_time = time.time()
    print(f"Starting retrieve_nodes for query: {query[:50]}...")

    if config is None:
        config = get_config(index)
    if not verify_config(config):
        raise ValueError("Config is not valid")
    
    llm = get_llm(llm_model_provider=LLM_MODEL_PROVIDER, llm_model=LLM_MODEL)
    Settings.llm = llm
    
    if not USE_DEFAULT_TOKENIZER:
        tokenizer = get_tokenizer(config['tokenizer_provider'], config['tokenizer_model_name'], llm)
        Settings.tokenizer = tokenizer
    
    if use_graph and not GRAPH_AVAILABLE:
        raise ValueError('Graph not available. Using vector index/keyword index.')

    # langfuse, langfuse_available = None, False #initialize_langfuse()
    langfuse, langfuse_available = initialize_langfuse()
    
    # retrieve more initial nodes in case of rerank
    similarity_top_k = top_k*2 if rerank else top_k

    setup_time = time.time()

    # load the index
    if mode in ('vector', 'hybrid'):
        if RETRIEVER_VECTOR_STORE == 'pinecone':
            print('loading pinecone index')
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(name=PINECONE_INDEX_NAME)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=config['chroma_collection'])
            embed_model = get_embed_model(config['embedding_provider'], config['embedding_model'])
            Settings.embed_model = embed_model
            vector_index = VectorStoreIndex.from_vector_store(
                        vector_store,
                        embed_model=embed_model,
                    )
        elif RETRIEVER_VECTOR_STORE == 'chromadb':
            print('loading chroma index')
            chroma_path = Path(PROCESSED_DATA_PATH) / 'chromadb'
            chroma_path = chroma_path.as_posix()
            chroma_colection_name = config['chroma_collection']
            # chroma_client = chromadb.HttpClient(port=8001)
            try:
                chroma_client = chromadb.PersistentClient(path=chroma_path)
                print('chroma client created')
            except Exception as e:
                print(f"Error creating chroma client: {e}")
                raise e
            
            try:
                chroma_collection = chroma_client.get_collection(chroma_colection_name)
                print('chroma collection loaded')
            except Exception as e:
                print(f"Error loading chroma collection: {e}")
                raise e
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            embed_model = get_embed_model(config['embedding_provider'], config['embedding_model'])
            Settings.embed_model = embed_model
            vector_index = VectorStoreIndex.from_vector_store(
                        vector_store,
                        embed_model=embed_model,
                    )
        else:
            raise ValueError(f"Vector store {RETRIEVER_VECTOR_STORE} not supported. Please choose from ['chromadb', 'pinecone']")

    print('index loaded')
    index_creation_time = time.time()

    # get the retriever
    if mode == 'vector':
        retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    elif mode == 'keyword':
        loaded_docstore = SimpleDocumentStore.from_persist_path(config['docstore_path'])
        docstore_nodes = list(loaded_docstore.docs.values())
        retriever = BM25Retriever.from_defaults(nodes=docstore_nodes, similarity_top_k=similarity_top_k, stemmer=Stemmer.Stemmer("english"), language="english")
    elif mode == 'hybrid':
        vector_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
        loaded_docstore = SimpleDocumentStore.from_persist_path(config['docstore_path'])
        docstore_nodes = list(loaded_docstore.docs.values())
        keyword_retriever = BM25Retriever.from_defaults(nodes=docstore_nodes, similarity_top_k=similarity_top_k, stemmer=Stemmer.Stemmer("english"), language="english")
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            num_queries=3,
            similarity_top_k = similarity_top_k,
            llm=llm,
            retriever_weights=[0.7, 0.3],
            mode = "dist_based_score",
            use_async=True,
        )
    else:
        raise ValueError(f"Mode {mode} not supported")

    retriever_setup_time = time.time()

    # retrieve the nodes
    if not langfuse_available:
        retrieved_nodes = await retriever.aretrieve(query)
    else:
        with langfuse.start_as_current_span(name="Retrieve nodes"):
            retrieved_nodes = await retriever.aretrieve(query)
        langfuse.flush()

    retrieval_time = time.time()

    # rerank
    if rerank:
        cohere_rerank = CohereRerank(
            top_n=top_k, model="rerank-v3.5", api_key=COHERE_API_KEY
        )
        retrieved_nodes = cohere_rerank.postprocess_nodes(nodes=retrieved_nodes, query_str=query)

    rerank_time = time.time()
    
    # Print timing breakdown
    print(f"\n=== TIMING BREAKDOWN ===")
    print(f"Setup (LLM, embeddings, settings): {setup_time - start_time:.3f}s")
    print(f"Index loading: {index_creation_time - setup_time:.3f}s")
    print(f"Retriever setup ({mode} mode): {retriever_setup_time - index_creation_time:.3f}s")
    print(f"Node retrieval: {retrieval_time - retriever_setup_time:.3f}s")
    if rerank:
        print(f"Reranking: {rerank_time - retrieval_time:.3f}s")
        print(f"Total time: {rerank_time - start_time:.3f}s")
    else:
        print(f"Reranking: skipped")
        print(f"Total time: {retrieval_time - start_time:.3f}s")
    print(f"========================\n")

    return retrieved_nodes

async def query_index(query, index, top_k=5, mode='hybrid', rerank=True, use_graph=False, config=None):
    
    if config is None:
        config = get_config(index)
    if not verify_config(config):
        raise ValueError("Config is not valid")
    
    retieved_nodes = await retrieve_nodes(query, index, top_k=top_k, mode=mode, rerank=rerank, use_graph=use_graph, config=config)

    llm = get_llm(llm_model_provider=LLM_MODEL_PROVIDER, llm_model=LLM_MODEL)
    embed_model = get_embed_model(config['embedding_provider'], config['embedding_model'])

    Settings.llm = llm
    Settings.embed_model = embed_model

    if not USE_DEFAULT_TOKENIZER:
        tokenizer = get_tokenizer(config['tokenizer_provider'], config['tokenizer_model_name'], llm)
        Settings.tokenizer = tokenizer

    langfuse, langfuse_available = initialize_langfuse()

    response_synthesizer = get_response_synthesizer(llm=llm, response_mode=ResponseMode.COMPACT)
    query_bundle = QueryBundle(query)
    if not langfuse_available:
        response = response_synthesizer.synthesize(query_bundle, retieved_nodes)
    else:
        with langfuse.start_as_current_span(name="reranked query"):
            response = response_synthesizer.synthesize(query_bundle, retieved_nodes)
        langfuse.flush()
    
    return response



def get_nodes_tokens(nodes):
    context = ''
    for node in nodes:
        context = context + '\n\n' + node.node.get_content(metadata_mode=MetadataMode.LLM)
    token_count = len(Settings.tokenizer(context))
    return token_count






