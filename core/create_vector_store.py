# notes
# refer for pdf RAG: https://cookbook.openai.com/examples/parse_pdf_docs_for_rag

## Full Flow
# !pip install llama-index-llms-gemini
import json
import os
import time
import logging
import math
import asyncio
from pathlib import Path
import multiprocessing
from dotenv import load_dotenv
from datetime import datetime, timezone
import logging
import subprocess
import hashlib
from collections import deque
import nest_asyncio

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, CodeSplitter, SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor, DocumentContextExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.llms.litellm import LiteLLM


import chromadb
import tiktoken

import sys
sys.path.append(str(Path(__file__).parent.absolute()))  # add the parent directory to the path

from utilities import create_custom_logger, get_large_files, setup_llm_logs, GoogleGenAIDummyTokensizer, HuggingfaceTokenizer
from custom_components.custom_extractors import CustomDocumentContextExtractor
from custom_components.custom_parsers import CustomMarkdownNodeParser
from custom_components.custom_google_genai import CustomGoogleGenAI


load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY', '')

# sample data - https://www.gutenberg.org/cache/epub/24022/pg24022.txt
# sample data: 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt'

ROOT_OUTPUT_DIR = os.environ.get('PROCESSED_DATA_PATH', 'processed_data')

DEFAULT_CHROMADB_PATH = ROOT_OUTPUT_DIR + '/' + 'chromadb'

## Config
INDEX_NAME = 'google_genai-api'
INPUT_DIR = 'data/google_genai/api/'  # change this to the input directory
OUTPUT_DIR = ROOT_OUTPUT_DIR + '/' + INDEX_NAME  # do not change this

FILE_TYPES = ['.md', '.mdx']
# Select Extractors - To add metadata to each node. Some of them may use many LLM calls. Use only if needed.
METADATA_EXTRACTORS = ['CustomDocumentContextExtractor']  # choose from ['TitleExtractor', 'SummaryExtractor', 'KeywordExtractor', 'CustomDocumentContextExtractor' etc]

CHROMADB_PATH = DEFAULT_CHROMADB_PATH  # change if you like
CHROMADB_COLLECTION = INDEX_NAME

LLM_MODEL_PROVIDER = 'litellm'  # choose from ['litellm', 'ollama', 'gemini', 'groq']
LLM_MODEL = "gemini/gemini-2.5-flash" # "cerebras/llama-3.3-70b"  #  # "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo" # "cerebras/llama-3.3-70b"  # "groq/llama-3.3-70b-versatile"  # "cerebras/llama-3.3-70b"
RATE_LIMIT = 7 # LLM req/min, -1 if no limit

RUN_PARALLEL = False  # process nodes in parallel using async

# LITELLM_MODEL = "gemini/gemini-2.5-flash"  # "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" #
# GEMINI_MODEL = "models/gemini-2.5-flash"  # "models/gemini-2.5-flash"
# GROQ_MODEL = "llama-3.3-70b-versatile"
# OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0"

HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

TIKTOKEN_TOKENIZER_MODEL = "cl100k_base"
GEMINI_TOKENIZER_MODEL = LLM_MODEL
HUGGINGFACE_TOKENIZER_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"


EMBEDDING_PROVIDER = 'GoogleGenAIEmbedding'  # choose from ['HuggingFaceEmbedding', 'GoogleGenAIEmbedding', etc]
EMBEDDING_MODEL = GEMINI_EMBEDDING_MODEL

# Only used for token counting, best use tiktoken unless accuracy is needed
TOKENIZER_PROVIDER = 'tiktoken'  # chose from ['gemini', 'huggingface', 'tiktoken' etc]
TOKENIZER_MODEL_NAME = TIKTOKEN_TOKENIZER_MODEL  # choose from ['models/gemini-2.0-flash', 'meta-llama/Meta-Llama-3.1-70B-Instruct', 'cl100k_base' etc]
MAX_NODE_TOKENS = 2000

DOCSTORE_PATH = (Path(OUTPUT_DIR) / 'docstore.json').as_posix()
CONFIG_PATH = (Path(OUTPUT_DIR) / 'config.json').as_posix()

EXCLUDE_FILES = ['__all_docs__.md']  # files/folders to exclude. eg ['file.txt', 'folder1/', 'folder2/b.txt', 'folder2/folder3/']
SKIP_LARGE_FILES = True

def load_existing_configs(config_path):
    """Load existing configs from the config file, handling errors gracefully."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as fp:
                content = fp.read()
                if content.strip():  # Ensure file is not empty
                    configs = json.loads(content)  # Changed from json.load(fp)
                    if not isinstance(configs, list):  # If existing config is not a list, wrap it in a list
                        logger.warning(f"Existing config in {config_path} is not a list. Wrapping it in a list.")
                        configs = [configs]
                    return configs
        except json.JSONDecodeError:
            raise Exception(f"Could not decode JSON from {config_path}. Starting with an empty config list.")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while reading {config_path}: {e}. Starting with an empty config list.")
    return []


# Find and exclude large files over 20KB if enabled
def get_large_files_to_exclude(excluded_files):
    if SKIP_LARGE_FILES:
        large_files = get_large_files(INPUT_DIR, min_size_kb=50, extensions=('.txt', '.md', '.mdx'))
        for file_path, size_kb in large_files:
            # Convert absolute path to relative path from INPUT_DIR
            rel_path = Path(file_path).absolute().relative_to(Path(INPUT_DIR).absolute()).as_posix()
            excluded_files.append(rel_path)
            print(f"Excluding large file: {rel_path} ({size_kb:.1f}KB)")
    return excluded_files


def get_llm(config):
    if config['llm_model_provider'] == 'groq':
        return Groq(model=config['llm_model'], api_key=GROQ_API_KEY, max_retries=2, retry_on_rate_limit=True) # Number of retry attempts
    elif config['llm_model_provider'] == 'gemini':
        return CustomGoogleGenAI(
            model=config['llm_model'],
            api_key=GEMINI_API_KEY, 
            max_retries=2,  # Number of retry attempts
            retry_on_rate_limit=True
        )
    elif config['llm_model_provider'] == 'ollama':
        return Ollama(model=config['llm_model'], request_timeout=120.0, context_window=8192, )
    elif config['llm_model_provider'] == 'litellm':
        import litellm
        litellm.suppress_debug_info = True
        return LiteLLM(model=config['llm_model'], max_tokens=8192, max_retries=6)
    else:
        raise NotImplementedError(f"LLM provider {config['llm_model_provider']} invalid or not implemented")


def get_embed_model(config):
    if config['embedding_provider'] == 'HuggingFaceEmbedding':
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=config['embedding_model'])
    elif config['embedding_provider'] == 'GoogleGenAIEmbedding':
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        return GoogleGenAIEmbedding(model_name=config['embedding_model'], api_key=GEMINI_API_KEY)
    else:
        raise NotImplementedError(f"Embedding provider {config['embedding_provider']} invalid or not implemented")
    

def get_tokenizer(config, llm):
    if config['tokenizer_provider'] == 'gemini':
        return GoogleGenAIDummyTokensizer(llm=llm).encode
    elif config['tokenizer_provider'] == 'huggingface':
        return HuggingfaceTokenizer(model=config['tokenizer_model_name']).encode
    elif config['tokenizer_provider'] == 'tiktoken':
        return tiktoken.get_encoding(encoding_name=config['tokenizer_model_name']).encode
    else:
        raise NotImplementedError(f"{config['tokenizer_provider']} invalid or not implemented")


def get_metadata_extractors(config, llm, docstore=None):
    metadata_extractors = []
    for extractor in config['metadata_extractors']:
        if extractor == 'TitleExtractor':
            metadata_extractors.append(TitleExtractor(llm=llm, show_progress=False))
        elif extractor == 'SummaryExtractor':
            metadata_extractors.append(SummaryExtractor(llm=llm, show_progress=False))
        elif extractor == 'KeywordExtractor':
            metadata_extractors.append(KeywordExtractor(llm=llm, show_progress=False))
        elif extractor == 'DocumentContextExtractor':
            if docstore is None:
                raise ValueError("docstore with original documents is required for Contextual Extractor")
            context_extractor = DocumentContextExtractor(
                # these 2 are mandatory
                docstore=docstore,
                max_context_length=128000,
                # below are optional
                llm=llm,  # default to Settings.llm
                oversized_document_strategy="warn",
                max_output_tokens=100,
                key="context",
                prompt=DocumentContextExtractor.SUCCINCT_CONTEXT_PROMPT,
                show_progress=False
            )
            metadata_extractors.append(context_extractor)
        elif extractor == 'CustomDocumentContextExtractor':
            if docstore is None:
                raise ValueError("docstore with original documents is required for Contextual Extractor")
            context_extractor = CustomDocumentContextExtractor(
                # these 2 are mandatory
                docstore=docstore,
                max_context_length=128000,
                # below are optional
                llm=llm,  # default to Settings.llm
                oversized_document_strategy="warn",
                # max_output_tokens=100,
                key="context",
                prompt=CustomDocumentContextExtractor.ORIGINAL_CONTEXT_PROMPT,
                show_progress=False
            )
            metadata_extractors.append(context_extractor)
        else:
            raise ValueError(f"Extractor {extractor} not available")
    return metadata_extractors


def get_nodes_from_document(document, embed_model, tokenizer, max_tokens, max_header_level=3):
    file_extension = Path(document.metadata['file_name']).suffix.lower()
    if file_extension in ('.md', '.mdx'):
        node_parser = CustomMarkdownNodeParser(max_tokens=max_tokens, max_header_level=max_header_level, split_pattern=r'\*\*(.*?)\*\*', tokenizer=tokenizer)
    elif file_extension == '.txt':
        node_parser = SemanticSplitterNodeParser(buffer_size=3, embed_model=embed_model)
    elif file_extension == '.py':
        node_parser = CodeSplitter(language='python', chunk_lines=70, chunk_lines_overlap=10, max_chars=3000)
    else:
        raise ValueError(f"Filetype not supported.")
    
    nodes = node_parser.get_nodes_from_documents([document])
    return nodes

def get_nodes_from_documents(documents, embed_model, tokenizer, max_tokens):
    all_nodes = []
    for document in documents:
        nodes = get_nodes_from_document(document, embed_model, tokenizer, max_tokens)
        # if file is markdown and has more than 20 nodes, use max_header_level=2 to reduce the no of chunks
        if len(nodes) > 20:
            file_extension = Path(document.metadata['file_name']).suffix.lower()
            if file_extension in ('.md', '.mdx'):
                nodes = get_nodes_from_document(document, embed_model, tokenizer, max_tokens, max_header_level=2)
        all_nodes.extend(nodes)
    return all_nodes



def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False
    

async def process_nodes_with_ratelimit(nodes, transformations, run_parallel=True, rate_limit=-1, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    actual_rate_limit = rate_limit
    # decrease rate limit as per the no of llm based transformers
    llm_ops = 0
    for transformation in transformations:
        if hasattr(transformation, 'llm'):
            llm_ops += 1
    if llm_ops > 1 and actual_rate_limit > 0:
        actual_rate_limit = actual_rate_limit // llm_ops

    transformed_nodes = []
    batch_size_for_pipeline = actual_rate_limit if actual_rate_limit > 0 else 60 
    
    if not nodes: # Handle empty nodes list
        return []

    total_batches = math.ceil(len(nodes) / batch_size_for_pipeline)
    
    for batch_idx in range(total_batches):
        batch_start_time = time.time()
        # Get current batch of nodes
        start_idx = batch_idx * batch_size_for_pipeline
        end_idx = min(start_idx + batch_size_for_pipeline, len(nodes)) 
        batch_nodes = nodes[start_idx:end_idx]
        
        if not batch_nodes: # Should not happen if total_batches is calculated correctly from non-empty nodes
            continue

        logger.info(f"Batch {batch_idx + 1}/{total_batches}, processing {len(batch_nodes)} nodes.")

        pipeline = IngestionPipeline(
            transformations=transformations
        )
        
        if run_parallel:
            batch_retries = 0
            while True:
                try:
                    processed_batch = await pipeline.arun(nodes=batch_nodes, in_place=False, show_progress=False)
                    # processed_batch = pipeline.run(nodes=batch_nodes, in_place=False, show_progress=False)

                    break
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx + 1}/{total_batches}. Retrying...: {e}")
                    time.sleep(70)
                    batch_retries += 1
                    if batch_retries > 1:
                        logger.error('Aborting batch ...')
                        processed_batch = []
                        break
        else:  # process nodes one by one
            processed_batch = []
            for i, node in enumerate(batch_nodes):
                node_failure = False
                node_retries = 0
                # for transformation in transformations:
                processed_node = node
                while True:
                    try:
                        processed_nodes = await pipeline.arun(nodes=[processed_node], in_place=False, show_progress=False)
                        processed_node = processed_nodes[0]
                        # processed_node = transformation.process_nodes(nodes=[processed_node], in_place=False, show_progress=False)
                        # if hasattr(transformation, 'llm'):
                        #     time.sleep(1)
                        break
                    except Exception as e:
                        logger.error(f"Error processing node {i}. Retrying...: {e}")
                        time.sleep(70)
                        node_retries += 1
                        if node_retries > 1:
                            logger.error(f'Aborting node {i}...')
                            node_failure = True
                            break
                if not node_failure:
                    processed_batch.append(processed_node)

        transformed_nodes.extend(processed_batch)
        batch_end_time = time.time()
        elapsed_time = batch_end_time - batch_start_time
        logger.info(f"Batch {batch_idx + 1}/{total_batches} finished in {elapsed_time:.1f}s")
        if actual_rate_limit > 0 and elapsed_time < 60:
            await asyncio.sleep(60 - elapsed_time + 1)
                            
    return transformed_nodes


def setup_application_logging(output_dir):
    logging.basicConfig(
        level=logging.WARNING,
        filename=(Path(output_dir) / "warnings.log").as_posix(),              # All logs go here
        filemode="a",                    # 'w' to overwrite each run
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Create your application logger
    logger = logging.getLogger('documentation_agent')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger
    
    # Create file handler
    file_handler = logging.FileHandler((Path(output_dir) / "rag.log").as_posix())
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    return logger


async def create_vector_store(config):
    
    # check if embedding model is consistent if config exists (in case of rerun)
    existing_configs = load_existing_configs(CONFIG_PATH)
    if len(existing_configs) > 0:
        if existing_configs[-1]['embedding_model'] != EMBEDDING_MODEL:
            raise Exception(f"Embedding model mismatch: Current model '{EMBEDDING_MODEL}' differs from last run's model '{existing_configs[-1]['embedding_model']}'. Using different embedding models will make vector store incompatible.")


    # Setup logging
    logger = setup_application_logging(OUTPUT_DIR)
    llm_logger = create_custom_logger('LLMlogger', (Path(OUTPUT_DIR) / "llm_events.log").as_posix())
    setup_llm_logs(llm_logger, Settings, show_text=False, short_inputs=False, short_outputs=False)

    
    is_jupyter_notebook = is_notebook()

    # # for jupyter notebooks - to fix event loop issue
    if is_jupyter_notebook:
        import nest_asyncio
        nest_asyncio.apply()

    # nest_asyncio.apply()

    # for jupyter notebook - Start ChromaDB server
    if is_jupyter_notebook:
        process = subprocess.Popen(["chroma", "run", "--path", CHROMADB_PATH])
        remote_db = chromadb.HttpClient()
        chroma_collection = remote_db.get_or_create_collection(CHROMADB_COLLECTION)
    else:
        db = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        chroma_collection = db.get_or_create_collection(CHROMADB_COLLECTION)
    
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Initialize stores
    # if Path(DOCSTORE_PATH).exists():
    #     docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
    #     logger.info(f"Loaded existing docstore with {len(docstore.docs)} documents")
    # else:
    #     docstore = SimpleDocumentStore()

    # storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    
    llm = get_llm(config)
    embed_model = get_embed_model(config)
    tokenizer = get_tokenizer(config, llm)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.tokenizer = tokenizer

    # tokenizer = GeminiTokenizer()
    # Settings.tokenizer = tokenizer
    max_tokens = config['max_node_tokens']

    # Step 1: load Documents
    files_to_exlude = get_large_files_to_exclude(EXCLUDE_FILES)
    documents = SimpleDirectoryReader(input_dir=INPUT_DIR, exclude=files_to_exlude, recursive=True, filename_as_id=True,
                                       required_exts=FILE_TYPES).load_data()
    for i, document in enumerate(documents):
        document.doc_id = Path(document.metadata['file_path']).relative_to(Path(INPUT_DIR).absolute()).as_posix()
        document.metadata['file_path'] = document.doc_id
    print('Document Count', len(documents))
    
    original_docs_docstore = SimpleDocumentStore()
    await original_docs_docstore.async_add_documents(documents)
    print('docstore created')

    # Track processed documents and generate batch IDs
    # processed_node_ids = set(docstore.docs.keys())
    processed_node_ids = set(chroma_collection.get(include=[])['ids'])
    print(f"{len(processed_node_ids)} nodes already processed")


    ## ! Important. #todo
    # Implement Logic to split markdown nodes further since Chunk size not followed by MarkdownNode parse
    try:
        doc_batch_size = 1
        all_nodes = []
        # can be optimised to create docstore only for docs being processed in one iteration
        metadata_extractors = get_metadata_extractors(config, llm, original_docs_docstore)
        transformations = metadata_extractors + [embed_model]
        
        for index in range(0, len(documents), doc_batch_size):
            t_start = time.time()
            logger.info(f"doc [{index}-{index+doc_batch_size}] start {time.time()} {documents[index].doc_id}")
            # Step 2: Chunk Documents into Nodes
            nodes = get_nodes_from_documents(documents=documents[index:index+doc_batch_size],
                                            embed_model=embed_model, tokenizer=tokenizer, max_tokens=max_tokens)
            # assign node id to nodes
            for node in nodes:
                node.node_id = f"{node.ref_doc_id}-{node.start_char_idx}-{node.end_char_idx}"
                # node.node_id = str(hashlib.sha256(f"{node.ref_doc_id} {node.start_char_idx} {node.end_char_idx} {node.text}".encode()).hexdigest())
            

            # logger.info(f"before process {len(nodes)} nodes")
            nodes = [node for node in nodes if node.node_id not in processed_node_ids]
            if len(nodes) == 0:
                logger.info(f"skipping nodes as they are already processed")
                continue

            # remove duplicate nodes: sometimes some nodes may have same doc, start, end due to issue in parser, or exactly same content in same doc
            # Remove duplicate nodes based on node_id
            seen_node_ids = set()
            unique_nodes = []
            for node in nodes:
                if node.node_id not in seen_node_ids:
                    seen_node_ids.add(node.node_id)
                    unique_nodes.append(node)
            nodes = unique_nodes

            logger.info(f"processing {len(nodes)} nodes")
            # Step 3: extract metadata and embeddings for nodes
            nodes = await process_nodes_with_ratelimit(nodes=nodes, transformations=transformations, run_parallel=RUN_PARALLEL, rate_limit = RATE_LIMIT, logger=logger)
            
            if nodes is None or len(nodes) == 0:
                raise Exception('Error processing nodes. Aborting.')

            # Step 4: Save the Nodes/Chunks in vector store
            total_tokens = sum([len(Settings.tokenizer(node.get_content(metadata_mode=MetadataMode.EMBED))) for node in nodes])
            node_ids = await vector_store.async_add(nodes)
            # docstore.add_documents(nodes)
            # docstore.persist(DOCSTORE_PATH)
            logger.info(f"added {len(nodes)} to vector store. Total tokens = {total_tokens}")
            all_nodes.extend(nodes)
            t_end = time.time()
            logger.info(f"time for docs {index} to {index+doc_batch_size} = {round(t_end-t_start)}, nodes: {len(nodes)}")
            # time.sleep(1)

    finally:
        
        # Persist the docstore
        # docstore.persist(DOCSTORE_PATH)
        # logger.info(f"saving doctore to {DOCSTORE_PATH}")
        
        
        # Check if config file exists and read existing config
        all_configs = existing_configs
        
        current_run_config = config.copy() # Use a copy to avoid modifying the original config dict for the current run
        current_run_config['run_time'] = datetime.now(timezone.utc).isoformat()
        current_run_config['run_nodes'] = len(all_nodes) if 'all_nodes' in locals() else 0 # ensure all_nodes exists
        current_run_config['run_docs'] = list(set([node.ref_doc_id for node in all_nodes]))
        
        all_configs.append(current_run_config)

        with open(CONFIG_PATH, 'w') as fp:
            json.dump(all_configs, fp, indent=4)
        
        if 'process' in locals() and process.poll() is None: # Check if process exists and is running
            process.terminate()


def main():
    try:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    except FileExistsError as e:
        print(f'Processed_data directory {OUTPUT_DIR} already exists. Please delete that or specify another directory.')
        exit(1)
    print('output dir', OUTPUT_DIR)

    config = {
        'index_name': INDEX_NAME,
        'llm_model_provider': LLM_MODEL_PROVIDER,
        'llm_model': LLM_MODEL,
        'rate_limit': RATE_LIMIT,
        'input_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
        'file_types': FILE_TYPES,
        'vector_store': 'chroma',
        'chromadb_path': CHROMADB_PATH,
        'chroma_collection': CHROMADB_COLLECTION,
        'doctsore_path': DOCSTORE_PATH,
        'embedding_provider': EMBEDDING_PROVIDER,
        'embedding_model': EMBEDDING_MODEL,
        'tokenizer_provider': TOKENIZER_PROVIDER,
        'tokenizer_model_name': TOKENIZER_MODEL_NAME,
        'max_node_tokens': MAX_NODE_TOKENS,
        'metadata_extractors': METADATA_EXTRACTORS,
        'datetime': datetime.now(timezone.utc).isoformat(),
    }
    
    t3 = time.time()
    asyncio.run(create_vector_store(config))
    t4 = time.time()
    print('time', round(t4-t3))

if __name__ == '__main__':
    # Fix for Windows (needed if freezing the application) - to support pipeline num_workers
    multiprocessing.freeze_support()  # Ensures proper multiprocessing behavior on Windows
    main()  # Runs the actual logic
    
