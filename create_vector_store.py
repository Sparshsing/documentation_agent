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

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, CodeSplitter, SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor, DocumentContextExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from custom_components.custom_google_genai import CustomGoogleGenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq


import chromadb
import tiktoken

from utilities import create_custom_logger, get_large_files, setup_llm_logs, GoogleGenAIDummyTokensizer, HuggingfaceTokenizer
from custom_components.custom_extractors import CustomDocumentContextExtractor
from custom_components.custom_parsers import CustomMarkdownNodeParser
load_dotenv()


# todo:
# 3. markdown: use Markdown splitter, but customize to handle token counts of chunk.
# 5. sample data - https://www.gutenberg.org/cache/epub/24022/pg24022.txt
# 6. sample data: 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt'

INPUT_DIR = 'data/langchain/docs/docs/'

EXCLUDE_FILES = []  # files/folders to exclude. eg ['file.txt', 'folder1/', 'folder2/b.txt', 'folder2/folder3/']
SKIP_LARGE_FILES_20KB = True
# Find and exclude large files over 20KB if enabled
if SKIP_LARGE_FILES_20KB:
    large_files = get_large_files(INPUT_DIR, min_size_kb=50, extensions=('.txt', '.md', '.mdx'))
    for file_path, size_kb in large_files:
        # Convert absolute path to relative path from INPUT_DIR
        rel_path = Path(file_path).absolute().relative_to(Path(INPUT_DIR).absolute()).as_posix()
        EXCLUDE_FILES.append(rel_path)
        print(f"Excluding large file: {rel_path} ({size_kb:.1f}KB)")

# directory to store processed data like vecor store, doctore etc
PROCESSED_DIR = (Path('processed_data') / '_'.join(Path(INPUT_DIR).relative_to('./data').parts)).as_posix()

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GROQ_API_KEY = os.environ['GROQ_API_KEY']

DEFAULT_GEMINI_MODEL = "models/gemini-2.0-flash"  # "models/gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0"

HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

DEFAULT_GEMINI_TOKENIZER_MODEL = DEFAULT_GEMINI_MODEL
HUGGINGFACE_TOKENIZER_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
TIKTOKEN_TOKENIZER_MODEL = "cl100k_base"


def get_llm(config):
    if config['llm_model_provider'] == 'groq':
        return Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY, max_retries=2, retry_on_rate_limit=True) # Number of retry attempts
    elif config['llm_model_provider'] == 'gemini':
        return CustomGoogleGenAI(
            model=config['llm_model'],
            api_key=GEMINI_API_KEY, 
            max_retries=2,  # Number of retry attempts
            retry_on_rate_limit=True
        )
    elif config['llm_model_provider'] == 'ollama':
        return Ollama(model=OLLAMA_MODEL, request_timeout=120.0, context_window=8192, )
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


def get_nodes_from_documents(documents, embed_model, tokenizer, max_tokens):
    md_docs = []
    txt_docs = []
    py_docs = []
    for doc in documents:
        if Path(doc.metadata['file_name']).suffix.lower() in ('.md', '.mdx'):
            md_docs.append(doc)
        elif Path(doc.metadata['file_name']).suffix.lower() == '.txt':
            txt_docs.append(doc)
        elif Path(doc.metadata['file_name']).suffix.lower() == '.py':
            py_docs.append(doc)
        else:
            raise ValueError(f"Filetype not supported.")
    
    nodes = []
    if len(md_docs) > 0:
        # md_node_parser = MarkdownNodeParser(chunk_size=512, chunk_overlap=32)
        md_node_parser = CustomMarkdownNodeParser(max_tokens=max_tokens, split_pattern=r'\*\*(.*?)\*\*', tokenizer=tokenizer)
        md_nodes = md_node_parser.get_nodes_from_documents(md_docs)
        nodes += md_nodes
    if len(txt_docs) > 0:
        txt_node_parser = SemanticSplitterNodeParser(buffer_size=3, embed_model=embed_model)
        txt_nodes = txt_node_parser.get_nodes_from_documents(txt_docs)
        nodes += txt_nodes
    if len(py_docs) > 0:
        py_node_parser = CodeSplitter(language='python', chunk_lines=70, chunk_lines_overlap=10, max_chars=3000)
        py_nodes = py_node_parser.get_nodes_from_documents(py_docs)
        nodes += py_nodes
    
    return nodes



def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False
    

async def process_nodes_with_ratelimit(nodes, transformations, rate_limit=-1, max_retries=2, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # decrease rate limit as per the no of llm based transformers
    llm_ops = 0
    for transformation in transformations:
        if hasattr(transformation, 'llm'):
            llm_ops += 1
    if llm_ops > 1 and rate_limit > 0:
        rate_limit = rate_limit // llm_ops
    
    transformed_nodes = []
    batch_size = rate_limit if rate_limit > 0 else 100
    total_batches = math.ceil(len(nodes) / batch_size)
    for batch_idx in range(total_batches):
        for retries in range(max_retries):
            try:
                batch_start = time.time()
                # Get current batch of nodes
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(nodes)) 
                batch_nodes = nodes[start_idx:end_idx]
                logger.info(f"batch {batch_idx} of {total_batches}, {len(batch_nodes)} nodes, start {batch_start}")

                pipeline = IngestionPipeline(
                    transformations=transformations
                )
                batch_nodes = await pipeline.arun(nodes=batch_nodes, in_place=False, show_progress=False)
                # batch_nodes = pipeline.run(nodes=batch_nodes, in_place=False, num_workers=4)


                batch_end = time.time()
                elapsed_time = batch_end - batch_start
                logger.info(f"batch {batch_idx} finished in {elapsed_time:.1f}s")
                if rate_limit > 0 and elapsed_time < 60:
                    await asyncio.sleep(60 - elapsed_time)

                transformed_nodes.extend(batch_nodes)
                break
            except Exception as e:
                if retries == max_retries - 1:
                    logger.error(f"batch {batch_idx} Errored {e}, aborting. ")
                    print('!!! Aborting due to too many errors')
                    logger.error("!!! Aborting due to too many errors")
                    return None

                # print(f"batch {batch_idx} Errored, restarting", e)
                logger.error(f"batch {batch_idx} Errored, restarting. "+ str(e))
                await asyncio.sleep(60 * (retries+1)**2)
                            
    return transformed_nodes


async def create_vector_store():

    LLM_MODEL_PROVIDER = 'gemini'  # choose from ['ollama', 'gemini', 'groq']
    LLM_MODEL = DEFAULT_GEMINI_MODEL
    RATE_LIMIT = 15 # LLM req/min, -1 if no limit
    
    EMBEDDING_PROVIDER = 'GoogleGenAIEmbedding'  # choose from ['HuggingFaceEmbedding', 'GoogleGenAIEmbedding', etc]
    EMBEDDING_MODEL = DEFAULT_GEMINI_EMBEDDING_MODEL

    TOKENIZER_PROVIDER = 'gemini'  # chose from ['gemini', 'huggingface', 'tiktoken' etc]
    TOKENIZER_MODEL_NAME = DEFAULT_GEMINI_TOKENIZER_MODEL  # choose from ['models/gemini-2.0-flash', 'meta-llama/Meta-Llama-3.1-70B-Instruct', 'cl100k_base' etc]
    MAX_NODE_TOKENS = 2000

    FILE_TYPES = ['.md', '.mdx']
    # Select Extractors - To add metadata to each node. Some of them may use many LLM calls. Use only if needed.
    METADATA_EXTRACTORS = ['CustomDocumentContextExtractor']  # choose from ['TitleExtractor', 'SummaryExtractor', 'KeywordExtractor', 'CustomDocumentContextExtractor' etc]

    CHROMADB_PATH = (Path(PROCESSED_DIR) / 'chromadb').as_posix()
    CHROMADB_COLLECTION = 'contextual'

    DOCSTORE_PATH = (Path(PROCESSED_DIR) / 'docstore.json').as_posix()
    CONFIG_PATH = (Path(PROCESSED_DIR) / 'config.json').as_posix()
    
    config = {
        'llm_model_provider': LLM_MODEL_PROVIDER,
        'llm_model': DEFAULT_GEMINI_MODEL,
        'rate_limit': RATE_LIMIT,
        'input_dir': INPUT_DIR,
        'output_dir': PROCESSED_DIR,
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

    logging.basicConfig(filename=(Path(PROCESSED_DIR) / "rag.log").as_posix(), level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


    llm_logger = create_custom_logger('LLMlogger', (Path(PROCESSED_DIR) / "llm_events.log").as_posix())
    setup_llm_logs(llm_logger, Settings, show_text=False, short_inputs=False, short_outputs=False )

    
    is_jupyter_notebook = is_notebook()

    # # for jupyter notebooks - to fix event loop issue
    if is_jupyter_notebook:
        import nest_asyncio
        nest_asyncio.apply()

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
    documents = SimpleDirectoryReader(input_dir=INPUT_DIR, exclude=EXCLUDE_FILES, recursive=True, filename_as_id=True,
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
        doc_batch_size = 5
        all_nodes = []
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
            nodes = await process_nodes_with_ratelimit(nodes=nodes, transformations=transformations , rate_limit = RATE_LIMIT, logger=logger)
            
            if nodes is None:
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
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as fp:
                existing_config = json.load(fp)
                config = existing_config
        
        run_numbers = [int(k.split('_')[1]) for k in config.keys() if k.startswith('run_') and k.endswith('_time')]
        next_run = 1 if not run_numbers else max(run_numbers) + 1
        
        # Store the current run info
        config[f'run_{next_run}_time'] = datetime.now(timezone.utc).isoformat()
        config[f'run_{next_run}_nodes'] = len(all_nodes)
        with open(CONFIG_PATH, 'w') as fp:
            json.dump(config, fp)
        
        if 'process' in locals():
            process.terminate()
            process.wait()


def main():
    try:
        Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    except FileExistsError as e:
        print(f'Processed_data directory {PROCESSED_DIR} already exists. Please delete that or specify another directory.')
        exit(1)
    print('output dir', PROCESSED_DIR)

    
    t3 = time.time()
    asyncio.run(create_vector_store())
    t4 = time.time()
    print('time', round(t4-t3))

if __name__ == '__main__':
    # Fix for Windows (needed if freezing the application) - to support pipeline num_workers
    multiprocessing.freeze_support()  # Ensures proper multiprocessing behavior on Windows
    main()  # Runs the actual logic
    
