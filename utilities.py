
import logging
import os
import time
import math
from pathlib import Path
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import (
    LLMCompletionEndEvent,
    LLMChatEndEvent,
)
from llama_index.core.instrumentation.events.embedding import EmbeddingEndEvent

import nbformat
import nbconvert
from nbconvert import PythonExporter, MarkdownExporter
import asyncio
from llama_index.core.ingestion import IngestionPipeline
from vertexai.preview import tokenization




logger = logging.getLogger(__name__)


# custom logger
def create_custom_logger(logger_name, logfile_path):
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    handler = logging.FileHandler(logfile_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def setup_llm_logs(llm_logger, settings, show_text=False, short_inputs=False, short_outputs=False):
    # Debugging - Observability - LLM calls

    class ModelEventHandler(BaseEventHandler):
        @classmethod
        def class_name(cls) -> str:
            """Class name."""
            return "ModelEventHandler"

        def handle(self, event) -> None:
            """Logic for handling event."""
            if isinstance(event, LLMCompletionEndEvent):
                llm_logger.info(f"LLM Prompt length: {len(settings.tokenizer(event.prompt))}")
                llm_logger.info(f"LLM Completion length: {len(settings.tokenizer(event.response.text))}")
                llm_logger.info('------------------------------------')
            elif isinstance(event, LLMChatEndEvent):
                messages_str = "\n".join([str(x) for x in event.messages])
                if short_inputs:
                    messages_str = f"{messages_str[:50]} ... {messages_str[:50]}"
                response_msg = str(event.response.message)
                if short_outputs:
                    response_msg = response_msg[:50] + "..."
                llm_logger.info(f"LLM Input Messages length: {len(settings.tokenizer(messages_str))}")
                llm_logger.info(f"LLM Response Length: {len(settings.tokenizer(str(event.response.message)))}")
                if show_text:
                    llm_logger.info(f"LLM Input Messages : {messages_str}")
                    llm_logger.info(f"LLM Response: {response_msg}")
                    llm_logger.info('------------------------------------')
                
            elif isinstance(event, EmbeddingEndEvent):
                pass
                # logger.info(f"Embedding {len(event.chunks)} text chunks")
    
    from llama_index.core.instrumentation import get_dispatcher
    # root dispatcher
    root_dispatcher = get_dispatcher()
    # register event handler
    root_dispatcher.add_event_handler(ModelEventHandler())


def convert_notebook_to_python(notebook_path, output_path=None, debug=False):
    """
    Converts a Jupyter Notebook to a Python script.

    Args:
        notebook_path (str): The path to the Jupyter Notebook file (.ipynb).
        output_path (str, optional): The path to save the resulting Python script.
                                    If None, the Python script will be saved in the same
                                    directory as the notebook, with the same name but
                                    the extension changed to '.py'. Defaults to None.
        debug (bool): Whether to print debug information
    """
    try:
        # 1. Read the Jupyter Notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 2. Create a PythonExporter
        exporter = PythonExporter()

        # 3. Convert the Notebook to Python code
        (body, resources) = exporter.from_notebook_node(nb)

        # 4. Determine the output path
        if output_path is None:
            output_path = notebook_path.replace('.ipynb', '.py')

        # 5. Write the Python code to a file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body)
        if debug:
            print(f"Notebook '{notebook_path}' successfully converted to Python script '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Notebook file not found at path '{notebook_path}'")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")


def convert_notebook_to_markdown(notebook_path, output_path=None, include_outputs=False, max_output_length=1000, debug=False):
    """
    Converts a Jupyter Notebook to a Markdown file.

    Args:
        notebook_path (str): The path to the Jupyter Notebook file (.ipynb).
        output_path (str, optional): The path to save the resulting Markdown file.
                                   If None, saves in same directory with .md extension.
        include_outputs (bool): Whether to include cell outputs in the markdown file.
        max_output_length (int): Maximum length for each output. Outputs longer than this will be trimmed.
                               Set to -1 to keep full output. Only applies if include_outputs is True.
        debug (bool): Whether to print debug information.
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Configure the exporter
        exporter = MarkdownExporter()
        if not include_outputs:
            exporter.exclude_output = True
            exporter.exclude_output_stdin = True

        else:
            class TrimOutputPreprocessor(nbconvert.preprocessors.Preprocessor):
                def preprocess_cell(self, cell, resources, cell_index):
                    if 'outputs' in cell and max_output_length > 0:
                        for output in cell.outputs:
                            if 'text' in output:
                                if len(output.text) > max_output_length:
                                    output.text = (output.text[:max_output_length] + 
                                                 '\n... [Output truncated] ...')
                            elif 'data' in output:
                                for mime_type, data in output.data.items():
                                    if isinstance(data, str) and len(data) > max_output_length:
                                        output.data[mime_type] = (data[:max_output_length] + 
                                                                '\n... [Output truncated] ...')
                    return cell, resources

            exporter.register_preprocessor(TrimOutputPreprocessor, enabled=True)
            exporter.exclude_output = False
            exporter.exclude_output_stdin = False


        (body, resources) = exporter.from_notebook_node(nb)

        if output_path is None:
            output_path = notebook_path.replace('.ipynb', '.md')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body)
        if debug:
            print(f"Notebook '{notebook_path}' successfully converted to Markdown '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Notebook file not found at path '{notebook_path}'")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")


def convert_notebooks(input_dir, output_dir=None, format='md', include_outputs=False, max_output_length=1000, debug=False):
    """
    Convert all .ipynb files in input_dir to specified format
    
    Args:
        input_dir (str): Input directory containing notebooks
        output_dir (str, optional): Output directory for converted files.
            If None, converted file will be written in the same directory as the input file.
        format (str): Output format - 'py' or 'md'
        max_output_length (int): Maximum length for each output in markdown files.
            Outputs longer than this will be trimmed. Set to -1 to keep full output.
        include_outputs (bool): Whether to include cell outputs in the converted files
        debug (bool): Whether to print debug information
    """
    input_path = Path(input_dir)
    if output_dir is None:
        output_base = Path(input_dir)
    else:
        output_base = Path(output_dir)
        output_base.mkdir(parents=True, exist_ok=True)
    
    for notebook_path in input_path.rglob('*.ipynb'):
        rel_path = notebook_path.relative_to(input_path)
        output_path = output_base / rel_path.with_suffix(f'.{format}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'py':
            convert_notebook_to_python(str(notebook_path), str(output_path), debug=debug)
        elif format == 'md':
            convert_notebook_to_markdown(str(notebook_path), str(output_path), 
                                      include_outputs=include_outputs, 
                                      max_output_length=max_output_length, debug=debug)


def get_large_files(directory, min_size_kb=20, extensions=('.txt', '.md', '.mdx')):
    large_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                size_kb = os.path.getsize(file_path) / 1024
                if size_kb > min_size_kb:
                    large_files.append((file_path,size_kb) )
    return large_files




# pip install --upgrade google-cloud-aiplatform[tokenization]
# class GeminiTokenizer:
#     def __init__(self, model='gemini-1.5-flash-002'):
#         self._tokenizer = tokenization.get_tokenizer_for_model(model)  # 2.0 not availabale, but gives same count
    
#     def encode(self, text: str):
#         tokens = self._tokenizer.compute_tokens(text)
#         token_ids = tokens.tokens_info[0].token_ids
#         return token_ids
    
class GoogleGenAIDummyTokensizer:
    """
    This class is used to tokenize text using the Google GenAI API. Dummy tokens are returned based on token count.
    """
    def __init__(self, llm):
        self._client = llm._client
        self.model = llm.model
    
    def encode(self, text: str):
        token_count = self._client.models.count_tokens(
            model=self.model,
            contents=text,
        )
        token_count = token_count.total_tokens
        tokens = [i for i in range(1, token_count+1)]
        return tokens
    

class HuggingfaceTokenizer:
    def __init__(self, model='meta-llama/Meta-Llama-3.1-70B-Instruct'):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model)
    def encode(self, text: str):
        return self._tokenizer.encode(text)