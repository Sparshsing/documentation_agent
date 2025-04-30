"""Markdown node parser."""
import re
from typing import Any, List, Optional, Sequence, Callable

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from pydantic import Field, PrivateAttr
from llama_index.core.utils import get_tokenizer


class CustomMarkdownNodeParser(NodeParser):
    """Markdown node parser.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
        embed_model: (BaseEmbedding): embedding model to use
        max_header_level (Optional[int]): maximum header level to split on (1 for #, 2 for ##, etc.)
        max_tokens (Optional[int]): maximum tokens per node. If None, no size-based splitting
        split_pattern (Optional[str]): regex pattern to further split large nodes on.
            If None, they will be split based on paragraphs.
            Example: r'\*\*(.*?)\*\*' for **text**
    """

    _sentence_splitter = PrivateAttr()
    _tokenizer: Callable = PrivateAttr()
    max_tokens: Optional[int] = Field(default=None)
    max_header_level: Optional[int] = Field(default=None)
    split_pattern: Optional[str] = Field(default=None)


    def __init__(
        self,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        tokenizer: Optional[Callable] = None,
        max_header_level: Optional[int] = None,
        max_tokens: Optional[int] = None,
        split_pattern: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        """Initialize with parameters."""
        if max_tokens is not None and tokenizer is None:
            raise ValueError("tokenizer cannot be None if max_tokens is set.")
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            max_header_level=max_header_level,
            max_tokens=max_tokens,
            split_pattern=split_pattern,
            callback_manager=callback_manager,
        )
        if max_tokens:
            self._tokenizer = tokenizer or get_tokenizer()
            self._sentence_splitter = SentenceSplitter(chunk_size=max_tokens, chunk_overlap=0, 
                                                       paragraph_separator="\n\n", tokenizer=self._tokenizer)

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        max_header_level: Optional[int] = None,
        max_tokens: Optional[int] = None,
        split_pattern: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "CustomMarkdownNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        
        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            max_header_level=max_header_level,
            max_tokens=max_tokens,
            split_pattern=split_pattern,
            callback_manager=callback_manager,
        )
        

    # def _split_by_pattern(self, text: str) -> List[str]:
    #     """Split text based on pattern while preserving the matches."""
    #     if not self.split_pattern:
    #         return [text]
        
    #     splits = []
    #     last_end = 0
        
    #     for match in re.finditer(self.split_pattern, text):
    #         # Add text before the pattern
    #         if match.start() > last_end:
    #             splits.append(text[last_end:match.start()].strip())
            
    #         splits.append(match.group(0))  # Keep the full pattern (e.g., **text**)
            
    #         last_end = match.end()
        
    #     # Add remaining text
    #     if last_end < len(text):
    #         splits.append(text[last_end:].strip())
        
    #     return [s for s in splits if s.strip()]

    # def _split_by_pattern(self, text: str) -> List[str]:
    #     """Split text based on pattern while preserving the matches."""
    #     if not self.split_pattern:
    #         return [text]
        
    #     splits = []
    #     last_split = 0
    #     current_text = ""
        
    #     for match in re.finditer(self.split_pattern, text):
    #         # Start a new split when we find a pattern
    #         if current_text:
    #             splits.append(current_text.strip())
    #             current_text = ""
            
    #         # Add everything from last split point up to and including the pattern
    #         current_text = text[last_split:match.end()]
    #         last_split = match.end()
        
    #     # Add any remaining text after the last pattern
    #     if last_split < len(text):
    #         if current_text:
    #             current_text += text[last_split:]
    #         else:
    #             current_text = text[last_split:]
        
    #     if current_text:
    #         splits.append(current_text.strip())
    
    #     return [s for s in splits if s.strip()]

    def _split_by_pattern(self, text: str) -> List[str]:
        """Split text based on pattern while preserving the matches."""
        if not self.split_pattern:
            return [text]
        
        # Find all pattern matches
        matches = list(re.finditer(self.split_pattern, text))
        if not matches:
            return [text]
        
        splits = []
        current_start = 0
        
        # Handle text before first pattern
        if matches[0].start() > 0:
            splits.append(text[0:matches[0].start()].strip())
        
        # Process each pattern match
        for i, match in enumerate(matches):
            # Get the end of the current section
            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            # Extract section from pattern start to next pattern (or end)
            section = text[match.start():next_start].strip()
            if section:
                splits.append(section)
        
        return [s for s in splits if s.strip()]

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from document by splitting on headers."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        markdown_nodes = []
        lines = text.split("\n")
        current_section = ""
        header_stack: List[tuple[int, str]] = []
        code_block = False
        
        for line in lines:
            if line.lstrip().startswith("```"):
                code_block = not code_block
                current_section += line + "\n"
                continue
            
            if not code_block:
                header_match = re.match(r"^(#+)\s(.*)", line)
                if header_match:
                    header_level = len(header_match.group(1))
                    
                    # Skip headers deeper than max_header_level
                    if self.max_header_level and header_level > self.max_header_level:
                        current_section += line + "\n"
                        continue

                    # Process the current section before starting a new one
                    if current_section.strip():
                        new_nodes = self._process_section(
                            current_section.strip(),
                            node,
                            "/".join(h[1] for h in header_stack[:-1]),
                        )
                        markdown_nodes.extend(new_nodes)

                    header_text = header_match.group(2)
                    while header_stack and header_stack[-1][0] >= header_level:
                        header_stack.pop()
                    header_stack.append((header_level, header_text))
                    current_section = "#" * header_level + f" {header_text}\n"
                    continue

            current_section += line + "\n"

        # Process the final section
        if current_section.strip():
            new_nodes = self._process_section(
                current_section.strip(),
                node,
                "/".join(h[1] for h in header_stack[:-1]),
            )
            markdown_nodes.extend(new_nodes)

        return markdown_nodes

    def _process_section(
        self,
        text: str,
        node: BaseNode,
        header_path: str,
    ) -> List[TextNode]:
        """Process a section of text, splitting by pattern and/or size if necessary."""
        # Check if section text needs to be divided further
        if not self.max_tokens or len(self._tokenizer(text)) <= self.max_tokens:
            return [self._build_node_from_split(text, node, header_path)]
        
        # First split by pattern if specified
        pattern_splits = self._split_by_pattern(text)
        
        # Then process each split for size if needed
        final_nodes = []
        for split in pattern_splits:

            chunks = self._sentence_splitter.get_nodes_from_documents([Document(text=split)], show_progress=False)
            for chunk in chunks:
                final_nodes.append(self._build_node_from_split(chunk.text, node, header_path))


            # split_tokens = len(self._tokenizer(split))
            # if not self.max_tokens or split_tokens <= self.max_tokens:
            #     final_nodes.append(self._build_node_from_split(split, node, header_path))
            #     continue

            # # Split by size while preserving paragraphs
            # current_chunk = ""
            # paragraphs = split.split("\n\n")
            
            # for paragraph in paragraphs:
            #     # if len(current_chunk) + len(paragraph) + 2 <= self.max_chars:
            #     if len(self._tokenizer(current_chunk)) + len(self._tokenizer(paragraph)) + 1 <= self.max_tokens:
            #         current_chunk += (paragraph + "\n\n")
            #     else:
            #         if current_chunk:
            #             final_nodes.append(
            #                 self._build_node_from_split(current_chunk.strip(), node, header_path)
            #             )
            #         current_chunk = paragraph + "\n\n"
            
            # if current_chunk:
            #     final_nodes.append(
            #         self._build_node_from_split(current_chunk.strip(), node, header_path)
            #     )

        return final_nodes
    
    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        header_path: str,
    ) -> TextNode:
        """Build node from single text split."""
        node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        if self.include_metadata:
            node.metadata["header_path"] = (
                "/" + header_path + "/" if header_path else "/"
            )

        return node

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse nodes."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes
    