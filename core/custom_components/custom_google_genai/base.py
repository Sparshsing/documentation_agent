"""Google's hosted Gemini API."""

import os
import typing
import time
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import asyncio
from datetime import datetime, timedelta
import threading

import llama_index.core.instrumentation as instrument
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    achat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection, Model
from llama_index.core.prompts import PromptTemplate
from llama_index.core.program.utils import FlexibleModel
from .utils import (
    chat_from_gemini_response,
    chat_message_to_gemini,
    convert_schema_to_function_declaration,
    prepare_chat_params,
)

import google.genai
import google.auth
import google.genai.types as types

dispatcher = instrument.get_dispatcher(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"
DEAFULT_RPM_LIMIT = 10
logger = logging.getLogger(__name__)

# Retry constants
DEFAULT_MAX_RETRIES = 3
RETRY_DELAY_RATE_LIMIT_SEC = 60
RETRY_DELAY_OTHER_SEC = 3

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class VertexAIConfig(typing.TypedDict):
    credentials: Optional[google.auth.credentials.Credentials] = None
    project: Optional[str] = None
    location: Optional[str] = None


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class CustomGoogleGenAI(FunctionCallingLLM):
    """
    Google GenAI LLM.

    Examples:
        `pip install llama-index-llms-google-genai`

        ```python
        from llama_index.llms.google_genai import GoogleGenAI

        llm = GoogleGenAI(model="gemini-2.0-flash", api_key="YOUR_API_KEY")
        resp = llm.complete("Write a poem about a magic backpack")
        print(resp)
        ```
    """

    model: str = Field(default=DEFAULT_MODEL, description="The Gemini model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=2.0,
    )
    context_window: Optional[int] = Field(
        default=None,
        description="The context window of the model. If not provided, the default context window 200000 will be used.",
    )
    is_function_calling_model: bool = Field(
        default=True, description="Whether the model is a function calling model."
    )

    _max_tokens: int = PrivateAttr()
    _client: google.genai.Client = PrivateAttr()
    _generation_config: types.GenerateContentConfigDict = PrivateAttr()
    _model_meta: types.Model = PrivateAttr()
    _rate_limits: Dict[str, int] = PrivateAttr()
    _token_limits: Dict[str, int] = PrivateAttr()
    _request_timestamps: Dict[str, List[float]] = PrivateAttr()
    _token_usage: Dict[str, List[Tuple[float, int]]] = PrivateAttr()
    _rate_limit_semaphore: asyncio.Semaphore = PrivateAttr()
    _sync_rate_limit_semaphore: threading.Semaphore = PrivateAttr()
    _timestamp_lock: asyncio.Lock = PrivateAttr()
    _sync_timestamp_lock: threading.Lock = PrivateAttr()
    _max_retries: int = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        context_window: Optional[int] = None,
        vertexai_config: Optional[VertexAIConfig] = None,
        http_options: Optional[types.HttpOptions] = None,
        debug_config: Optional[google.genai.client.DebugConfig] = None,
        generation_config: Optional[types.GenerateContentConfig] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_function_calling_model: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **kwargs: Any,
    ):
        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        api_key = api_key or os.getenv("GOOGLE_API_KEY", None)
        vertexai = (
            vertexai_config is not None
            or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false") != "false"
        )
        project = (vertexai_config or {}).get("project") or os.getenv(
            "GOOGLE_CLOUD_PROJECT", None
        )
        location = (vertexai_config or {}).get("location") or os.getenv(
            "GOOGLE_CLOUD_LOCATION", None
        )

        config_params: Dict[str, Any] = {
            "api_key": api_key,
        }

        if vertexai_config is not None:
            config_params.update(vertexai_config)
            config_params["api_key"] = None
            config_params["vertexai"] = True
        elif vertexai:
            config_params["project"] = project
            config_params["location"] = location
            config_params["api_key"] = None
            config_params["vertexai"] = True

        if http_options:
            config_params["http_options"] = http_options

        if debug_config:
            config_params["debug_config"] = debug_config

        client = google.genai.Client(**config_params)
        model_meta = client.models.get(model=model)

        super().__init__(
            model=model,
            temperature=temperature,
            context_window=context_window,
            callback_manager=callback_manager,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )

        self._max_retries = max_retries
        self.model = model
        self._client = client
        self._model_meta = model_meta
        # store this as a dict and not as a pydantic model so we can more easily
        # merge it later
        self._generation_config = (
            generation_config.model_dump()
            if generation_config
            else types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ).model_dump()
        )
        self._max_tokens = (
            max_tokens or model_meta.output_token_limit or DEFAULT_NUM_OUTPUTS
        )
        
        # Initialize rate limiting parameters
        self._rate_limits = {
            "gemini-2.5-pro-preview-03-25": 5,  # requests per minute
            "gemini-2.5-flash-preview-04-17": 10,
            "gemini-2.0-flash": 15,
            "gemini-2.0-flash-lite": 30,
            'gemma-3-27b-it': 5
        }
        self._token_limits = {
            "gemini-2.5-pro-preview-03-25": 250000,  # tokens per minute
            "gemini-2.5-flash-preview-04-17": 250000,
            "gemini-2.0-flash": 1000000,
            "gemini-2.0-flash-lite": 1000000,
            'gemma-3-27b-it': 15000
        }
        self._request_timestamps = {model: [] for model in self._rate_limits}
        self._token_usage = {model: [] for model in self._rate_limits}
        # Initialize Semaphores based on RPM limit
        rpm_limit = self._rate_limits.get(model, DEAFULT_RPM_LIMIT) # Default if model not specified
        self._rate_limit_semaphore = asyncio.Semaphore(rpm_limit)
        self._sync_rate_limit_semaphore = threading.Semaphore(rpm_limit)
        # Initialize locks
        self._timestamp_lock = asyncio.Lock()
        self._sync_timestamp_lock = threading.Lock()

    def _check_and_update_rate_limit_window(self, model: str, estimated_tokens: int) -> Tuple[bool, float, str]:
        """Checks RPM/TPM limits and updates timestamps if possible.
        
        This method assumes the caller holds the appropriate lock (_timestamp_lock or _sync_timestamp_lock).
        
        Returns:
            Tuple[bool, float, str]: (proceed, sleep_time, log_message)
        """
        current_time = time.time()
        # --- Clean up old timestamps --- 
        self._request_timestamps[model] = [
            ts for ts in self._request_timestamps[model]
            if current_time - ts < 60
        ]
        self._token_usage[model] = [
            (ts, tkn) for ts, tkn in self._token_usage[model]
            if current_time - ts < 60
        ]

        # --- Check Limits --- 
        rpm_limit = self._rate_limits.get(model, DEAFULT_RPM_LIMIT)
        request_limit_hit = len(self._request_timestamps[model]) >= rpm_limit

        token_limit = self._token_limits.get(model)
        token_limit_hit = False
        total_recent_tokens = 0
        if token_limit is not None:
            total_recent_tokens = sum(tkn for _, tkn in self._token_usage[model])
            token_limit_hit = total_recent_tokens + estimated_tokens >= token_limit
        
        # --- Decide Action --- 
        if not request_limit_hit and not token_limit_hit:
            # Limits OK, record request timestamp
            self._request_timestamps[model].append(time.time())
            logger.debug(f"Rate limit window check passed for {model}.", )
            return True, 0.0, ""
        else:
            # Limit Hit: Calculate required sleep time
            sleep_time = 1.1 # Min sleep if limit hit but no timestamps
            log_msg = ""

            if request_limit_hit:
                oldest_req_ts = min(self._request_timestamps[model])
                sleep_time_req = max(0, 60 - (current_time - oldest_req_ts)) + 1.1
                sleep_time = max(sleep_time, sleep_time_req)
                log_msg += f" Request limit hit (count: {len(self._request_timestamps[model])}/{rpm_limit}, oldest: {current_time - oldest_req_ts:.1f}s ago)."
            
            if token_limit_hit:
                if self._token_usage[model]:
                    oldest_token_ts = min(ts for ts, _ in self._token_usage[model])
                    sleep_time_token = max(0, 60 - (current_time - oldest_token_ts)) + 1.1
                    sleep_time = max(sleep_time, sleep_time_token)
                    log_msg += f" Token limit hit (current: {total_recent_tokens}, needs: {estimated_tokens}, limit: {token_limit}, oldest: {current_time - oldest_token_ts:.1f}s ago)."
                elif token_limit is not None and estimated_tokens >= token_limit:
                    log_msg += f" Single request tokens ({estimated_tokens}) exceed limit ({token_limit})."
                    sleep_time = max(sleep_time, 1.1)
            
            return False, sleep_time, log_msg

    async def _wait_for_rate_limit_window(self, model: str, estimated_tokens: int) -> None:
        """Waits if necessary to comply with RPM and TPM rolling windows, using a lock.
        """
        while True:
            async with self._timestamp_lock:
                proceed, sleep_time, log_msg = self._check_and_update_rate_limit_window(model, estimated_tokens)
            if proceed:
                break # Exit the wait loop
            else:
                logger.debug(f"[ASYNC] Rate limit window requires waiting {sleep_time:.2f}s for {model}.{log_msg}")
                await asyncio.sleep(sleep_time)
                # Loop continues to re-acquire lock and re-check after sleep

    def _wait_for_rate_limit_window_sync(self, model: str, estimated_tokens: int) -> None:
        """Synchronous version of waiting for rate limit window, using a lock."""
        while True:
            with self._sync_timestamp_lock:
                proceed, sleep_time, log_msg = self._check_and_update_rate_limit_window(model, estimated_tokens)

            if proceed:
                break
            else:
                logger.debug(f"[SYNC] Rate limit window requires waiting {sleep_time:.2f}s for {model}.{log_msg}")
                time.sleep(sleep_time)

    def _record_token_usage(self, model: str, tokens_used: int) -> None:
        """Record token usage for a request."""
        current_time = time.time()
        self._token_usage[model].append((current_time, tokens_used))

    @classmethod
    def class_name(cls) -> str:
        return "GenAI"

    @property
    def metadata(self) -> LLMMetadata:
        if self.context_window is None:
            base = self._model_meta.input_token_limit or 200000
            total_tokens = base + self._max_tokens
        else:
            total_tokens = self.context_window

        return LLMMetadata(
            context_window=total_tokens,
            num_output=self._max_tokens,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=self.is_function_calling_model,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        chat_fn = chat_to_completion_decorator(self.chat)
        return chat_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        chat_fn = achat_to_completion_decorator(self.achat)
        return await chat_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        chat_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return chat_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        chat_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await chat_fn(prompt, **kwargs)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Estimate token usage
        estimated_tokens = sum(len(msg.content.split()) * 1.3 for msg in messages if msg.content)
        estimated_tokens = int(estimated_tokens)
        
        self._sync_rate_limit_semaphore.acquire()
        try:
            # Wait for time window if necessary
            self._wait_for_rate_limit_window_sync(self.model, estimated_tokens)
            
            generation_config = {
                **(self._generation_config or {}),
                **kwargs.pop("generation_config", {}),
            }
            params = {**kwargs, "generation_config": generation_config}
            next_msg, chat_kwargs = prepare_chat_params(self.model, messages, **params)
            
            retries = 0
            last_exception = None
            while retries < self._max_retries:
                try:
                    # print('Sending SYNC Gemini request', str(messages[0].content)[:15], datetime.now().strftime("%H:%M:%S.%f"))
                    chat = self._client.chats.create(**chat_kwargs)
                    response = chat.send_message(next_msg.parts)

                    # Record actual token usage
                    actual_tokens = estimated_tokens # Start with estimated
                    if response.text: # Add response tokens if available
                        actual_tokens += len(response.text.split()) * 1.3
                    if response.usage_metadata and response.usage_metadata.total_token_count is not None:
                        actual_tokens = response.usage_metadata.total_token_count
                    else:
                        # If no usage_metadata, refine based on prompt and candidate tokens if possible
                        prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                        candidate_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                        if prompt_tokens > 0 or candidate_tokens > 0: # use if available
                             actual_tokens = prompt_tokens + candidate_tokens
                    self._record_token_usage(self.model, int(actual_tokens))
                    return chat_from_gemini_response(response)
                except google.genai.errors.ClientError as e:
                    if not(e.code == 429 or '429' in str(e)):
                        raise e
                    last_exception = e
                    retries += 1
                    logger.warning(
                        f"Rate limit exceeded for model {self.model} during chat. "
                        f"Retrying ({retries}/{self._max_retries}) after {RETRY_DELAY_RATE_LIMIT_SEC}s. Error: {e}"
                    )
                    time.sleep(RETRY_DELAY_RATE_LIMIT_SEC)
                except Exception as e:
                    last_exception = e
                    retries += 1
                    logger.warning(
                        f"Error during chat API call for model {self.model}. "
                        f"Retrying ({retries}/{self._max_retries}) after {RETRY_DELAY_OTHER_SEC}s. Error: {e}"
                    )
                    time.sleep(RETRY_DELAY_OTHER_SEC)
            
            logger.error(f"Max retries ({self._max_retries}) reached for chat with model {self.model}. Last error: {last_exception}")
            raise last_exception # type: ignore

        finally:
            # Ensure semaphore is always released
            self._sync_rate_limit_semaphore.release()

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Estimate token usage
        estimated_tokens = sum(len(msg.content.split()) * 1.3 for msg in messages if msg.content)
        estimated_tokens = int(estimated_tokens)

        async with self._rate_limit_semaphore:
            # Wait for time window if necessary (acquires lock internally)
            await self._wait_for_rate_limit_window(self.model, estimated_tokens)

            generation_config = {
                **(self._generation_config or {}),
                **kwargs.pop("generation_config", {}),
            }
            params = {**kwargs, "generation_config": generation_config}
            next_msg, chat_kwargs = prepare_chat_params(self.model, messages, **params)

            retries = 0
            last_exception = None
            while retries < self._max_retries:
                try:
                    # print('Sending ASYNC Gemini request', str(messages[0].content)[:15],datetime.now().strftime("%H:%M:%S.%f"))
                    chat = self._client.aio.chats.create(**chat_kwargs)
                    response = await chat.send_message(next_msg.parts)

                    # Record token usage
                    actual_tokens = estimated_tokens # Start with estimated
                    if response.text: # Add response tokens if available
                        actual_tokens += len(response.text.split()) * 1.3
                    if response.usage_metadata and response.usage_metadata.total_token_count is not None:
                        actual_tokens = response.usage_metadata.total_token_count
                    else:
                        # If no usage_metadata, refine based on prompt and candidate tokens if possible
                        prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                        candidate_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                        if prompt_tokens > 0 or candidate_tokens > 0: # use if available
                             actual_tokens = prompt_tokens + candidate_tokens
                    self._record_token_usage(self.model, int(actual_tokens))
                    
                    return chat_from_gemini_response(response)
                except google.genai.errors.ClientError as e:
                    if not(e.code == 429 or '429' in str(e)):
                        raise e
                    last_exception = e
                    retries += 1
                    logger.warning(
                        f"[ASYNC] Rate limit exceeded for model {self.model} during achat. "
                        f"Retrying ({retries}/{self._max_retries}) after {RETRY_DELAY_RATE_LIMIT_SEC}s. Error: {e}"
                    )
                    await asyncio.sleep(RETRY_DELAY_RATE_LIMIT_SEC)
                except Exception as e:
                    last_exception = e
                    retries += 1
                    logger.warning(
                        f"[ASYNC] Error during achat API call for model {self.model}. "
                        f"Retrying ({retries}/{self._max_retries}) after {RETRY_DELAY_OTHER_SEC}s. Error: {e}"
                    )
                    await asyncio.sleep(RETRY_DELAY_OTHER_SEC)
            
            logger.error(f"[ASYNC] Max retries ({self._max_retries}) reached for achat with model {self.model}. Last error: {last_exception}")
            raise last_exception # type: ignore
        # Semaphore is automatically released by 'async with'

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        params = {**kwargs, "generation_config": generation_config}
        next_msg, chat_kwargs = prepare_chat_params(self.model, messages, **params)
        chat = self._client.chats.create(**chat_kwargs)
        response = chat.send_message_stream(next_msg.parts)

        def gen() -> ChatResponseGen:
            content = ""
            existing_tool_calls = []
            for r in response:
                if not r.candidates:
                    continue

                top_candidate = r.candidates[0]
                content_delta = top_candidate.content.parts[0].text
                if content_delta:
                    content += content_delta
                llama_resp = chat_from_gemini_response(r)
                existing_tool_calls.extend(
                    llama_resp.message.additional_kwargs.get("tool_calls", [])
                )
                llama_resp.delta = content_delta
                llama_resp.message.content = content
                llama_resp.message.additional_kwargs["tool_calls"] = existing_tool_calls
                yield llama_resp

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        params = {**kwargs, "generation_config": generation_config}
        next_msg, chat_kwargs = prepare_chat_params(self.model, messages, **params)
        chat = self._client.aio.chats.create(**chat_kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            existing_tool_calls = []
            async for r in await chat.send_message_stream(next_msg.parts):
                if candidates := r.candidates:
                    if not candidates:
                        continue

                    top_candidate = candidates[0]
                    if response_content := top_candidate.content:
                        if parts := response_content.parts:
                            content_delta = parts[0].text
                            if content_delta:
                                content += content_delta
                            llama_resp = chat_from_gemini_response(r)
                            existing_tool_calls.extend(
                                llama_resp.message.additional_kwargs.get(
                                    "tool_calls", []
                                )
                            )
                            llama_resp.delta = content_delta
                            llama_resp.message.content = content
                            llama_resp.message.additional_kwargs[
                                "tool_calls"
                            ] = existing_tool_calls
                            yield llama_resp

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Union[str, dict] = "auto",
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict and call the tool."""
        if tool_choice == "auto":
            tool_mode = types.FunctionCallingConfigMode.AUTO
        elif tool_choice == "none":
            tool_mode = types.FunctionCallingConfigMode.NONE
        else:
            tool_mode = types.FunctionCallingConfigMode.ANY

        function_calling_config = types.FunctionCallingConfig(mode=tool_mode)

        if tool_choice not in ["auto", "none"]:
            if isinstance(tool_choice, dict):
                raise ValueError("Gemini does not support tool_choice as a dict")

            # assume that the user wants a tool call to be made
            # if the tool choice is not in the list of tools, then we will make a tool call to all tools
            # otherwise, we will make a tool call to the tool choice
            tool_names = [tool.metadata.name for tool in tools if tool.metadata.name]
            if tool_choice not in tool_names:
                function_calling_config.allowed_function_names = tool_names
            else:
                function_calling_config.allowed_function_names = [tool_choice]

        tool_config = types.ToolConfig(
            function_calling_config=function_calling_config,
        )

        tool_declarations = []
        for tool in tools:
            if tool.metadata.fn_schema:
                function_declaration = convert_schema_to_function_declaration(
                    self._client, tool
                )
                tool_declarations.append(function_declaration)

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": (
                [types.Tool(function_declarations=tool_declarations)]
                if tool_declarations
                else None
            ),
            "tool_config": tool_config,
            **kwargs,
        }

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.name,
                    tool_name=tool_call.name,
                    tool_kwargs=dict(tool_call.args),
                )
            )

        return tool_selections

    @dispatcher.span
    def structured_predict_without_function_calling(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}

        messages = prompt.format_messages(**prompt_args)
        response = self._client.models.generate_content(
            model=self.model,
            contents=list(map(chat_message_to_gemini, messages)),
            **{
                **llm_kwargs,
                **{
                    "config": {
                        "response_mime_type": "application/json",
                        "response_schema": output_cls,
                    }
                },
            },
        )

        if isinstance(response.parsed, BaseModel):
            return response.parsed
        else:
            raise ValueError("Response is not a BaseModel")

    @dispatcher.span
    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}

        if self.is_function_calling_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in llm_kwargs
                else llm_kwargs["tool_choice"]
            )

        return super().structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    @dispatcher.span
    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        """Structured predict."""
        llm_kwargs = llm_kwargs or {}

        if self.is_function_calling_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in llm_kwargs
                else llm_kwargs["tool_choice"]
            )

        return await super().astructured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    @dispatcher.span
    def stream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[Model, FlexibleModel], None, None]:
        """Stream structured predict."""
        llm_kwargs = llm_kwargs or {}

        if self.is_function_calling_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in llm_kwargs
                else llm_kwargs["tool_choice"]
            )
        return super().stream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )

    @dispatcher.span
    async def astream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Union[Model, FlexibleModel], None]:
        """Stream structured predict."""
        llm_kwargs = llm_kwargs or {}

        if self.is_function_calling_model:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in llm_kwargs
                else llm_kwargs["tool_choice"]
            )
        return await super().astream_structured_predict(
            output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
        )
