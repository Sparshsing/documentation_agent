from dotenv import load_dotenv
import asyncio
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
import google.generativeai as genai

from webpage_crawler import get_markdown


load_dotenv()

MODEL = "gemini-2.0-flash-exp"

genai.configure(api_key=os.environ['GEMINI_API_KEY'])


# Topic model definition
class Topic(BaseModel):
    title: str = Field(description="The title of the topic")
    url: Optional[str] = Field(description="The URL of the topic")
    supertopic: Optional[str] = Field(description="The parent topic")


# Output model for the list of topics
class TopicList(BaseModel):
    topics: List[Topic] = Field(description="List of documentation topics extracted from the text")


async def extract_topics(url: str) -> List[Topic]:
    """
    Extract topics from the given documentation webpage using Gemini (schema in model config).
    
    Args:
        url (str): The input webpage url to analyze
        
    Returns:
        List[Topic]: A list of extracted topics
    """

    # get webpage content
    webpage_content = await get_markdown(url)

    model = genai.GenerativeModel(MODEL)

    response = model.generate_content(
        f"""
        This is the webpage of a library documentation:

        Webpage content:
        {webpage_content}

        Find out the list of topics in the webpage and their corresponding links. The topics can be nested. 
        Find out their parent topic if available.
        """,
        generation_config=genai.GenerationConfig(
            max_output_tokens=8000,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=TopicList.model_json_schema(),
        ),
    )

    result = TopicList.model_validate_json(response.text)
    return result.topics


async def extract_topics(url: str) -> List[Topic]:
    """
    Extract topics from the given documentation webpage url using Gemini (schema in prompt).
    
    Args:
        url (str): The input webpage url to analyze
        
    Returns:
        List[Topic]: A list of extracted topics
    """

    # get webpage content
    webpage_content = await get_markdown(url)

    model = genai.GenerativeModel(MODEL)

    prompt = f"""
    This is the webpage of a library documentation:

    Webpage content:
    {webpage_content}

    Find out the list of topics in the webpage and their corresponding links. The topics can be nested. 
    Find out their parent topic if available.
    Return the topics in JSON format.
    The response should follow this schema:
    {json.dumps(TopicList.model_json_schema(), indent=2)}
    """
    
    # Generate response with JSON configuration
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=8000,
            temperature=0.1,
            response_mime_type="application/json"
        )
    )

    # Parse response and validate with Pydantic
    result = TopicList.model_validate_json(response.text)
    return result.topics



