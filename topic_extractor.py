import os
import asyncio
import json
from typing import List, Optional
from urllib.parse import urljoin

from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from google import genai
from google.genai import types
import litellm
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_verbose

from webpage_crawler import get_cleaned_html


load_dotenv()

LITELLM_MODEL = "gemini/gemini-2.5-flash-preview-05-20"
# LITELLM_MODEL = "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"


GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']

LANGCHAIN_MODEL = "llama-3.3-70b-versatile"
LANGCHAIN_MODEL_API_KEY = os.environ['GROQ_API_KEY']


class Topic(BaseModel):
    title: str = Field(description="The title of the topic")
    url: Optional[str] = Field(description="The URL of the topic")
    supertopic: Optional[str] = Field(description="The parent topic")


class TopicList(BaseModel):
    topics: List[Topic] = Field(description="List of documentation topics extracted from the text")


def replace_relative_urls(html_content, base_url):
    """Function to replace relative URLs with absolute URLs"""
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup.find_all(href=True):
        tag['href'] = urljoin(base_url, tag['href'])
    return str(soup)


def extract_topics(webpage_content: str) -> List[Topic]:
    """
    Extract topics from the given documentation webpage.
    
    Args:
        webpage_content (str): Simplified html of the webpage
        
    Returns:
        List[Topic]: A list of extracted topics
    """

    messages = [
        {
            "role": "user",
            "content": f"""
                This is the webpage of a library documentation:

                Webpage content:
                {webpage_content}

                You are a smart HTML parser and your job is to extract a structured list of documentation topics from the provided HTML content of a documentation page.
                Task:
                From the HTML of the left-hand menu, extract a structured list of topics (with hierarchy if applicable). Use the tag structure (like <ul>, <li>, <a>, or role attributes) to infer topics and subtopics.
            """
        }
    ]

    response = litellm.completion(
        model= LITELLM_MODEL,
        messages=messages,
        response_format=TopicList,
        max_completion_tokens=8192
    )

    result = TopicList.model_validate_json(response.choices[0].message.content)
    return result.topics


def extract_topics_using_gemini(webpage_content: str) -> List[Topic]:
    """
    Extract topics from the given documentation webpage url using Gemini (schema in model config and schema in prompt).
    
    Args:
        webpage_content (str): Simplified html of the webpage
        
    Returns:
        List[Topic]: A list of extracted topics
    """

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    This is the webpage of a library documentation:

    Webpage content:
    {webpage_content}

    You are a smart HTML parser and your job is to extract a structured list of documentation topics from the provided HTML content of a documentation page.
    Task:
    From the HTML of the left-hand menu, extract a structured list of topics (with hierarchy if applicable). Use the tag structure (like <ul>, <li>, <a>, or role attributes) to infer topics and subtopics.
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents = prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=TopicList,
        )
    )

    return response.parsed.topics

    ## Provide schema in Prompt
    # prompt = f"""
    # This is the webpage of a library documentation:

    # Webpage content:
    # {webpage_content}

    # You are a smart HTML parser and your job is to extract a structured list of documentation topics from the provided HTML content of a documentation page.
    # Task:
    # From the HTML of the left-hand menu, extract a structured list of topics (with hierarchy if applicable). Use the tag structure (like <ul>, <li>, <a>, or role attributes) to infer topics and subtopics.
        
    # The response should follow this schema:
    # {json.dumps(TopicList.model_json_schema(), indent=2)}
    # """
    
    # # Generate response with JSON configuration
    # response = client.models.generate_content(
    #     model=GEMINI_MODEL,
    #     contents=prompt,
    #     config=types.GenerateContentConfig(
    #         max_output_tokens=8192,
    #         response_mime_type="application/json"
    #     )
    # )

    # # Parse response and validate with Pydantic
    # result = TopicList.model_validate_json(response.text)
    # return result.topics


async def extract_topics_using_langchain(url: str) -> List[Topic]:
    """
    Extract topics from the given documentation webpage url using LangChain.
    
    Args:
        url (str): The input webpage url to analyze
        
    Returns:
        List[Topic]: A list of extracted topics
    """

    set_verbose(True)

    # get webpage content
    webpage_content = await get_cleaned_html(url)

    # replace relative urls with absolute urls
    webpage_content = replace_relative_urls(webpage_content, url)
    
    # Initialize the parser
    parser = PydanticOutputParser(pydantic_object=TopicList)

    # Create the prompt template
    template = """
    This is the webpage of a library documentation:

    Webpage content:
    {webpage_content}

    Find out the list of topics in the webpage and their corresponding links. The topics can be nested. 
    Find out their parent topic if available.

    Format the output as a JSON list of topics matching this schema:
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Initialize the language model
    llm = ChatGroq(temperature=0, model_name=LANGCHAIN_MODEL, api_key=LANGCHAIN_MODEL_API_KEY , max_tokens=8000)

    # Create the chain
    chain = prompt | llm | parser

    result = chain.invoke({'webpage_content': webpage_content})
    
    set_verbose(False)
    
    return result.topics


# Example usage
if __name__ == "__main__":
    try:
        url = "https://ai.google.dev/gemini-api/docs#python"

        # get webpage content
        webpage_content = asyncio.run(get_cleaned_html(url))
        # replace relative urls with absolute urls
        webpage_content = replace_relative_urls(webpage_content, url)

        topics = extract_topics(webpage_content)
        for topic in topics:
            print(f"Title: {topic.title}")
            print(f"URL: {topic.url}")
            if topic.supertopic:
                print(f"Supertopic: {topic.supertopic}")
            print("---")
        
        with open("topics.json", "w") as f:
            f.write(topics.model_dump_json(indent=4))
            
    except Exception as e:
        print(f"Error extracting topics: {e}")
        raise e
