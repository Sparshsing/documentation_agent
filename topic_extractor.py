import asyncio
import os
import json
from typing import List, Optional
from urllib.parse import urljoin

from pydantic import BaseModel, Field
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from webpage_crawler import get_cleaned_html


from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_verbose


load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash-exp"
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']

LANGCHAIN_MODEL = "llama-3.3-70b-versatile"
LANGCHAIN_MODEL_API_KEY = os.environ['GROQ_API_KEY']



# Topic model definition
class Topic(BaseModel):
    title: str = Field(description="The title of the topic")
    url: Optional[str] = Field(description="The URL of the topic")
    supertopic: Optional[str] = Field(description="The parent topic")


# Output model for the list of topics
class TopicList(BaseModel):
    topics: List[Topic] = Field(description="List of documentation topics extracted from the text")


# Function to replace relative URLs with absolute URLs
def replace_relative_urls(html_content, base_url):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup.find_all(href=True):
        tag['href'] = urljoin(base_url, tag['href'])
    return str(soup)


async def extract_topics_using_gemini(url: str) -> List[Topic]:
    """
    Extract topics from the given documentation webpage using Gemini (schema in model config).
    
    Args:
        url (str): The input webpage url to analyze
        
    Returns:
        List[Topic]: A list of extracted topics
    """

    # get webpage content
    webpage_content = await get_cleaned_html(url)

    # replace relative urls with absolute urls
    webpage_content = replace_relative_urls(webpage_content, url)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

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
            response_schema=TopicList,
        ),
    )


    result = TopicList.model_validate_json(response.text)
    return result.topics


async def extract_topics_using_gemini_fallback(url: str) -> List[Topic]:
    """
    Extract topics from the given documentation webpage url using Gemini (schema in prompt).
    
    Args:
        url (str): The input webpage url to analyze
        
    Returns:
        List[Topic]: A list of extracted topics
    """

    # get webpage content
    webpage_content = await get_cleaned_html(url)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

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
        topics = asyncio.run(extract_topics_using_gemini("https://docs.crawl4ai.com/"))
        for topic in topics:
            print(f"Title: {topic.title}")
            print(f"URL: {topic.url}")
            if topic.supertopic:
                print(f"Supertopic: {topic.supertopic}")
            print("---")
            
    except Exception as e:
        print(f"Error extracting topics: {e}")
        raise e
