from typing import List, Optional
from dotenv import load_dotenv
import asyncio
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_verbose

from webpage_crawler import get_markdown


load_dotenv()

MODEL = "llama-3.3-70b-versatile"


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
    Extract topics from the given documentation webpage url using LangChain.
    
    Args:
        url (str): The input webpage url to analyze
        
    Returns:
        List[Topic]: A list of extracted topics
    """

    set_verbose(True)
    
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

    # get webpage content
    webpage_content = await get_markdown(url)

    # Initialize the language model
    llm = ChatGroq(temperature=0, model_name=MODEL, max_tokens=8000)

    # Create the chain
    chain = prompt | llm | parser

    result = chain.invoke({'webpage_content': webpage_content})
    
    set_verbose(False)
    
    return result.topics


# Example usage
if __name__ == "__main__":
    try:
        topics = asyncio.run(extract_topics("https://docs.crawl4ai.com)/"))
        for topic in topics:
            print(f"Title: {topic.title}")
            print(f"URL: {topic.url}")
            if topic.supertopic:
                print(f"Supertopic: {topic.supertopic}")
            print("---")
            
    except Exception as e:
        print(f"Error extracting topics: {e}")
        raise e
