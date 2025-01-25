import logging
from pathlib import Path
import asyncio
import re
from typing import Union

from topic_extractor_gemini import extract_topics, Topic, TopicList
from webpage_crawler import get_markdown


DATA_DIR = 'data'
MAX_CONCURRENT = 5

# Configure logger
Path('logs').mkdir(exist_ok=True)

# Get the logger
logger = logging.getLogger('download_logger')
logger.setLevel(logging.DEBUG)

# Create the file handler and formatter
file_handler = logging.FileHandler('logs/download.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

# Add file handler to the logger
logger.addHandler(file_handler)
logger.propagate = False # do not propagate the logs to root logger


# Create a semaphore to limit concurrency
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


def clean_filename(name: str) -> str:
    """Remove special characters from a string to make it safe for file/directory names."""
    # Replace common problematic characters with underscores
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove any other non-alphanumeric characters except dashes and underscores
    cleaned = re.sub(r'[^\w\-_.]', '_', cleaned)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')


async def download_topic(topic: Topic, target_dir: Path) -> Union[bool, None]:
    """Download and save the content of a topic to the specified directory.
    
    Args:
        topic: The Topic object containing the URL and metadata
        target_dir: The directory path where to save the downloaded content
        
    Returns:
        bool: True if download successful, False if failed, None if no URL
    """
    logger.debug((topic.supertopic or "") + " -> " + topic.title)
    if topic.url:
        logger.debug("Crawling " + topic.url)
    else:
        return None
    try:
        async with semaphore:
            page_markdown = await get_markdown(topic.url)
            topic_file = target_dir / clean_filename(topic.supertopic) / f"{clean_filename(topic.title)}.md" if topic.supertopic else target_dir / f"{clean_filename(topic.title)}.md"
            topic_file.parent.mkdir(parents=True, exist_ok=True)
            with open(topic_file, 'w', encoding='utf-8') as f:
                f.write(page_markdown)
            logger.debug(f"Saved {topic_file}")
            return True
    except Exception as err:
        logger.error(f"Failed to download {topic.url}", err)
        return False


async def download_docs(title: str, url: str):

    data_dir = Path(DATA_DIR)
    target_dir = data_dir / title

    try:
        # Check and create directories
        if not data_dir.exists():
            logger.debug(f"Creating data directory: {data_dir}")
            data_dir.mkdir()

        if not target_dir.exists():
            logger.info(f"Creating project directory: {target_dir}")
            target_dir.mkdir()
        else:
            logger.info(f"Directory already exists: {target_dir}")

    except PermissionError as e:
        logger.error(f"Permission denied while creating directory: {e}")
    except OSError as e:
        logger.error(f"Failed to create directory: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)

    # extract the topics list from the URL and save the markdown of each topic
    topics = await extract_topics(url)
    
    # add the home page as a topic
    home = Topic(title='_home_' + title, url=url, supertopic=None)
    topics.insert(0, home)
    
    # save topics list
    with open(target_dir / 'topics.json', 'w', encoding='utf-8') as f:
        f.write(TopicList(topics=topics).model_dump_json())

    results = await asyncio.gather(*[download_topic(topic, target_dir) for topic in topics])
    failed_topics = [topic for topic, result in zip(topics, results) if result==False]
    if len(failed_topics) > 0:
        logger.debug("Failed topics:")
        for topic in failed_topics:
            logger.debug(topic.title)
    logger.info("Topic Extraction Complete")


if __name__ == "__main__":
    title = 'crawl4ai'
    url = 'https://docs.crawl4ai.com/'
    asyncio.run(download_docs(title=title, url=url))
