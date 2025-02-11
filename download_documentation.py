import os
import logging
from pathlib import Path
import asyncio
import re
from typing import Union
import zipfile
import io
import tiktoken
from github import Github
import requests

from topic_extractor import extract_topics_using_gemini, Topic, TopicList
from webpage_crawler import get_markdown




DATA_DIR = 'data'
PROCESS_DATA_DIR = 'processed_data'
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


def get_token_count(text: str) -> int:
    """Count the number of tokens in a text using tiktoken cl100k_base encoding.
    Args:
        text (str): The input text to be tokenized.
    Returns:
        int: The number of tokens in the text.
    """

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


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
            
            logger.debug(f"Saved {topic_file}: {get_token_count(page_markdown)} tokens")
            return True
    except Exception as err:
        logger.error(f"Failed to download {topic.url}", err)
        return False


def setup_directory(dir: Path) -> None:
    try:
        # Check and create directories
        if not dir.exists():
            logger.debug(f"Creating directory: {dir}")
            dir.mkdir()
        else:
            logger.info(f"Directory already exists: {dir}")

    except PermissionError as e:
        logger.error(f"Permission denied while creating directory: {e}")
    except OSError as e:
        logger.error(f"Failed to create directory: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)


def read_md_files(directory: Path) -> str:
    content = ""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content += f.read() + "\n\n"
    return content


async def download_docs_from_url(title: str, url: str):

    # add trailing slash to the URL if not present
    if not url.endswith('/'):
        url += '/'

    data_dir = Path(DATA_DIR)
    target_dir = data_dir / title
    setup_directory(data_dir)
    setup_directory(target_dir)

    # extract the topics list from the URL and save the markdown of each topic
    topics = await extract_topics(url)
    logger.info("Topic Extraction Complete")
    
    # add the home page as a topic
    home = Topic(title='_home_' + title, url=url, supertopic=None)
    topics.insert(0, home)
    
    # save topics list
    topics_path = Path(PROCESS_DATA_DIR) / title / 'topics.json'
    with open(topics_path, 'w', encoding='utf-8') as f:
        f.write(TopicList(topics=topics).model_dump_json())
    
    results = await asyncio.gather(*[download_topic(topic, target_dir) for topic in topics])

    # save all content in a single file
    all_content = read_md_files(target_dir)
    all_content_path = Path(PROCESS_DATA_DIR) / title / f"{title}.md"
    Path(all_content_path).parent.mkdir(exist_ok=True)
    with open(all_content_path, 'w', encoding='utf-8') as f:
        f.write(all_content)
    logger.info(f"All Contents saved to file {all_content_path}: {get_token_count(all_content)} tokens")

    failed_topics = [topic for topic, result in zip(topics, results) if result==False]
    if len(failed_topics) > 0:
        logger.debug("Failed topics:")
        for topic in failed_topics:
            logger.debug(topic.title)
    logger.info("Documents download complete")


def download_folder_from_github(repo, folder, output_dir, branch='main'):
    url = f"https://github.com/{repo}/archive/{branch}.zip"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            repo_prefix = f"{repo.split('/')[-1]}-{branch}/"
            folder_prefix = repo_prefix + folder.strip("/") + "/"

            for file in zip_ref.namelist():
                if file.startswith(folder_prefix) and not file.endswith("/"):  # Exclude directories
                    relative_path = file[len(folder_prefix):]  # Remove repo and folder prefix
                    file_output_path = os.path.join(output_dir, relative_path)

                    os.makedirs(os.path.dirname(file_output_path), exist_ok=True)
                    with open(file_output_path, "wb") as f:
                        f.write(zip_ref.read(file))

                    print(f"Downloaded: {file_output_path}")
    else:
        print(f"Failed to download {url}, Status Code: {response.status_code}")
    

def download_docs_from_github_repo(title: str, repo_identifier: str, repo_folder):
    data_dir = Path(DATA_DIR)
    target_dir = data_dir / title
    setup_directory(data_dir)
    setup_directory(target_dir)

    download_folder_from_github(repo_identifier, folder=repo_folder, output_dir=target_dir.as_posix())
    print(f"Repo {repo_identifier} Folder '{repo_folder}' downloaded to '{target_dir.as_posix()}'")
    logger.info("Documents download complete")


if __name__ == "__main__":
    title = 'crawl4ai'
    url = 'https://docs.crawl4ai.com/'

    asyncio.run(download_docs_from_url(title=title, url=url))
