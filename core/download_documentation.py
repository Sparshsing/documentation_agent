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
import json

from topic_extractor import extract_topics, Topic
from webpage_crawler import get_markdown, get_cleaned_html, replace_relative_urls


MAX_CONCURRENT = 5

# Get the logger
logger = logging.getLogger('documentation_agent.download')

# Create a semaphore to limit concurrency
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


def clean_filename(name: str) -> str:
    """Remove special characters from a string to make it safe for file/directory names."""
    # Replace common problematic characters with underscores
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove any other non-alphanumeric characters except dashes and underscores
    cleaned = re.sub(r'[^\w\-_.]', '_', cleaned)
    return cleaned.strip()


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


def setup_directory(dir: Path) -> None:
    try:
        # Check and create directories
        if not dir.exists():
            logger.debug(f"Creating directory: {dir}")
            dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {dir}")

    except PermissionError as e:
        logger.error(f"Permission denied while creating directory: {e}")
    except OSError as e:
        logger.error(f"Failed to create directory: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)


def save_all_content_to_file(target_dir: str, output_file_path: str) -> None:
    """Save all markdown content from target directory to a single combined file.
    
    Args:
        target_dir (str): Directory containing the markdown files to combine.
        output_file_path (str): Path of file with all contents.
    """
    concatenated_docs = ""
    target_dir = Path(target_dir)
    # Match both .md and .mdx files
    for file_path in target_dir.rglob("*"):
        if file_path.suffix in {".md", ".mdx"} and file_path.is_file():
            relative_path = file_path.relative_to(target_dir).as_posix()
            concatenated_docs += f"\n\n=== File: {relative_path} ===\n\n"
            concatenated_docs += file_path.read_text(encoding='utf-8')
            concatenated_docs += "\n\n=== End File ===\n"

    all_content_path = Path(output_file_path)
    all_content_path.parent.mkdir(parents=True, exist_ok=True)

    all_content_path.write_text(concatenated_docs, encoding='utf-8')
    logger.info(f"All contents saved to file {all_content_path}: {get_token_count(concatenated_docs)} tokens")


async def download_url_as_markdown(url: str, filepath: Path) -> Union[bool, None]:
    """Download and save content as markdownfrom a URL to a specified file path.
    
    Args:
        url (str): The URL to download the markdown from.
        filepath (Path): The file path (including directories) where to save the downloaded content.
    
    Returns:
        bool: True if download successful, False if failed, None if no URL
    """
    logger.debug(f"Downloading {url}")
    if not url:
        return None
    try:
        async with semaphore:
            page_markdown = await get_markdown(url)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page_markdown)
            
            logger.debug(f"Saved {filepath}: {get_token_count(page_markdown)} tokens")
            return True
    except Exception as err:
        logger.error(f"Failed to download {url}", err)
        return False
    

async def download_docs_from_website(website_url: str, download_path: str):

    # add trailing slash to the URL if not present
    if not website_url.endswith('/'):
        website_url += '/'

    target_dir = Path(download_path)
    setup_directory(target_dir)

    # extract the topics list from the URL and save the markdown of each topic
    webpage_content = await get_cleaned_html(website_url)
    # replace relative urls with absolute urls
    webpage_content = replace_relative_urls(webpage_content, website_url)

    topics = extract_topics(webpage_content)
    logger.info("Topic Extraction Complete")
    
    # add the home page as a topic
    home = Topic(title='__root__', url=website_url, supertopic=None)
    topics.insert(0, home)
    
    # save topics list
    topics_path = Path(download_path) / 'topics.json'
    with open(topics_path, 'w', encoding='utf-8') as f:
        json.dump([topic.model_dump() for topic in topics], f, ensure_ascii=False, indent=2)
    
    # Prepare download tasks by computing file paths first
    tasks = []
    for topic in topics:
        # sanitize title and build filename
        filename = f"{clean_filename(topic.title)}.md"
        # include supertopic directory if present
        if topic.supertopic:
            subdir = clean_filename(topic.supertopic)
            filepath = target_dir / subdir / filename
        else:
            filepath = target_dir / filename
        tasks.append(download_url_as_markdown(topic.url, filepath))
    results = await asyncio.gather(*tasks)

    failed_topics = [topic for topic, result in zip(topics, results) if result==False]
    if len(failed_topics) > 0:
        logger.error(f"Failed topics: {[topic.title for topic in failed_topics]}")
    logger.info("Documents download complete")


def download_folder_from_github(repo: str, folder: str, output_dir: str, branch: str = 'main'):
    """Download a specific folder from a GitHub repository.
    
    Args:
        repo (str): GitHub repository in format 'owner/repo'.
        folder (str): Folder path within the repository.
        output_dir (str): Local directory to save files.
        branch (str, optional): Branch name. Defaults to 'main'.
    """
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
    

def download_docs_from_github_repo(repo_identifier: str, repo_folder: str, download_path: str):
    """Download documentation from a GitHub repository folder.
    
    Args:
        repo_identifier (str): GitHub repository in format 'owner/repo'.
        repo_folder (str): Folder path containing documentation files.
        download_path (str): download folder path.
    """
    target_dir = Path(download_path)
    setup_directory(target_dir)

    download_folder_from_github(repo_identifier, folder=repo_folder, output_dir=target_dir.as_posix())
    print(f"Repo {repo_identifier} Folder '{repo_folder}' downloaded to '{target_dir.as_posix()}'")
    logger.info("Documents download complete")


if __name__ == "__main__":
    url = 'https://ai.google.dev/api?lang=python'
    download_path = 'data/google_genai/api'

    asyncio.run(download_docs_from_website(url, download_path))
    save_all_content_to_file(download_path, os.path.join(download_path, '__all_docs__.md'))
