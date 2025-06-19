import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.cache_context import CacheMode
from bs4 import BeautifulSoup
from urllib.parse import urljoin

async def get_markdown(url: str) -> str:
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(
                url=url, cache_mode=CacheMode.DISABLED
            )
            if result.url != url:
                print(f"Redirected to {result.url}")
            if not result.success:
                raise Exception(result.error_message)
            if result.status_code == 404:
                raise Exception(f"url not found")
            return result.markdown
        except Exception as err:
            print("Crawler failed for", url)
            raise err
        

async def get_cleaned_html(url: str) -> str:
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(
                url=url,
            )
            if result.url != url:
                print(f"Redirected to {result.url}")
            if not result.success:
                raise Exception(result.error_message)
            if result.status_code == 404:
                raise Exception(f"url not found")
            return result.cleaned_html
        except Exception as err:
            print("Crawler failed for", url)
            raise err


def replace_relative_urls(html_content, base_url):
    """Function to replace relative URLs with absolute URLs"""
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup.find_all(href=True):
        tag['href'] = urljoin(base_url, tag['href'])
    return str(soup)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python webpage_crawler.py <url> <output_file>")
        sys.exit(1)
    
    url = sys.argv[1]
    output_file = sys.argv[2]
    
    markdown = asyncio.run(get_markdown(url))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

    html = asyncio.run(get_cleaned_html(url))
    with open(output_file+".html", 'w', encoding='utf-8') as f:
        f.write(html)

