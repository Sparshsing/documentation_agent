import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.cache_context import CacheMode

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

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <url> <output_file>")
        sys.exit(1)
    
    url = sys.argv[1]
    output_file = sys.argv[2]
    
    markdown = asyncio.run(get_markdown(url))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

