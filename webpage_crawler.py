import asyncio
from crawl4ai import AsyncWebCrawler

async def get_markdown(url: str) -> str:
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(
                url=url,
            )
            return result.markdown
        except Exception as err:
            print(err)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    markdown = asyncio.run(get_markdown(url))
    print(markdown)
