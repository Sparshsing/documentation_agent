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
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <url> <output_file>")
        sys.exit(1)
    
    url = sys.argv[1]
    output_file = sys.argv[2]
    
    markdown = asyncio.run(get_markdown(url))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

