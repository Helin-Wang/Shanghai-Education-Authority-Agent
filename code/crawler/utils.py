from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, DefaultMarkdownGenerator
import time
from datetime import datetime
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import json
import re
import asyncio
from tqdm.asyncio import tqdm
from bs4 import BeautifulSoup
from markitdown import MarkItDown
import io
from typing import List
from tqdm import tqdm
BASE_URL = "https://www.shmeea.edu.cn"

# TODO: Do not use css selector to crawl content, use html to crawl content; and then use beautifulsoup to extract <div class=trout-region-content/class=Article_content>


# TODO
# Current version: 
# (1) only crawl HTML; 
# (2) only crawl content related to 2023&2024
async def crawl_contentpage(content_list):
    """
    Crawl a list of content pages and return the extracted data
    """
    # filter
    content_list = [item for item in content_list if item['year'] in ['2023', '2024']]
    crawled_data = []
    batch_size = 5
    for i in tqdm(range(0, len(content_list), batch_size), desc="Batches"):
        batch = content_list[i:i + batch_size]
        tasks = [crawl_single_contentpage(item) for item in batch]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]
        crawled_data.extend(results)
    return crawled_data

async def crawl_single_contentpage(content_src_item: dict, max_retries: int = 3, retry_delay: float = 1.0):
    """
    Crawl a single content page and return the extracted data
    
    Args:
        content_src_item: Dictionary containing the content source information
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
    """
    cfg = CrawlerRunConfig(
        js_code="window.scrollTo(0, document.body.scrollHeight);",
        wait_for="body",
        page_timeout=30000,
        verbose=True
    )
    url = content_src_item['link']
    
    if not url.endswith('.html'):
        return None
    
    for attempt in range(max_retries + 1):  # +1 because we want to include the initial attempt
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url, config=cfg)
                if result.success:
                    # Use BeautifulSoup to extract content based on URL domain
                    soup = BeautifulSoup(result.html, 'html.parser')
                    
                    if 'edu.sh.gov.cn' in url:
                        # Extract div with class trout-region-content
                        content_div = soup.find('div', class_='trout-region-content')
                    else:
                        # Extract div with class Article_content
                        content_div = soup.find('div', class_='Article_content')
                    
                    if content_div:
                        content_src_item['content'] = str(content_div)
                        content_src_item['crawl_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        return content_src_item
                    else:
                        print(f"⚠️ No content div found for {url}")
                        # Fallback: save raw HTML for debugging
                        content_src_item['content'] = result.html
                        content_src_item['crawl_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        return content_src_item
                else:
                    if attempt < max_retries:
                        print(f"⚠️ Attempt {attempt + 1} failed for {url}, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(f"❌ Failed to crawl {url} after {max_retries + 1} attempts")
                        return None
        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️ Exception on attempt {attempt + 1} for {url}: {e}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                print(f"❌ Error crawling {url} after {max_retries + 1} attempts: {e}")
                return None
    
    return None


async def crawl_all_subpage_list_pages(base_url: str, type1: str, type2: str, max_pages: int = 30):
    """
    Crawl all subpage list pages for a specific category
    """
    urls = []
    for i in range(1, max_pages + 1):
        url = generate_subpage_urls(base_url, i)
        urls.append(url)
        
    # Crawl pages in batches to avoid overwhelming the server
    batch_size = 5
    
    crawled_data = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        tasks = [
            crawl_single_subpage_list_page(type1, type2, url)
            for url in batch
        ]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]
        crawled_data.extend(results)
        
        # Early termination
        if len(results) < 10:
            break
    
    return [item for sublist in crawled_data for item in sublist]

async def crawl_single_subpage_list_page(type1: str, type2: str, url: str):
    """
    Crawl a single subpage list page and return the extracted data
    """
    css_schema = {
        "name": "Subpage",
        "baseSelector": "ul.pageList li",  # Target specific list items within pageList
        "fields": [
            {"name": "title", "selector": "a", "type": "text"},
            {"name": "link", "selector": "a", "type": "attribute", "attribute": "href"},
            {"name": "published_date", "selector": "span.listTime", "type": "text"},
        ]
    }
    css_strategy = JsonCssExtractionStrategy(schema=css_schema)
    
    cfg = CrawlerRunConfig(
        js_code="window.scrollTo(0, document.body.scrollHeight);",
        wait_for="body",
        markdown_generator=DefaultMarkdownGenerator(),
        page_timeout=30000,
        verbose=True,
        extraction_strategy=css_strategy
    )
    async with AsyncWebCrawler(verbose=False) as crawler:
        try:
            result = await crawler.arun(url=url, config=cfg)
            
            if result.success and result.extracted_content:
                extracted_content = json.loads(result.extracted_content)
                
                for item in extracted_content:
                    if '.html' in item['link'] and item['link'].startswith('/'):
                        item['link'] = BASE_URL + item['link']
                    item['year'] = extract_year(item)
                    item['category'] = [type1, type2]
                    
                return extracted_content
                # return {
                #     'url': url,
                #     'metadata': result.metadata,
                #     'title': result.metadata.get('title', 'Unknown Title'),
                #     'html': result.html,
                #     'markdown': result.markdown.raw_markdown,
                #     'html_length': len(result.html),
                #     'markdown_length': len(result.markdown.raw_markdown),
                #     'links_found': len(result.links.get('internal', [])) + len(result.links.get('external', [])),
                #     'crawl_time': time.time(),
                #     'extracted_content': json.loads(result.extracted_content)
                # }
            else:
                print(f"❌ Failed to crawl {url}: {result.error_message}")
                return None
        except Exception as e:
            print(f"❌ Exception crawling {url}: {str(e)}")
            return None
         
def generate_subpage_urls(seed_url: str, page_num: int):
    """Generate subpage URLs: index.html, index_2.html, ..., index_N.html"""
    if page_num == 1:
        return seed_url
    else:
        return seed_url.replace("index.html", f"index_{page_num}.html")
    
def extract_year(crawled_item: dict):
    # Extract from title
    if re.search(r"(\d{4})年", crawled_item['title']):
        return re.search(r"(\d{4})年", crawled_item['title']).group(1)
    else:
        # If not found in title, extract from published date
        return crawled_item['published_date'][:4]

CN_NUMS = "一二三四五六七八九十百千"
def classify_heading_level(text: str) -> int | None:
    s = text.strip()
    
    # Level 1: 一、
    if re.match(rf'^([{CN_NUMS}]+、)', s):
        return 1

    # Level 3: （一）… / (一)… / 
    if re.match(rf'^[（(][{CN_NUMS}]+[)）]', s):
        return 3
    
    # Level 2: 1. / 1、 / （1） / (1)
    if re.match(r'^[（(]?\d+[)）]?\s*[、.．)]\s*', s):
        return 2

    return None

def promote_fake_headings(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for p in soup.find_all("p"):
        text = p.get_text(separator="", strip=True).strip()
        if not text:
            continue
        level = classify_heading_level(text)
        if level:
            # Replace <p> with <hN>
            tag = soup.new_tag(f'h{min(level,6)}')
            tag.string = text
            p.replace_with(tag)
            print(text)
            print(level)

    return str(soup)

def convert_html_to_markdown(html: str) -> str:
    promoted = promote_fake_headings(html)
    html_bytes = promoted.encode('utf-8')
    stream = io.BytesIO(html_bytes)

    md = MarkItDown()
    result = md.convert_stream(stream, file_extension=".html")

    markdown_text = result.text_content
    return markdown_text

def postprocess_content(data: List[dict]) -> List[dict]:
    for item in tqdm(data, desc="Converting HTML to Markdown"):
        item['markdown'] = convert_html_to_markdown(item['content'])
    return data
    