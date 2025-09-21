import asyncio
import json
from crawler.utils import crawl_all_subpage_list_pages, crawl_contentpage, postprocess_content
import yaml
import pandas as pd
import os
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_page", type=int, default=0)
    parser.add_argument("--end_page", type=int, default=30)
    args = parser.parse_args()
    start_page = args.start_page
    end_page = args.end_page
    
    # Check whether the content_src_with_content.json exists
    if os.path.exists("./config/content_src.json"):
        with open("./config/content_src.json", "r", encoding="utf-8") as f:
            content_src = json.load(f)
    else:
        # Load config
        url_src = yaml.safe_load(open("./config/url_src.yaml", "r", encoding="utf-8"))
        max_pages = 30
        content_src = []
        for type1, value in url_src.items():
            for type2, url in value.items():
                data = asyncio.run(crawl_all_subpage_list_pages(url, type1, type2, 30))
                content_src.extend(data)
                
        # Save to json
        with open("./config/content_src.json", "w", encoding="utf-8") as f:
            json.dump(content_src, f, ensure_ascii=False, indent=4)
            
        # Save as a dataframe
        df = pd.DataFrame(content_src)
        df.to_csv("./config/content_src.csv", index=False, encoding="utf-8")
    
    
    # Crawl content pages
    start_page = min(start_page, len(content_src))
    end_page = min(end_page, len(content_src))
    
    data = asyncio.run(crawl_contentpage(content_src[start_page:end_page]))
    data = postprocess_content(data)
    with open(f"../data/v0_content_{start_page}_{end_page}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    