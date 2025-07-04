from bs4 import BeautifulSoup
from datetime import datetime
from types import SimpleNamespace
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import OpenAI
from supabase import create_client
import os
import json
from curl_cffi import requests

load_dotenv()


#llm = OpenAI(
#    api_key=os.environ.get("OPENAI_API_KEY"),
#    )
#
#supabase = create_client("https://mprqvvcbtqnltjnyznsd.supabase.co", os.environ.get("SUPABASE_KEY"))


FEEDTYPES = {
    'NASDAQ': 'API'
}
FEEDS = {
    'NASDAQ': {
        'AAPL': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=AAPL|STOCKS&offset=0&limit=10&fallback=true',
        'MSFT': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=MSFT|STOCKS&offset=0&limit=10&fallback=true',
        'GOOGL': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=GOOGL|STOCKS&offset=0&limit=10&fallback=true',
        'AMZN': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=AMZN|STOCKS&offset=0&limit=10&fallback=true',
        'TSLA': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=TSLA|STOCKS&offset=0&limit=10&fallback=true'
    }
    #'NASDAQ': {
    #    'TSLA': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=TSLA|STOCKS&offset=0&limit=10&fallback=true'
    #}
}

def get_message(ticker, text):
    #user_prompt = user_prompt_base + f"\nTicker symbol: {ticker}\n\nText: {text}"
    user_prompt = f"\nTicker symbol: {ticker}\n\nText: {text}"
    return [
        #{"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def parse_api_feed(response):
    result = {
        "channel": {
            "items": []
        }
    }
    result_str = json.dumps(result)

    if response.content:
        result_ = json.loads(response.content)
        items = result_["data"]["rows"]

        for item in items:
            title = item["title"]
            link = "https://www.nasdaq.com" + item["url"]
            result["channel"]["items"].append({
                "links": [{"content": link}],
                "title": {"content": title},
                "pub_date": {"content":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            })
        result_str = json.dumps(result)
    return json.loads(result_str, object_hook=lambda d: SimpleNamespace(**d))

def feed_parser(source, feed_url):
    parsed_feed = None
    if FEEDTYPES[source] == 'API':
        response = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}, impersonate="chrome")
        parsed_feed = parse_api_feed(response)
    return parsed_feed

def get_article_content(url, source):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}, impersonate="chrome")
    if response.status_code == 200:
        content_lines = []
        bs = BeautifulSoup(response.content, "html.parser")
        
        if source == 'NASDAQ':
            body_content = bs.find('div', class_='body__content')
            if body_content:
                content_tags = body_content.find_all(['p','table'])
                for p in content_tags:
                    if len(p.contents) == 1 and p.contents[0].name == 'a':
                        continue
                    links = p.find_all('a')
                    skip_ad = False
                    if links:
                        for link in links:
                            if 'rel' in link.attrs and 'nofollow' in link.attrs['rel']:
                                skip_ad = True
                    if skip_ad:
                        continue
                    if p.name == 'table':
                        rows = p.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 1:
                                row_str = ' : '.join([cell.text.strip() for cell in cells])
                                content_lines.append(row_str)
                    elif p.text.strip():
                        content_lines.append(p.text.strip())
                if content_lines:
                    return '\n'.join(content_lines)
                else:
                    return "Content not available"
    return None

def get_all_articles():
    all_articles = []
    for source, feed_dict in FEEDS.items():
        for symbol, feed_url in feed_dict.items():
            parsed_feed = feed_parser(source, feed_url)
            if parsed_feed is None:
                print(f"Failed to parse feed for {source} {symbol}")
                continue
            entries = [[source, entry, symbol, "", False] for entry in parsed_feed.channel.items]
            all_articles.extend(entries)
            #print(all_articles)
            #import pdb; pdb.set_trace()

    print('Starting analysing articles...')
    visited = []
    current = {}
    export_data = []
    ticker_articles = {} 
    for article in all_articles:
        current[article[2]] = False
    for article in all_articles:
        link = article[1].links[0].content
        if link in visited:
            continue
        visited.append(link)
        article_content = get_article_content(link, article[0])
        # Collect the article data
        ticker = article[2]
        data = {
            "title": article[1].title.content,
            "link": link,
            "ticker": ticker,
            "content": article_content
        }
        if ticker not in ticker_articles:
            ticker_articles[ticker] = []
        ticker_articles[ticker].append(data)
    
    today = datetime.now().strftime('%Y%m%d')
    base_dir = 'all_news_json'
    output_dir = os.path.join(base_dir, f'news_json_{today}')
    os.makedirs(output_dir, exist_ok=True)

    for ticker, articles in ticker_articles.items():
        filename = f'news_{ticker}_{today}.json'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    get_all_articles()


