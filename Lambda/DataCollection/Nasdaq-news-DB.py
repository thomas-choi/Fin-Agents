from bs4 import BeautifulSoup
from datetime import datetime, timezone
from types import SimpleNamespace
from dotenv import load_dotenv
import os
import json
import boto3
from curl_cffi import requests
import logging
import pandas as pd
from dataUtil_Pgsql import save_news_to_postgres

load_dotenv()

DEFAULT_TICKERS = os.getenv('DEFAULT_TICKERS', '')
SOURCE          = 'Nasdaq'
df = pd.read_csv(DEFAULT_TICKERS, dtype=str)
tk_list = pd.read_csv(DEFAULT_TICKERS)['Symbol'].tolist()
symbols = df['Symbol'].dropna().unique()
LOCALRUN        = os.getenv('LOCALRUN', 'True').lower() in ('true', '1', 'yes')
S3_BUCKET       = os.getenv('S3_BUCKET', '')
BASE_DIR        = os.getenv('DATA_DIR', './DATA')
TABLE_NAME    = os.getenv('TABLE_NAME', '')

FEEDTYPES = {
    'NASDAQ': 'API'
}
FEEDS = {
    'NASDAQ': {
        symbol: f"https://www.nasdaq.com/api/news/topic/articlebysymbol?q={symbol}|STOCKS&offset=0&limit=10&fallback=true" for symbol in symbols
    }
}

if not LOCALRUN:
    if not S3_BUCKET:
        logging.error("S3_BUCKET environment variable is required when LOCALRUN is False")
        raise ValueError("S3_BUCKET not set")
    s3_client = boto3.client('s3')
    logging.info(f"Configured to save to S3 bucket: {S3_BUCKET}")
    logging.info(f"Configured to save to S3 bucket")
else:
    logging.info("Configured to save to local directory")

def save_s3_json(obj, s3_key):
    json_data = json.dumps(obj, ensure_ascii=False, indent=2)
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json_data.encode('utf-8'),
        ContentType='application/json'
    )
    logging.info(f"Saved to S3: {s3_key}")

def save_json(path: str, obj, overwrite: bool = False):
    if LOCALRUN:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not overwrite:
            logging.info(f"Local file exists, skipping: {path}")
            return
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved to local: {path}")
    else:
        # Convert local path to S3 key (remove BASE_DIR prefix and normalize)
        s3_key = path.replace(BASE_DIR, '').lstrip('/')
        try:
            # Check if object already exists
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            if overwrite:
                save_s3_json(obj, s3_key)
            else:
                # Object exists and overwrite is False, skip saving
                logging.info(f"S3 object exists, skipping: {s3_key}")
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Object does not exist, proceed to upload
                save_s3_json(obj, s3_key)
            else:
                logging.error(f"Error checking S3 object {s3_key}: {e}")
                raise

def save_news_items(news_items: list[dict], ticker: str, run_date: str, source: str):
    """
    Aggregate ALL news for (run_date, ticker) into ONE file:
      DATA/NEWS/yfinance/<run_date>/<ticker>/news.json

    Each object has:
      date (YYYY-MM-DD HH:MM:SS UTC), title, link, publisher, full_text
    """
    out_dir = os.path.join(BASE_DIR, 'NEWS', source, run_date, ticker)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'news.json')
    save_json(out_path, news_items, overwrite=True)
    logging.info(f"Saved aggregated news for {ticker} to {out_path}")

def save_progress(run_date: str, last_processed_index: int, total_tickers: int):
    progress_path = os.path.join(BASE_DIR, 'DBPROGRESS', SOURCE, run_date, 'progress.json')
    progress_data = {
        'last_processed_index': last_processed_index,
        'total_tickers': total_tickers,
        'run_date': run_date
    }
    save_json(progress_path, progress_data, overwrite=True)

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
                "pub_date": {"content":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")},
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

def run(event, context):
    all_articles = []
    for source, feed_dict in FEEDS.items():
        for symbol, feed_url in feed_dict.items():
            parsed_feed = feed_parser(source, feed_url)
            if parsed_feed is None:
                print(f"Failed to parse feed for {source} {symbol}")
                continue
            entries = [[source, entry, symbol, "", False] for entry in parsed_feed.channel.items]
            all_articles.extend(entries)

    print('Starting analysing articles...')
    visited = []
    current = {}
    export_data = []
    for article in all_articles:
        current[article[2]] = False
    for t_index, ticker in enumerate(tk_list):
        # Process articles for this ticker
        ticker_articles = [a for a in all_articles if a[2] == ticker]
        for article in ticker_articles:
            link = article[1].links[0].content
            if link in visited:
                continue
            visited.append(link)
            article_content = get_article_content(link, article[0])
            # Collect the article data
            ticker = article[2]
            pub_date = article[1].pub_date.content
            run_date = datetime.strptime(pub_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            data = {
                "date": pub_date,
                "source": SOURCE,
                "ticker": ticker,
                "title": article[1].title.content,
                "link": link,
                "content": article_content,
            }
            export_data.append(data)
        progress_date = datetime.now(timezone.utc).date().isoformat()
        save_progress(progress_date, t_index, len(tk_list))
        save_news_to_postgres(export_data, TABLE_NAME)
        #save_news_items(export_data, ticker, run_date, 'Nasdaq')
    return export_data

if __name__ == '__main__':
    run(None, None)
    


