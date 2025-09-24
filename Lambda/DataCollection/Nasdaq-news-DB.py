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

# Configure logging
# Init logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [Nasdaq-collect] %(message)s'
)
logger = logging.getLogger()

# DEBUG from .env
DEBUG = os.getenv('DEBUG', 'False')
if DEBUG.lower() == "debug":
    logger.setLevel(logging.DEBUG)
    logging.debug('Debug logging enabled')
else:
    logger.setLevel(logging.INFO)
    logging.info('Info logging enabled')  


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

def normalize_to_std(dt_str: str) -> str | None:
    if not dt_str:
        return None
    try:
        iso = dt_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    try:
        s = dt_str.strip()
        if ', ' in s:
            s = s.split(', ', 1)[1]
        s = s.replace('—','-').replace('–','-').replace('‒','-').replace('−','-')
        dt2 = datetime.strptime(s, "%m/%d/%Y - %H:%M")
        return dt2.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    
def parse_dates_from_soup(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    date_mod = date_pub = None
    # JSON-LD
    for s in soup.find_all('script', type='application/ld+json'):
        try:
            raw = (s.string or s.text or "").strip()
            if not raw:
                continue
            j = json.loads(raw)
            items = j if isinstance(j, list) else [j]
            expanded = []
            for obj in items:
                if isinstance(obj, dict) and isinstance(obj.get("@graph"), list):
                    expanded.extend(obj["@graph"])
                else:
                    expanded.append(obj)
            for obj in expanded:
                if not isinstance(obj, dict):
                    continue
                if not date_mod and obj.get("dateModified"):
                    date_mod = str(obj["dateModified"]).strip()
                if not date_pub and obj.get("datePublished"):
                    date_pub = str(obj["datePublished"]).strip()
        except Exception:
            continue
    # Meta tags
    if not date_mod:
        m = soup.find('meta', attrs={'property':'article:modified_time'}) or soup.find('meta', attrs={'name':'dateModified'})
        if m and m.get('content'):
            date_mod = m['content'].strip()
    if not date_pub:
        m = soup.find('meta', attrs={'property':'article:published_time'}) or soup.find('meta', attrs={'name':'datePublished'})
        if m and m.get('content'):
            date_pub = m['content'].strip()
    return date_mod, date_pub

def parse_api_feed(response, source: str, symbol: str):
    result = {
        "channel": {
            "items": []
        }
    }
    result_str = json.dumps(result)

    if response is None or getattr(response, "status_code", 0) != 200:
        logging.warning(
            f"Feed HTTP error: source={source} symbol={symbol} status={getattr(response,'status_code',None)} url={getattr(response,'url',None)}"
        )
        return SimpleNamespace(channel=SimpleNamespace(items=[]))

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
    else:
        logging.warning(
            f"Empty feed content: source={source} symbol={symbol} url={getattr(response,'url',None)}"
        )
    return json.loads(result_str, object_hook=lambda d: SimpleNamespace(**d))

def feed_parser(source, symbol, feed_url):
    parsed_feed = None
    if FEEDTYPES[source] == 'API':
        try:
            response = requests.get(
                feed_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"},
                impersonate="chrome",
                timeout=10
            )
        except Exception as e:
            logging.error(f"Feed request failed: source={source} symbol={symbol} url={feed_url} err={e}")
            return None
        parsed_feed = parse_api_feed(response, source, symbol)
    return parsed_feed

def get_article_content_and_date(url: str, source: str) -> tuple[str | None, str]:
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"},
        impersonate="chrome"
    )
    if resp.status_code != 200:
        return None, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    bs = BeautifulSoup(resp.content, "html.parser")

    # Extract article content based on source
    content_lines = []
    if source == 'NASDAQ':
        body_content = bs.find('div', class_='body__content')
        if body_content:
            for p in body_content.find_all(['p','table']):
                if len(getattr(p, "contents", [])) == 1 and getattr(p.contents[0], "name", None) == 'a':
                    continue
                links = p.find_all('a')
                if any(('rel' in a.attrs and 'nofollow' in a.attrs.get('rel', [])) for a in links):
                    continue
                if p.name == 'table':
                    rows = p.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if cells:
                            content_lines.append(' : '.join([cell.get_text(strip=True) for cell in cells]))
                else:
                    text = p.get_text(strip=True)
                    if text:
                        content_lines.append(text)
    content = '\n'.join(content_lines) if content_lines else "Content not available"

    # chosen modified date first, then published date, if none use current UTC time
    dmod, dpub = parse_dates_from_soup(bs)
    chosen = dmod or dpub
    pub_date = normalize_to_std(chosen) or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return content, pub_date

def run(event, context):
    all_articles = []
    #collect all articles from all feeds
    for source, feed_dict in FEEDS.items():
        for symbol, feed_url in feed_dict.items():
            parsed_feed = feed_parser(source, symbol, feed_url)
            if parsed_feed is None:
                logging.warning(f"Failed to parse feed for {source} {symbol}")
                continue
            entries = [[source, entry, symbol, "", False] for entry in parsed_feed.channel.items]
            all_articles.extend(entries)

    logging.info('Starting analysing articles...')
    visited = []
    current = {}
    export_data = [] # list of dicts to save to Postgres

    for article in all_articles:
        current[article[2]] = False

    for t_index, ticker in enumerate(tk_list):
        # Process articles for this ticker
        ticker_articles = [a for a in all_articles if a[2] == ticker]
        ticker_data = [] # list of dicts for this ticker
        
        for article in ticker_articles:
            link = article[1].links[0].content
            if link in visited:
                continue
            visited.append(link)
            article_content, pub_date = get_article_content_and_date(link, article[0])
            # Collect the article data
            ticker_symbol = article[2]
            data = {
                "date": pub_date,
                "source": SOURCE,
                "ticker": ticker_symbol,
                "title": article[1].title.content,
                "link": link,
                "content": article_content,
            }
            ticker_data.append(data)
            export_data.append(data)

        # Save ticker data to Postgres
        if ticker_data:
            save_news_to_postgres(ticker_data, TABLE_NAME)
            logging.info(f"Saved {len(ticker_data)} news for {ticker} to database")

        # Save progress    
        progress_date = datetime.now(timezone.utc).date().isoformat()
        save_progress(progress_date, t_index, len(tk_list))
        logging.info(f"Processed {t_index+1}/{len(tk_list)}: {ticker}")

        # Save aggregated news items to JSON file
        #if ticker_data:
            #save_news_items(ticker_data, ticker, run_date, SOURCE)
    logging.info(f"Total processed news: {len(export_data)} for {len(tk_list)} tickers")
    return export_data

if __name__ == '__main__':
    run(None, None)
    


