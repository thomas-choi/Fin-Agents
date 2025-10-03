import os
import json
import time
import boto3
import random
import logging
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from dataUtil_Pgsql import save_news_to_postgres, save_statistics_to_postgres, save_quarterly_financials_to_postgres

# --------------------------------------------------
# Load environment & configuration
# --------------------------------------------------
load_dotenv()

BASE_DIR         = os.getenv('DATA_DIR', './DATA')
SOURCE           = 'yfinance'
DEFAULT_TICKERS  = os.getenv('DEFAULT_TICKERS', '')
TABLE_NAME       = os.getenv('TABLE_NAME', '')
FIN_TABLE_NAME   = os.getenv('FIN_TABLE_NAME', '')
STAT_TABLE_NAME  = os.getenv('STAT_TABLE_NAME', '')
S3_BUCKET        = os.getenv('S3_BUCKET', '')
LOCALRUN         = os.getenv('LOCALRUN', 'True').lower() in ('true', '1', 'yes')
BATCH_SIZE       = int(os.getenv('BATCH_SIZE', '25'))
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO").upper()
FILTER_RELATED   = os.getenv("YF_FILTER_RELATED", "True").lower() in ('true','1','yes','on')
FETCH_FULL_TEXT  = os.getenv("YF_FETCH_FULLTEXT", "True").lower() in ('true','1','yes','on')
FULLTEXT_SLEEP   = float(os.getenv("YF_FULLTEXT_SLEEP", "0.05"))  # throttle between article fetches
MAX_NEWS_PER_TKR = int(os.getenv("YF_MAX_NEWS_PER_TICKER", "50")) # hard cap per ticker batch
DEDUP_KEY        = os.getenv("YF_DEDUP_KEY", "link")              # 'link' or 'title' or 'link_title'

logging.basicConfig(
    level=logging._nameToLevel.get(LOG_LEVEL, logging.INFO),
    format='%(asctime)s %(levelname)s [yf-collect] %(message)s',
    force=True
)
logger = logging.getLogger("yf-collect")
logger.info(f"Log level set to {LOG_LEVEL}")

# --------------------------------------------------
# S3 client if needed
# --------------------------------------------------
if not LOCALRUN:
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET not set while LOCALRUN=False")
    s3_client = boto3.client('s3')
    logger.info(f"Configured S3 bucket: {S3_BUCKET}")
else:
    logger.info("Local run mode (files saved locally)")

# --------------------------------------------------
# HTTP session
# --------------------------------------------------
session = requests.Session()
session.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/114.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
})

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def df_to_serializable(df: pd.DataFrame) -> dict:
    raw = df.to_dict()
    out = {}
    for col, inner in raw.items():
        date_key = col.strftime('%m/%d/%Y') if hasattr(col, 'strftime') else str(col)
        out[date_key] = {}
        for row_key, val in inner.items():
            out[date_key][str(row_key)] = None if pd.isna(val) else val
    return out

def save_s3_json(obj, s3_key):
    json_data = json.dumps(obj, ensure_ascii=False, indent=2)
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json_data.encode('utf-8'),
        ContentType='application/json'
    )
    logger.info(f"Saved to S3: {s3_key}")

def read_progress(run_date: str) -> dict:
    progress_path = os.path.join(BASE_DIR, 'DBPROGRESS', SOURCE, run_date, 'progress.json')
    if LOCALRUN:
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Read progress failed {progress_path}: {e}")
                return {'last_processed_index': -1, 'total_tickers': 0, 'run_date': run_date}
        return {'last_processed_index': -1, 'total_tickers': 0, 'run_date': run_date}
    else:
        s3_key = progress_path.replace(BASE_DIR, '').lstrip('/')
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            return {'last_processed_index': -1, 'total_tickers': 0, 'run_date': run_date}
        except Exception as e:
            logger.error(f"Read S3 progress failed {s3_key}: {e}")
            return {'last_processed_index': -1, 'total_tickers': 0, 'run_date': run_date}

def save_json(path: str, obj, overwrite: bool = False):
    if LOCALRUN:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not overwrite:
            logger.debug(f"File exists skip: {path}")
            return
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved local: {path}")
    else:
        s3_key = path.replace(BASE_DIR, '').lstrip('/')
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            if overwrite:
                save_s3_json(obj, s3_key)
            else:
                logger.debug(f"S3 object exists skip: {s3_key}")
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                save_s3_json(obj, s3_key)
            else:
                logger.error(f"S3 head error {s3_key}: {e}")
                raise

def save_news_items(news_items: list[dict], ticker: str, run_date: str, source: str):
    out_dir = os.path.join(BASE_DIR, 'NEWS', source, run_date, ticker)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'news.json')
    save_json(out_path, news_items, overwrite=True)
    logger.info(f"Saved aggregated news for {ticker}")

def save_progress(run_date: str, last_processed_index: int, total_tickers: int):
    progress_path = os.path.join(BASE_DIR, 'DBPROGRESS', SOURCE, run_date, 'progress.json')
    progress_data = {
        'last_processed_index': last_processed_index,
        'total_tickers': total_tickers,
        'run_date': run_date
    }
    save_json(progress_path, progress_data, overwrite=True)

# --------------------------------------------------
# News fetching & de-dup
# --------------------------------------------------

DISCLAIMER_PATTERNS = [
    "may earn commission or revenue on some items"
]

def clean_full_text(txt: str) -> str:
    if not txt:
        return txt
    lines = [l for l in txt.splitlines() if l.strip()]
    filtered = [
        l for l in lines
        if not any(p.lower() in l.lower() for p in DISCLAIMER_PATTERNS)
    ]
    return "\n\n".join(filtered).strip()

def fetch_full_text(url: str) -> str:
    if not FETCH_FULL_TEXT:
        return ''
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        container = soup.find('article') or soup
        paras = container.find_all('p')
        raw = "\n\n".join(p.get_text(strip=True) for p in paras)
        return clean_full_text(raw)
    except Exception as e:
        logger.debug(f"Full text fetch failed {url}: {e}")
        return ''

def _dedup_key(item: dict) -> str:
    link = (item.get('link') or '').strip()
    title = (item.get('title') or '').strip()
    if DEDUP_KEY == 'title':
        return title.lower()
    if DEDUP_KEY == 'link_title':
        return f"{link.split('?')[0].lower()}::{title.lower()}"
    # default link
    return link.split('?')[0].lower()

def fetch_news_for_ticker(ticker: str, visited: set[str]) -> list[dict]:
    """
    Fetch news for a ticker with:
      - optional filtering by relatedTickers (FILTER_RELATED)
      - global & per-ticker de-dup using visited (by link/title/key)
    """
    logger.info(f"Fetching news for {ticker}")
    url = 'https://query1.finance.yahoo.com/v1/finance/search'
    params = {'q': ticker, 'newsCount': min(MAX_NEWS_PER_TKR, 50), 'quotesCount': 0}
    try:
        resp = session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get('news', [])
    except Exception as e:
        logger.error(f"Search API error {ticker}: {e}")
        return []

    per_ticker_seen = set()
    news_list = []
    for item in items[:MAX_NEWS_PER_TKR]:
        link = item.get('link')
        title = item.get('title')
        if not link or not title:
            continue

        # relatedTickers filter
        related = item.get('relatedTickers')
        if FILTER_RELATED and isinstance(related, list) and related and ticker not in related:
            continue

        key_val = _dedup_key(item)
        if key_val in visited or key_val in per_ticker_seen:
            continue

        per_ticker_seen.add(key_val)
        visited.add(key_val)

        ts = item.get('providerPublishTime')
        if ts:
            dt_str = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        else:
            dt_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')

        full = fetch_full_text(link)

        news_list.append({
            'date': dt_str,
            'source': SOURCE,
            'ticker': ticker,
            'title': title,
            'link': link,
            'content': full,
        })

        if FETCH_FULL_TEXT:
            time.sleep(FULLTEXT_SLEEP)  # light throttle to be polite

    logger.info(f"Fetched {len(news_list)} unique news items for {ticker}")
    return news_list

# --------------------------------------------------
# Financials & statistics
# --------------------------------------------------
def fetch_financials_yf(ticker: str) -> dict:
    logger.info(f"Fetching quarterly financials for {ticker}")
    try:
        tkr = yf.Ticker(ticker)
        return {
            'income_statement_quarterly': df_to_serializable(tkr.quarterly_financials),
            'balance_sheet_quarterly': df_to_serializable(tkr.quarterly_balance_sheet),
            'cash_flow_quarterly': df_to_serializable(tkr.quarterly_cashflow),
        }
    except Exception as e:
        logger.error(f"Financials error {ticker}: {e}")
        return {}

def save_financials_yf(data: dict, ticker: str, run_date: str):
    path = os.path.join(BASE_DIR, 'FINANCIALS', SOURCE, run_date, ticker, 'financials.json')
    if not data:
        logger.debug(f"No financial data {ticker}")
        return
    if LOCALRUN and os.path.exists(path):
        logger.debug(f"Financials exists skip {path}")
    else:
        save_json(path, data)
    dates = list(data.get('income_statement_quarterly', {}))
    if dates:
        latest = max(datetime.strptime(d, '%m/%d/%Y').date() for d in dates)
        logger.info(f"Latest financial quarter {ticker}: {latest.strftime('%B %Y')}")

STAT_MAP = {
    'Market Cap': 'marketCap',
    'Enterprise Value': 'enterpriseValue',
    'Trailing P/E': 'trailingPE',
    'Forward P/E': 'forwardPE',
    'PEG Ratio': 'pegRatio',
    'Price/Sales (ttm)': 'priceToSalesTrailing12Months',
    'Price/Book': 'priceToBook',
    'Enterprise/Revenue': 'enterpriseToRevenue',
    'Enterprise/EBITDA': 'enterpriseToEbitda',
    'EBITDA': 'ebitda',
}

def fetch_statistics_yf(ticker: str, run_date: str) -> dict:
    logger.info(f"Fetching key statistics for {ticker}")
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.error(f"Stats error {ticker}: {e}")
        return {}
    rec = {'ticker': ticker, 'date': run_date}
    for label, key in STAT_MAP.items():
        rec[label] = info.get(key)
    return rec

def save_statistics_yf(data: dict, ticker: str, run_date: str):
    path = os.path.join(BASE_DIR, 'STATISTICS', SOURCE, run_date, ticker, 'statistics.json')
    if not data:
        logger.debug(f"No stats {ticker}")
        return
    if LOCALRUN and os.path.exists(path):
        logger.debug(f"Stats exists skip {path}")
    else:
        save_json(path, data)
    dt = datetime.strptime(run_date, '%Y-%m-%d')
    logger.info(f"Statistics snapshot {ticker}: {dt.strftime('%B %Y')}")

# --------------------------------------------------
# Orchestrator
# --------------------------------------------------
def run(event, context):
    tk_list = []
    run_date = datetime.now(timezone.utc).date().isoformat()

    progress = read_progress(run_date)
    start_index = max(event.get('start_index', progress['last_processed_index'] + 1), 0)

    if 'tickers' in event and event['tickers']:
        tk_list = event['tickers']
        logger.info(f"Tickers from event: {len(tk_list)}")
    else:
        if not DEFAULT_TICKERS:
            raise ValueError("DEFAULT_TICKERS not set")
        if not os.path.exists(DEFAULT_TICKERS) and LOCALRUN:
            raise FileNotFoundError(f"CSV not found: {DEFAULT_TICKERS}")
        try:
            tk_list = pd.read_csv(DEFAULT_TICKERS)['Symbol'].tolist()
        except Exception as e:
            logger.error(f"Read tickers failed: {e}")
            raise

    if event.get("testnews") == "True":
        tk_list = ['AAPL', 'GOOGL', 'MSFT']
        start_index = 0
        progress = {'last_processed_index': -1, 'total_tickers': len(tk_list), 'run_date': run_date}
        save_progress(run_date, -1, len(tk_list))
        logger.info(f"Test mode tickers: {tk_list}")

    if not tk_list:
        raise ValueError("Empty ticker list")

    tk_list = [t.strip().upper() for t in tk_list if t.strip()]

    if progress['total_tickers'] == 0:
        progress['total_tickers'] = len(tk_list)
        save_progress(run_date, progress['last_processed_index'], len(tk_list))

    today = datetime.now(timezone.utc).date()
    quarter_start = (today.month, today.day) in [(1,1),(4,1),(7,1),(10,1)]
    if event.get("quarter_start") == "True":
        quarter_start = True
        logger.info("Forced quarter_start=True")

    processed_tickers = []
    end_index = min(start_index + BATCH_SIZE, len(tk_list))

    # Global visited set (across tickers) for dedup
    visited_links: set[str] = set()

    for i in range(start_index, end_index):
        tk = tk_list[i]
        logger.info(f"Processing {tk} ({i+1}/{len(tk_list)})")

        # News with dedup
        news = fetch_news_for_ticker(tk, visited_links)
        if news:
            save_news_to_postgres(news, TABLE_NAME)
        else:
            logger.info(f"No unique news for {tk}")

        if quarter_start:
            fin = fetch_financials_yf(tk)
            for key in ['income_statement_quarterly', 'balance_sheet_quarterly', 'cash_flow_quarterly']:
                if fin.get(key):
                    save_quarterly_financials_to_postgres(tk, fin[key], FIN_TABLE_NAME)

            stats = fetch_statistics_yf(tk, run_date)
            if stats:
                save_statistics_to_postgres(stats, STAT_TABLE_NAME)
        else:
            logger.debug(f"Skip financials/statistics {tk} (not quarter start)")

        processed_tickers.append(tk)
        save_progress(run_date, i, len(tk_list))
        time.sleep(random.uniform(0.01, 0.15))

    next_index = end_index if end_index < len(tk_list) else None
    response = {
        'status': 'success' if processed_tickers else 'no_tickers_processed',
        'processed_tickers': processed_tickers,
        'date': run_date,
        'quarter_start': quarter_start,
        'next_index': next_index,
        'total_tickers': len(tk_list)
    }
    if next_index is not None:
        response['remaining_tickers'] = tk_list[end_index:]
    logger.info(f"Processed {len(processed_tickers)} tickers. Next index: {next_index}")
    return response

if __name__ == '__main__':
    test_event = {
        "quarter_start": "True",
        "testnews": "False"
    }
    run(test_event, None)