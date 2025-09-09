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
from dataUtil_Pgsql import save_news_to_postgres,save_statistics_to_postgres,save_quarterly_financials_to_postgres

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR        = os.getenv('DATA_DIR', './DATA')
SOURCE          = 'yfinance'
# Default tickers (path to CSV file with 'Symbol' column)
DEFAULT_TICKERS = os.getenv('DEFAULT_TICKERS', '')
TABLE_NAME    = os.getenv('TABLE_NAME', '')
FIN_TABLE_NAME = os.getenv('FIN_TABLE_NAME', '')
STAT_TABLE_NAME = os.getenv('STAT_TABLE_NAME', '')
S3_BUCKET       = os.getenv('S3_BUCKET', '')
LOCALRUN        = os.getenv('LOCALRUN', 'True').lower() in ('true', '1', 'yes')
BATCH_SIZE      = int(os.getenv('BATCH_SIZE', '25'))  # Number of tickers to process per Lambda invocation

# Init logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [yf-collect] %(message)s'
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

# Initialize S3 client if not running locally
if not LOCALRUN:
    if not S3_BUCKET:
        logging.error("S3_BUCKET environment variable is required when LOCALRUN is False")
        raise ValueError("S3_BUCKET not set")
    s3_client = boto3.client('s3')
    logging.info(f"Configured to save to S3 bucket: {S3_BUCKET}")
    logging.info(f"Configured to save to S3 bucket")
else:
    logging.info("Configured to save to local directory")

# Session for fetching article HTML
session = requests.Session()
session.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/114.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
})

# Helpers

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
    logging.info(f"Saved to S3: {s3_key}")

def read_progress(run_date: str) -> dict:
    progress_path = os.path.join(BASE_DIR, 'PROGRESS', SOURCE, run_date, 'progress.json')
    if LOCALRUN:
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error reading local progress file {progress_path}: {e}")
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
            logging.error(f"Error reading S3 progress file {s3_key}: {e}")
            return {'last_processed_index': -1, 'total_tickers': 0, 'run_date': run_date}

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
    progress_path = os.path.join(BASE_DIR, 'PROGRESS', SOURCE, run_date, 'progress.json')
    progress_data = {
        'last_processed_index': last_processed_index,
        'total_tickers': total_tickers,
        'run_date': run_date
    }
    save_json(progress_path, progress_data, overwrite=True)


# News (old-style)

def fetch_full_text(url: str) -> str:
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        container = soup.find('article') or soup
        paras = container.find_all('p')
        return "\n\n".join(p.get_text(strip=True) for p in paras)
    except Exception as e:
        logging.debug(f"Could not fetch full text from {url}: {e}")
        return ''


def fetch_news_for_ticker(ticker: str) -> list[dict]:
    logging.info(f"Fetching news for {ticker}")
    url = 'https://query1.finance.yahoo.com/v1/finance/search'
    params = {'q': ticker, 'newsCount': 50, 'quotesCount': 0}
    try:
        resp = session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get('news', [])
    except Exception as e:
        logging.error(f"Error fetching JSON for {ticker}: {e}")
        return []

    news_list = []
    for item in items:
        ts = item.get('providerPublishTime')
        dt_str = None
        if ts:
            dt_str = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        #uid = item.get('uuid')
        link = item.get('link')
        full = fetch_full_text(link) if link else ''
        news_list.append({
            #'id': uid,
            'date': dt_str,
            'ticker': ticker,
            'title': item.get('title'),
            'link': link,
            #'publisher': item.get('publisher'),
            'content': full,
            #'tags': item.get('relatedTickers', []),
        })
        time.sleep(random.uniform(0.01, 0.3))
    logging.info(f"Fetched {len(news_list)} news items for {ticker}")
    return news_list


# Quarterly Financials via yfinance

def fetch_financials_yf(ticker: str) -> dict:
    logging.info(f"Fetching quarterly financials for {ticker}")
    try:
        tkr = yf.Ticker(ticker)
        return {
            'income_statement_quarterly': df_to_serializable(tkr.quarterly_financials),
            'balance_sheet_quarterly': df_to_serializable(tkr.quarterly_balance_sheet),
            'cash_flow_quarterly': df_to_serializable(tkr.quarterly_cashflow),
        }
    except Exception as e:
        logging.error(f"Error fetching quarterlies for {ticker}: {e}")
        return {}


def save_financials_yf(data: dict, ticker: str, run_date: str):
    path = os.path.join(BASE_DIR, 'FINANCIALS', SOURCE, run_date, ticker, 'financials.json')
    if not data:
        logging.debug(f"No financial data to save for {ticker}")
        return
    if LOCALRUN:
        if os.path.exists(path):
            logging.debug(f"Financials file exists, skipping: {path}")
            #return
    else:
        s3_key = path.replace(BASE_DIR, '').lstrip('/')
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            logging.debug(f"S3 financials object exists, skipping: {s3_key}")
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                logging.error(f"Error checking S3 financials object {s3_key}: {e}")
                return
    save_json(path, data)
    dates = list(data.get('income_statement_quarterly', {}))
    if dates:
        latest = max(datetime.strptime(d, '%m/%d/%Y').date() for d in dates)
        logging.info(f"Latest financial quarter for {ticker}: {latest.strftime('%B %Y')}")


# Quarterly Key Statistics via yfinance

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
    logging.info(f"Fetching key statistics for {ticker}")
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logging.error(f"Error fetching stats for {ticker}: {e}")
        return {}
    rec = {'ticker': ticker, 'date': run_date}
    for label, key in STAT_MAP.items():
        rec[label] = info.get(key)
    return rec


def save_statistics_yf(data: dict, ticker: str, run_date: str):
    path = os.path.join(BASE_DIR, 'STATISTICS', SOURCE, run_date, ticker, 'statistics.json')
    if not data:
        logging.debug(f"No statistics data to save for {ticker}")
        return
    if LOCALRUN:
        if os.path.exists(path):
            logging.debug(f"Statistics file exists, skipping: {path}")
            #return
    else:
        s3_key = path.replace(BASE_DIR, '').lstrip('/')
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            logging.debug(f"S3 statistics object exists, skipping: {s3_key}")
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                logging.error(f"Error checking S3 statistics object {s3_key}: {e}")
                return
    save_json(path, data)
    dt = datetime.strptime(run_date, '%Y-%m-%d')
    logging.info(f"Latest statistics snapshot for {ticker}: {dt.strftime('%B %Y')}")


# Orchestrator

def run(event, context):
    # Load tickers from CSV file or event
    tk_list = []
    run_date = datetime.now(timezone.utc).date().isoformat()
    
    # Read progress from S3 or local
    progress = read_progress(run_date)
    start_index = max(event.get('start_index', progress['last_processed_index'] + 1), 0)
    
    if 'tickers' in event and event['tickers']:
        tk_list = event['tickers']
        logging.info(f"Using tickers from event: {len(tk_list)} tickers")
    else:
        if not DEFAULT_TICKERS:
            logging.error("DEFAULT_TICKERS environment variable must point to a valid CSV file")
            raise ValueError("DEFAULT_TICKERS not set")
        if not os.path.exists(DEFAULT_TICKERS) and LOCALRUN:
            logging.error(f"CSV file not found: {DEFAULT_TICKERS}")
            raise FileNotFoundError(f"CSV file not found: {DEFAULT_TICKERS}")
        
        try:
            tk_list = pd.read_csv(DEFAULT_TICKERS)['Symbol'].tolist()
        except Exception as e:
            logging.error(f"Error reading tickers from {DEFAULT_TICKERS}: {e}")
            raise

    if "testnews" in event:
        if event["testnews"] == "True":
            logging.info("Running in test mode, using hardcoded tickers")
            tk_list = ['AAPL', 'GOOGL', 'MSFT']  # For testing purposes
            start_index = 0
            progress = {'last_processed_index': -1, 'total_tickers': len(tk_list), 'run_date': run_date}
            save_progress(run_date, -1, len(tk_list))
            logging.info(f"Using testing tickers: {tk_list}")

    if not tk_list:
        logging.error("No tickers found")
        raise ValueError("Ticker list is empty")

    tk_list = [t.strip().upper() for t in tk_list if t.strip()]
    
    # Update progress with total tickers if not set
    if progress['total_tickers'] == 0:
        progress['total_tickers'] = len(tk_list)
        save_progress(run_date, progress['last_processed_index'], len(tk_list))

    # Check for quarter start
    today = datetime.now(timezone.utc).date()
    quarter_start = (today.month, today.day) in [(1, 1), (4, 1), (7, 1), (10, 1)]

    if "quarter_start" in event:
        if event["quarter_start"] == "True":
            quarter_start = True  # Force quarter start for testing
            logging.info("Forcing quarter start for testing")

    # Process tickers in batch
    processed_tickers = []
    end_index = min(start_index + BATCH_SIZE, len(tk_list))
    for i in range(start_index, end_index):
        tk = tk_list[i]
        logging.info(f"Processing ticker {tk} ({i + 1}/{len(tk_list)})")
        # Always fetch news
        news = fetch_news_for_ticker(tk)
        save_news_to_postgres(news, TABLE_NAME)
        #save_news_items(news, tk, run_date, SOURCE)

        if quarter_start:
            fin = fetch_financials_yf(tk)
            for key in ['income_statement_quarterly', 'balance_sheet_quarterly', 'cash_flow_quarterly']:
                if key in fin and fin[key]:
                    save_quarterly_financials_to_postgres(tk, fin[key], FIN_TABLE_NAME)
            #save_financials_yf(fin, tk, run_date)

            stats = fetch_statistics_yf(tk, run_date)
            if stats:
                save_statistics_to_postgres(stats, STAT_TABLE_NAME)
            #save_statistics_yf(stats, tk, run_date)
        else:
            logging.info(f"Skipping financials & statistics for {tk}; next run on quarter start")

        processed_tickers.append(tk)
        # Update progress after each ticker
        save_progress(run_date, i, len(tk_list))
        time.sleep(random.uniform(0.01, 0.2))

    # Prepare response for continuation
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
    logging.info(f"Processed {len(processed_tickers)} tickers. Next index: {next_index}")
    return response


if __name__ == '__main__':
    myrun = {
        "quarter_start": "True",
        "testnews": "False",
        "key3": "value3"
        }
    run(myrun, None)