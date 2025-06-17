import json
# import boto3   
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Initialize S3 client
# s3_client = boto3.client('s3')   

# Configuration
TICKER = "TSLA"
BUCKET_NAME = "yahoo-finance-data-collection"
BASE_URL = "https://finance.yahoo.com/quote/TSLA"

def fetch_news():
    """Scrape news articles from Yahoo Finance."""
    try:
        url = f"{BASE_URL}/news/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        for item in soup.select('li.js-stream-content')[:5]:  # Limit to 5 recent articles
            title = item.select_one('h3').text if item.select_one('h3') else ""
            link = item.select_one('a')['href'] if item.select_one('a') else ""
            summary = item.select_one('p').text if item.select_one('p') else ""
            news_items.append({"title": title, "link": link, "summary": summary})
        return news_items
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return []

def fetch_statistics():
    """Fetch key statistics using yfinance."""
    try:
        stock = yf.Ticker(TICKER)
        stats = {
            "valuation": {
                "market_cap": stock.info.get("marketCap"),
                "pe_ratio": stock.info.get("trailingPE"),
                "pb_ratio": stock.info.get("priceToBook")
            },
            "financial_highlights": {
                "revenue": stock.info.get("totalRevenue"),
                "ebitda": stock.info.get("ebitda"),
                "net_income": stock.info.get("netIncomeToCommon")
            },
            "trading_info": {
                "beta": stock.info.get("beta"),
                "52_week_high": stock.info.get("fiftyTwoWeekHigh"),
                "52_week_low": stock.info.get("fiftyTwoWeekLow")
            }
        }
        return stats
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        return {}

def fetch_financials():
    """Fetch financial statements using yfinance."""
    try:
        stock = yf.Ticker(TICKER)
        financials = {
            "income_statement": stock.financials.to_dict(),
            "balance_sheet": stock.balance_sheet.to_dict(),
            "cash_flow": stock.cash_flow.to_dict()
        }
        return financials
    except Exception as e:
        logger.error(f"Error fetching financials: {str(e)}")
        return {}

def save_to_s3(data, folder, filename):
    """Save data to S3 bucket."""
    try:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"{TICKER}/{folder}/{date_str}/{filename}"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Saved {key} to S3")
    except Exception as e:
        logger.error(f"Error saving to S3: {str(e)}")

def lambda_handler(event, context):
    """AWS Lambda handler."""
    try:
        # Collect data
        data = {
            "ticker": TICKER,
            "timestamp": datetime.utcnow().isoformat(),
            "news": fetch_news(),
            "statistics": fetch_statistics(),
            "financials": fetch_financials()
            # Add other sections (community, analysis, holders, sustainability) as needed
        }
        
        # Save to S3
        # save_to_s3(data, "daily", f"{TICKER}_daily.json")  
        print(data)
        
        return {
            "statusCode": 200,
            "body": json.dumps({"message": f"Data collected for {TICKER}"})
        }
    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    

if __name__ == "__main__":
    lambda_handler(None, None)
