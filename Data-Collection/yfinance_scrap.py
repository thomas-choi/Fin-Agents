import yfinance as yf
import boto3
import json
import logging
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any

# Configure logging to write to /tmp in Lambda
TODAY = datetime.now().strftime("%Y-%m-%d")  # e.g., "2025-05-29"
LOG_FILE = f"/tmp/scrape_financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration - Use environment variables for Lambda
S3_BUCKET = os.environ.get("S3_BUCKET", "financial-data-for-llm")

# Data type configurations based on the table
DATA_CONFIG = {
    "news": {"update_period": "daily", "sub_types": ["data"]},
    "community": {"update_period": "daily", "sub_types": ["data"]},
    "statistics": {
        "update_period": "quarterly",
        "sub_types": ["valuation-measure", "financial-highlights", "trading-information"]
    },
    "financials": {
        "update_period": "quarterly",
        "sub_types": ["income-statement", "balance-sheet", "cash-flow"]
    },
    "analysis": {
        "update_period": "daily",
        "sub_types": [
            "earnings-estimate", "revenue-estimate", "earnings-history",
            "eps-trend", "eps-revisions", "growth-estimates",
            "top-analysts", "upgrades-downgrades"
        ]
    },
    "holders": {"update_period": "daily", "sub_types": ["data"]},
    "sustainability": {"update_period": "daily", "sub_types": ["data"]}
}

def should_update(update_period: str, last_updated: str) -> bool:
    """Check if the data needs to be updated based on its update period."""
    if not last_updated:
        logger.debug("No last updated date found, triggering update")
        return True
    last_date = datetime.strptime(last_updated, "%Y-%m-%d")
    today = datetime.strptime(TODAY, "%Y-%m-%d")
    if update_period == "daily":
        return (today - last_date).days >= 1
    elif update_period == "quarterly":
        return (today - last_date).days >= 90
    elif update_period == "annual":
        return (today - last_date).days >= 365
    logger.warning(f"Unknown update period: {update_period}")
    return True

def fetch_data(ticker: str, data_type: str, sub_type: str) -> Dict[str, Any]:
    """Fetch data for a given ticker, data type, and subtype using yfinance."""
    logger.debug(f"Fetching {data_type}/{sub_type} for ticker {ticker}")
    stock = yf.Ticker(ticker)
    data = {}
    
    try:
        if data_type == "news":
            data = stock.news
        elif data_type == "community":
            # yfinance doesn't directly support community data; placeholder for custom scraping
            data = {"message": "Community data scraping not implemented"}
        elif data_type == "statistics":
            stats = stock.stats()
            if sub_type == "valuation-measure":
                data = stats.get("valuation_measures", {})
            elif sub_type == "financial-highlights":
                data = stats.get("financial_highlights", {})
            elif sub_type == "trading-information":
                data = stats.get("trading_information", {})
        elif data_type == "financials":
            if sub_type == "income-statement":
                data = stock.financials.to_dict()
            elif sub_type == "balance-sheet":
                data = stock.balance_sheet.to_dict()
            elif sub_type == "cash-flow":
                data = stock.cashflow.to_dict()
        elif data_type == "analysis":
            # yfinance analysis data; placeholder for custom mapping
            analysis = stock.analysis
            data = {"analysis_type": sub_type, "data": "To be implemented"}
        elif data_type == "holders":
            data = stock.major_holders.to_dict()
        elif data_type == "sustainability":
            data = stock.sustainability.to_dict()
    except Exception as e:
        logger.error(f"Failed to fetch {data_type}/{sub_type} for {ticker}: {str(e)}")
        raise
    
    logger.debug(f"Successfully fetched {data_type}/{sub_type} for {ticker}")
    return data

def save_to_s3(ticker: str, date: str, data_type: str, sub_type: str, data: Dict[str, Any]) -> None:
    """Save the scraped data to S3 in the specified structure."""
    s3_client = boto3.client("s3")  # No region or credentials needed; Lambda role handles it
    s3_key = f"{ticker}/{date}/{data_type}/{sub_type}.json"
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(data, indent=2)
        )
        logger.info(f"Saved data to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to save {data_type}/{sub_type} for {ticker} to S3: {str(e)}")
        raise

def check_last_updated(ticker: str, date: str, data_type: str, sub_type: str) -> str:
    """Check the last updated date for the given data in S3."""
    s3_client = boto3.client("s3")
    s3_key = f"{ticker}/{date}/{data_type}/{sub_type}.json"
    try:
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        last_updated = response["LastModified"].strftime("%Y-%m-%d")
        logger.debug(f"Found last updated date for {s3_key}: {last_updated}")
        return last_updated
    except s3_client.exceptions.ClientError:
        logger.debug(f"No existing data found for {s3_key}")
        return ""

def scrape_and_store(tickers: List[str]) -> Dict[str, Any]:
    """Main function to scrape data for a list of tickers and store in S3."""
    logger.info(f"Starting scraping process for tickers: {tickers}")
    results = {"success": [], "failed": []}
    
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        for data_type, config in DATA_CONFIG.items():
            update_period = config["update_period"]
            sub_types = config["sub_types"]
            
            for sub_type in sub_types:
                # Check if we need to update the data
                last_updated = check_last_updated(ticker, TODAY, data_type, sub_type)
                if not should_update(update_period, last_updated):
                    logger.info(f"Skipping {data_type}/{sub_type} for {ticker} - already up to date")
                    continue
                
                # Fetch and save the data
                try:
                    data = fetch_data(ticker, data_type, sub_type)
                    save_to_s3(ticker, TODAY, data_type, sub_type, data)
                    results["success"].append(f"{ticker}/{data_type}/{sub_type}")
                except Exception as e:
                    logger.error(f"Error processing {data_type}/{sub_type} for {ticker}: {str(e)}")
                    results["failed"].append(f"{ticker}/{data_type}/{sub_type}")
                    continue
    
    logger.info("Scraping process completed")
    return results

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler function."""
    try:
        # Extract tickers from the event
        tickers = event.get("tickers", [])
        if not tickers:
            logger.error("No tickers provided in the event")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No tickers provided"})
            }
        
        # Run the scraping process
        results = scrape_and_store(tickers)
        
        # Return the result
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Scraping completed",
                "results": results,
                "log_file": LOG_FILE
            })
        }
    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }