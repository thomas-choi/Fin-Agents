"""
AWS Lambda function for collecting financial news from multiple sources
and storing articles in Cloudflare R2 with deduplication.

Environment Variables:
- TICKER_LIST: Path to CSV file containing stock tickers
    Supports formats:
    - Local file: /path/to/tickers.csv
    - AWS S3: s3://bucket-name/path/to/tickers.csv
    - Cloudflare R2: r2://bucket-name/path/to/tickers.csv
- NEWS_BUCKET: Cloudflare R2 bucket name for news storage
- R2_ACCOUNT_ID: Cloudflare Account ID
- R2_ACCESS_KEY: Cloudflare R2 Access Key
- R2_SECRET_KEY: Cloudflare R2 Secret Key
- DB_HOST: DigitalOcean MySQL host
- DB_PORT: DigitalOcean MySQL port (default: 3306)
- DB_USER: Database user
- DB_PASSWORD: Database password
- DB_NAME: Database name (default: news_collector)
- DB_TABLE: State tracking table name (default: ticker_progress)
- LOG_LEVEL: Logging level (default: INFO)
- ALPHA_VANTAGE_API_KEY: Optional Alpha Vantage API key
- NEWSAPI_API_KEY: Optional NewsAPI key
- POLYGON_API_KEY: Optional Polygon.io API key
- FMP_API_KEY: Optional Financial Modeling Prep API key
"""

import json
import logging
import os
import hashlib
import csv
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import boto3
import requests
import yfinance as yf
from botocore.config import Config
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# AWS clients
s3_client = boto3.client('s3')

# Cloudflare R2 client (S3 compatible)
r2_client = None

class R2Storage:
    """Cloudflare R2 storage handler using S3 API"""
    
    def __init__(self):
        """Initialize R2 client with Cloudflare credentials"""
        account_id = os.getenv('R2_ACCOUNT_ID')
        access_key = os.getenv('R2_ACCESS_KEY')
        secret_key = os.getenv('R2_SECRET_KEY')
        bucket_name = os.getenv('NEWS_BUCKET')
        
        if not all([account_id, access_key, secret_key, bucket_name]):
            raise ValueError("Missing required Cloudflare R2 credentials in environment variables")
        
        self.bucket_name = bucket_name
        self.endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto',
            config=Config(signature_version='s3v4')
        )
        logger.info(f"R2 storage initialized for bucket: {bucket_name}")
    
    def article_exists(self, ticker: str, article_url: str) -> bool:
        """
        Check if article already exists by URL hash
        
        Args:
            ticker: Stock ticker symbol
            article_url: URL of the article
            
        Returns:
            True if article exists, False otherwise
        """
        try:
            url_hash = hashlib.sha256(article_url.encode()).hexdigest()[:16]
            path = self._get_article_path(ticker, article_url)
            
            try:
                self.client.head_object(Bucket=self.bucket_name, Key=path)
                logger.debug(f"Article already exists: {path}")
                return True
            
            except self.client.exceptions.NoSuchKey:
                return False
        except Exception as e:
            logger.error(f"Error checking article existence {path}: {e}")
            return False
    
    def save_article(self, article: Dict) -> Tuple[bool, Optional[str]]:
        """
        Save article to R2 with deduplication
        
        Args:
            article: Article dictionary with fields: ticker, publish_date, title, content, source, url, data_source
            
        Returns:
            Tuple of (success: bool, path: Optional[str])
        """
        try:
            ticker = article.get('ticker', 'UNKNOWN').upper()
            url = article.get('url', '')
            publish_date = article.get('publish_date')
            
            # Check for duplicates
            if self.article_exists(ticker, url):
                logger.info(f"Skipping duplicate article: {ticker} - {article.get('title', 'Unknown')}")
                return False, None
            
            # Generate article ID from URL hash
            article_id = self._generate_article_id(article)
            
            # Build path: news/ticker={TICKER}/year={YYYY}/month={MM}/day={DD}/{article_id}.json
            path = self._get_article_path(ticker, article_id, publish_date)
            
            # Prepare JSON content
            content = {
                'ticker': ticker,
                'publish_date': publish_date.isoformat() if isinstance(publish_date, datetime) else publish_date,
                'title': article.get('title', 'No title'),
                'content': article.get('content', 'No content available'),
                'source': article.get('source', 'Unknown'),
                'url': url,
                'data_source': article.get('data_source', 'unknown'),
                'stored_at': datetime.utcnow().isoformat()
            }
            
            # Upload to R2
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=path,
                Body=json.dumps(content, ensure_ascii=False, indent=2).encode('utf-8'),
                ContentType='application/json',
                Metadata={
                    'ticker': ticker,
                    'source': article.get('data_source', 'unknown')
                }
            )
            
            logger.info(f"Article saved to R2: {path}")
            return True, path
            
        except Exception as e:
            logger.error(f"Error saving article to R2: {e}", exc_info=True)
            return False, None
    
    def _generate_article_id(self, article: Dict) -> str:
        """Generate unique article ID from URL hash and title"""
        url = article.get('url', '')
        title = article.get('title', '')
        combined = f"{url}:{title}"
        return f"article-{hashlib.sha256(combined.encode()).hexdigest()[:12]}"
    
    def _get_article_path(self, ticker: str, article_key: str, publish_date: Optional[datetime] = None) -> str:
        """
        Generate R2 path for article
        
        Args:
            ticker: Stock ticker
            article_key: Article ID or URL
            publish_date: Publication date (uses current date if not provided)
            
        Returns:
            S3 path string
        """
        if publish_date is None:
            publish_date = datetime.utcnow()
        elif isinstance(publish_date, str):
            try:
                publish_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
            except:
                publish_date = datetime.utcnow()
        
        year = publish_date.strftime('%Y')
        month = publish_date.strftime('%m')
        day = publish_date.strftime('%d')
        
        return f"news/ticker={ticker}/year={year}/month={month}/day={day}/{article_key}.json"


# SQLAlchemy ORM Model
Base = declarative_base()

class TickerProgress(Base):
    """SQLAlchemy model for ticker progress tracking"""
    __tablename__ = 'ticker_progress'
    
    state_id = Column(String(50), primary_key=True)
    last_index = Column(Integer, default=0)
    last_ticker = Column(String(20))
    articles_processed = Column(Integer, default=0)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class TickerStateManager:
    """Manages ticker processing state in DigitalOcean MySQL database using SQLAlchemy"""
    
    def __init__(self):
        """Initialize SQLAlchemy engine and session for state management"""
        try:
            self.db_host = os.getenv('DB_HOST')
            self.db_port = int(os.getenv('DB_PORT', '3306'))
            self.db_user = os.getenv('DB_USER')
            self.db_password = os.getenv('DB_PASSWORD')
            self.db_name = os.getenv('DB_NAME', 'news_collector')
            self.table_name = os.getenv('DB_TABLE', 'ticker_progress')
            
            if not all([self.db_host, self.db_user, self.db_password]):
                raise ValueError("Missing required DigitalOcean MySQL credentials in environment variables")
            
            # Create database connection string
            db_url = f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            
            # Create engine with NullPool to minimize connection overhead in Lambda
            self.engine = create_engine(
                db_url,
                pool_pre_ping=True,
                poolclass=NullPool,
                connect_args={'connect_timeout': 10}
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(bind=self.engine)
            self.session = None
            
            # Create tables
            self._ensure_table_exists()
            logger.info(f"SQLAlchemy state manager initialized for database: {self.db_name}")
            
        except Exception as e:
            logger.error(f"Error initializing SQLAlchemy state manager: {e}")
            raise
    
    def _get_session(self):
        """Get or create SQLAlchemy session"""
        try:
            if self.session is None:
                self.session = self.SessionLocal()
            return self.session
        except Exception as e:
            logger.error(f"Error creating SQLAlchemy session: {e}")
            raise
    
    def _ensure_table_exists(self):
        """Create state table if it doesn't exist"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info(f"Ensured table {self.table_name} exists")
        except Exception as e:
            logger.error(f"Error creating state table: {e}")
            raise
    
    def get_last_processed_ticker(self) -> Optional[int]:
        """
        Get the index of the last successfully processed ticker
        
        Returns:
            Ticker index or None if no state exists
        """
        try:
            session = self._get_session()
            result = session.query(TickerProgress).filter(
                TickerProgress.state_id == 'ticker_progress'
            ).first()
            
            if result:
                return result.last_index
            return None
            
        except Exception as e:
            logger.error(f"Error reading state from database: {e}")
            return None
    
    def save_progress(self, ticker_index: int, ticker: str, articles_count: int) -> bool:
        """
        Save progress for current ticker
        
        Args:
            ticker_index: Index in ticker list
            ticker: Ticker symbol
            articles_count: Number of articles processed
            
        Returns:
            True if successful
        """
        try:
            session = self._get_session()
            
            # Try to get existing record
            existing = session.query(TickerProgress).filter(
                TickerProgress.state_id == 'ticker_progress'
            ).first()
            
            if existing:
                # Update existing record
                existing.last_index = ticker_index
                existing.last_ticker = ticker
                existing.articles_processed += articles_count
                existing.last_update = datetime.utcnow()
            else:
                # Create new record
                new_record = TickerProgress(
                    state_id='ticker_progress',
                    last_index=ticker_index,
                    last_ticker=ticker,
                    articles_processed=articles_count
                )
                session.add(new_record)
            
            session.commit()
            logger.info(f"Progress saved: {ticker} (index {ticker_index}), {articles_count} articles")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state to database: {e}")
            try:
                session.rollback()
            except:
                pass
            return False
    
    def close(self):
        """Close database connection"""
        try:
            if self.session:
                self.session.close()
                logger.info("Database session closed")
            if self.engine:
                self.engine.dispose()
                logger.info("Engine disposed")
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")


class NewsCollector:
    """Enhanced news collector for Lambda environment"""
    
    def __init__(self):
        """Initialize news collector with all available sources"""
        self.timeout = 8  # 8 seconds per ticker to stay within Lambda limits
    
    def get_news(self, ticker: str) -> List[Dict]:
        """
        Fetch news from all available sources for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of article dictionaries
        """
        all_articles = []
        
        # Yahoo Finance (always available)
        articles = self._yf_get_news(ticker)
        all_articles.extend(articles)
        
        # Alpha Vantage
        articles = self._av_get_news(ticker)
        all_articles.extend(articles)
        
        # NewsAPI
        articles = self._newsapi_get_news(ticker)
        all_articles.extend(articles)
        
        # Polygon.io
        articles = self._polygon_get_news(ticker)
        all_articles.extend(articles)
        
        # FMP
        articles = self._fmp_get_news(ticker)
        all_articles.extend(articles)
        
        logger.info(f"Collected {len(all_articles)} articles for {ticker}")
        return all_articles
    
    def _yf_get_news(self, ticker: str) -> List[Dict]:
        """Fetch from Yahoo Finance"""
        try:
            logger.info(f"Fetching Yahoo Finance news for: {ticker}")
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            if not news:
                return []
            
            articles = []
            for article in news[:10]:  # Limit to 10 per source
                content = article.get('content', {})
                title = content.get('title', 'No title available')
                
                link = 'No link available'
                if content.get('canonicalUrl') and content['canonicalUrl'].get('url'):
                    link = content['canonicalUrl']['url']
                elif content.get('clickThroughUrl') and content['clickThroughUrl'].get('url'):
                    link = content['clickThroughUrl']['url']
                
                pub_date_str = content.get('pubDate', '')
                publish_date = None
                if pub_date_str:
                    try:
                        publish_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    except ValueError:
                        publish_date = datetime.utcnow()
                
                content_text = content.get('summary', 'No content available')
                publisher = 'Unknown'
                if content.get('provider') and content['provider'].get('displayName'):
                    publisher = content['provider']['displayName']
                
                articles.append({
                    'ticker': ticker,
                    'publish_date': publish_date or datetime.utcnow(),
                    'title': title,
                    'url': link,
                    'content': content_text,
                    'source': publisher,
                    'data_source': 'yfinance'
                })
            
            logger.info(f"Yahoo Finance: {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {e}")
            return []
    
    def _av_get_news(self, ticker: str) -> List[Dict]:
        """Fetch from Alpha Vantage"""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            return []
        
        try:
            logger.info(f"Fetching Alpha Vantage news for: {ticker}")
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': api_key,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            data = response.json()
            
            if 'feed' not in data:
                return []
            
            articles = []
            for article in data['feed'][:10]:
                date_str = article.get('time_published', '')
                publish_date = None
                if date_str:
                    try:
                        publish_date = datetime.strptime(date_str, '%Y%m%dT%H%M%S')
                    except ValueError:
                        publish_date = datetime.utcnow()
                
                articles.append({
                    'ticker': ticker,
                    'publish_date': publish_date or datetime.utcnow(),
                    'title': article.get('title', 'No title'),
                    'url': article.get('url', ''),
                    'content': article.get('summary', 'No content available'),
                    'source': article.get('source', 'Unknown'),
                    'data_source': 'alpha_vantage'
                })
            
            logger.info(f"Alpha Vantage: {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    def _newsapi_get_news(self, ticker: str) -> List[Dict]:
        """Fetch from NewsAPI"""
        api_key = os.getenv('NEWSAPI_API_KEY')
        if not api_key:
            return []
        
        try:
            logger.info(f"Fetching NewsAPI news for: {ticker}")
            company_name_key = f"{ticker}_COMPANY_NAME"
            company_name = os.getenv(company_name_key, ticker)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{company_name} OR {ticker} stock",
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 30,
                'apiKey': api_key,
                'domains': 'bloomberg.com,reuters.com,marketwatch.com,wsj.com,cnbc.com,finance.yahoo.com,barrons.com'
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            data = response.json()
            
            if data.get('status') != 'ok' or not data.get('articles'):
                return []
            
            articles = []
            for article in data['articles'][:10]:
                date_str = article.get('publishedAt', '')
                publish_date = None
                if date_str:
                    try:
                        publish_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        publish_date = datetime.utcnow()
                
                articles.append({
                    'ticker': ticker,
                    'publish_date': publish_date or datetime.utcnow(),
                    'title': article.get('title', 'No title'),
                    'url': article.get('url', ''),
                    'content': article.get('description', 'No content available'),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'data_source': 'newsapi'
                })
            
            logger.info(f"NewsAPI: {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            return []
    
    def _polygon_get_news(self, ticker: str) -> List[Dict]:
        """Fetch from Polygon.io"""
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            return []
        
        try:
            logger.info(f"Fetching Polygon.io news for: {ticker}")
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                'ticker': ticker,
                'limit': 50,
                'order': 'desc',
                'sort': 'published_utc',
                'apiKey': api_key
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            data = response.json()
            
            if 'results' not in data or not data['results']:
                return []
            
            articles = []
            for article in data['results'][:10]:
                date_str = article.get('published_utc', '')
                publish_date = None
                if date_str:
                    try:
                        publish_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        publish_date = datetime.utcnow()
                
                articles.append({
                    'ticker': ticker,
                    'publish_date': publish_date or datetime.utcnow(),
                    'title': article.get('title', 'No title'),
                    'url': article.get('article_url', ''),
                    'content': article.get('description', 'No content available'),
                    'source': article.get('author', 'Unknown'),
                    'data_source': 'polygon'
                })
            
            logger.info(f"Polygon.io: {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"Error fetching Polygon.io news: {e}")
            return []
    
    def _fmp_get_news(self, ticker: str) -> List[Dict]:
        """Fetch from Financial Modeling Prep"""
        api_key = os.getenv('FMP_API_KEY')
        if not api_key:
            return []
        
        try:
            logger.info(f"Fetching FMP news for: {ticker}")
            url = "https://financialmodelingprep.com/api/v3/stock_news"
            params = {
                'tickers': ticker,
                'limit': 50,
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            data = response.json()
            
            if not isinstance(data, list) or not data:
                return []
            
            articles = []
            for article in data[:10]:
                date_str = article.get('publishedDate', '')
                publish_date = None
                if date_str:
                    try:
                        publish_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        publish_date = datetime.utcnow()
                
                articles.append({
                    'ticker': ticker,
                    'publish_date': publish_date or datetime.utcnow(),
                    'title': article.get('title', 'No title'),
                    'url': article.get('url', ''),
                    'content': article.get('text', 'No content available'),
                    'source': article.get('site', 'Unknown'),
                    'data_source': 'fmp'
                })
            
            logger.info(f"FMP: {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"Error fetching FMP news: {e}")
            return []


def load_ticker_list() -> List[str]:
    """
    Load ticker list from CSV file specified in environment variable
    
    Supports three path formats:
    - Local file: /path/to/tickers.csv
    - AWS S3: s3://bucket-name/path/to/tickers.csv
    - Cloudflare R2: r2://bucket-name/path/to/tickers.csv
    
    Returns:
        List of ticker symbols in uppercase
    """
    ticker_list_path = os.getenv('TICKER_LIST')
    if not ticker_list_path:
        raise ValueError("TICKER_LIST environment variable not set")
    
    try:
        tickers = []
        
        # Check if it's an S3 path
        if ticker_list_path.startswith('s3://'):
            bucket, key = ticker_list_path.replace('s3://', '').split('/', 1)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            reader = csv.reader(io.StringIO(content))
            logger.info(f"Loading ticker list from S3: {ticker_list_path}")
        
        # Check if it's a Cloudflare R2 path
        elif ticker_list_path.startswith('r2://'):
            r2_storage = R2Storage()
            bucket_name = ticker_list_path.replace('r2://', '').split('/', 1)[0]
            key = ticker_list_path.replace(f'r2://{bucket_name}/', '')
            
            response = r2_storage.client.get_object(Bucket=bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')
            reader = csv.reader(io.StringIO(content))
            logger.info(f"Loading ticker list from Cloudflare R2: {ticker_list_path}")
        
        else:
            # Local file
            with open(ticker_list_path, 'r') as f:
                reader = csv.reader(f)
            logger.info(f"Loading ticker list from local file: {ticker_list_path}")
        
        next(reader, None)  # Skip header
        for row in reader:
            if row and row[0].strip():
                tickers.append(row[0].strip().upper())
        
        logger.info(f"Loaded {len(tickers)} tickers from {ticker_list_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading ticker list: {e}")
        raise


def lambda_handler(event, context):
    """
    AWS Lambda handler function
    
    Args:
        event: Lambda event (typically empty for scheduled invocation)
        context: Lambda context with runtime information
    
    Returns:
        Response dict with status, processed tickers, and article count
    """
    start_time = datetime.utcnow()
    processed_tickers = 0
    total_articles = 0
    state_manager = None
    
    try:
        # Initialize components
        logger.info("Initializing news collector Lambda function")
        
        # Load ticker list
        tickers = load_ticker_list()
        if not tickers:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No tickers loaded'})
            }
        
        # Initialize state manager
        state_manager = TickerStateManager()
        last_index = state_manager.get_last_processed_ticker()
        start_index = (last_index + 1) % len(tickers) if last_index is not None else 0
        
        logger.info(f"Total tickers: {len(tickers)}, starting from index: {start_index}")
        
        # Initialize R2 storage
        r2_storage = R2Storage()
        
        # Initialize news collector
        news_collector = NewsCollector()
        
        # Process tickers (with 10 minute Lambda limit in mind)
        remaining_time = context.get_remaining_time_in_millis() / 1000 if context else 600
        
        current_index = start_index
        while remaining_time > 30:  # Keep 30 seconds buffer for cleanup
            ticker = tickers[current_index]
            
            try:
                logger.info(f"Processing ticker {current_index + 1}/{len(tickers)}: {ticker}")
                
                # Fetch news
                articles = news_collector.get_news(ticker)
                
                # Store articles
                saved_count = 0
                for article in articles:
                    success, path = r2_storage.save_article(article)
                    if success:
                        saved_count += 1
                
                logger.info(f"Ticker {ticker}: {len(articles)} articles fetched, {saved_count} saved")
                total_articles += saved_count
                
                # Save progress
                state_manager.save_progress(current_index, ticker, saved_count)
                processed_tickers += 1
                
                # Move to next ticker
                current_index = (current_index + 1) % len(tickers)
                
                # Check remaining time
                remaining_time = context.get_remaining_time_in_millis() / 1000 if context else 600
                
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}", exc_info=True)
                # Continue to next ticker on error
                current_index = (current_index + 1) % len(tickers)
                continue
        
        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'News collection completed successfully',
                'processed_tickers': processed_tickers,
                'total_articles': total_articles,
                'elapsed_seconds': elapsed_time,
                'next_start_index': current_index
            })
        }
        
        logger.info(f"Lambda execution completed: {processed_tickers} tickers, {total_articles} articles in {elapsed_time}s")
        return response
        
    except Exception as e:
        logger.error(f"Fatal error in Lambda handler: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'processed_tickers': processed_tickers,
                'total_articles': total_articles
            })
        }
    
    finally:
        # Close database connection
        if state_manager:
            try:
                state_manager.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")


# For local testing
if __name__ == "__main__":
    class MockContext:
        def get_remaining_time_in_millis(self):
            return 600000  # 10 minutes
    
    result = lambda_handler({}, MockContext())
    print(json.dumps(result, indent=2))
