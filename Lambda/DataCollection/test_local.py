#!/usr/bin/env python3
"""
Local Testing Script for News Collector Lambda Function
Allows testing without deploying to AWS
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockContext:
    """Mock Lambda context for local testing"""
    
    def __init__(self, timeout_ms=600000):
        self.start_time = datetime.utcnow()
        self.timeout_ms = timeout_ms
        self.function_name = "news-collector-local"
        self.request_id = "local-test-request"
        self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:news-collector-local"
        self.memory_limit_in_mb = 1024
    
    def get_remaining_time_in_millis(self):
        """Simulate remaining time for Lambda"""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        remaining = max(0, self.timeout_ms - elapsed)
        return int(remaining)


def test_imports():
    """Test that all required packages are available"""
    logger.info("Testing imports...")
    
    required_packages = [
        'boto3',
        'requests',
        'yfinance',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('_', '-').replace('-', '_'))
            logger.info(f"  ✓ {package}")
        except ImportError:
            logger.warning(f"  ✗ {package} not found")
            missing.append(package)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def test_environment_variables():
    """Test that all required environment variables are set"""
    logger.info("\nTesting environment variables...")
    logger.info("TICKER_LIST can be: local file path, s3://bucket/key, or r2://bucket/key")
    
    required_vars = [
        'TICKER_LIST',
        'NEWS_BUCKET',
        'R2_ACCOUNT_ID',
        'R2_ACCESS_KEY',
        'R2_SECRET_KEY',
        'DB_HOST',
        'DB_USER',
        'DB_PASSWORD'
    ]
    
    optional_vars = [
        'DB_PORT',
        'DB_NAME',
        'DB_TABLE',
        'LOG_LEVEL',
        'ALPHA_VANTAGE_API_KEY',
        'NEWSAPI_API_KEY',
        'POLYGON_API_KEY',
        'FMP_API_KEY'
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"  ✓ {var} = {value[:20]}...")
        else:
            logger.warning(f"  ✗ {var} not set (REQUIRED)")
            missing.append(var)
    
    logger.info("\n  Optional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"    ✓ {var} = {value[:20]}...")
        else:
            logger.info(f"    ○ {var} not set")
    
    if missing:
        logger.error(f"\nMissing required variables: {', '.join(missing)}")
        return False
    
    return True


def test_ticker_list():
    """Test that ticker list can be loaded"""
    logger.info("\nTesting ticker list loading...")
    
    try:
        from news_collector_lambda import load_ticker_list
        
        tickers = load_ticker_list()
        logger.info(f"  ✓ Loaded {len(tickers)} tickers")
        logger.info(f"    First 10: {tickers[:10]}")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Error loading ticker list: {e}")
        return False


def test_news_collector():
    """Test news collector with a single ticker"""
    logger.info("\nTesting news collector...")
    
    try:
        from news_collector_lambda import NewsCollector
        
        collector = NewsCollector()
        ticker = 'AAPL'
        
        logger.info(f"  Fetching news for {ticker}...")
        articles = collector.get_news(ticker)
        logger.info(f"  ✓ Retrieved {len(articles)} articles")
        
        if articles:
            article = articles[0]
            logger.info(f"    Sample article:")
            logger.info(f"      - Title: {article.get('title', 'N/A')[:60]}...")
            logger.info(f"      - Source: {article.get('data_source', 'N/A')}")
            logger.info(f"      - Date: {article.get('publish_date', 'N/A')}")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Error testing news collector: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_r2_connection():
    """Test R2 storage connection"""
    logger.info("\nTesting R2 connection...")
    
    try:
        from news_collector_lambda import R2Storage
        
        storage = R2Storage()
        logger.info(f"  ✓ Connected to R2 bucket: {storage.bucket_name}")
        
        # Try to list a few objects
        response = storage.client.list_objects_v2(
            Bucket=storage.bucket_name,
            MaxKeys=5
        )
        
        count = response.get('KeyCount', 0)
        logger.info(f"  ✓ Found {count} objects in bucket")
        
        if count > 0:
            for obj in response.get('Contents', [])[:3]:
                logger.info(f"    - {obj['Key']}")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Error connecting to R2: {e}")
        return False


def test_mysql():
    """Test MySQL state manager"""
    logger.info("\nTesting MySQL connection...")
    
    try:
        from news_collector_lambda import TickerStateManager
        
        manager = TickerStateManager()
        
        # Try to read current state
        last_index = manager.get_last_processed_ticker()
        logger.info(f"  ✓ Connected to MySQL database: {manager.db_name}")
        logger.info(f"  ✓ State table: {manager.table_name}")
        logger.info(f"  ✓ Last processed ticker index: {last_index}")
        
        # Close connection
        manager.close()
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Error connecting to MySQL: {e}")
        return False


def test_full_invocation():
    """Test full Lambda invocation"""
    logger.info("\nTesting full Lambda invocation...")
    
    try:
        from news_collector_lambda import lambda_handler
        
        context = MockContext()
        result = lambda_handler({}, context)
        
        logger.info(f"  ✓ Lambda execution completed")
        logger.info(f"    Response: {json.dumps(result, indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Error during Lambda invocation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests in sequence"""
    logger.info("=" * 70)
    logger.info("News Collector Lambda - Local Testing Suite")
    logger.info("=" * 70)
    
    tests = [
        ("Import Check", test_imports),
        ("Environment Variables", test_environment_variables),
        ("Ticker List Loading", test_ticker_list),
        ("News Collector", test_news_collector),
        ("R2 Connection", test_r2_connection),
        ("MySQL Connection", test_mysql),
        ("Full Lambda Invocation", test_full_invocation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
        
        logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 70)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 70)
    
    return passed == total


def load_env_file(env_file='.env'):
    """Load environment variables from .env file if it exists"""
    if os.path.exists(env_file):
        logger.info(f"Loading environment from {env_file}")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"✓ Loaded .env file")
        except ImportError:
            logger.warning("python-dotenv not installed, reading .env manually")
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Local testing script for News Collector Lambda"
    )
    parser.add_argument(
        '--test',
        choices=['imports', 'env', 'ticker-list', 'collector', 'r2', 'mysql', 'full', 'all'],
        default='all',
        help='Specific test to run (default: all)'
    )
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file (default: .env)'
    )
    parser.add_argument(
        '--ticker',
        default='AAPL',
        help='Ticker symbol for testing (default: AAPL)'
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_env_file(args.env_file)
    
    # Run tests
    test_map = {
        'imports': test_imports,
        'env': test_environment_variables,
        'ticker-list': test_ticker_list,
        'collector': test_news_collector,
        'r2': test_r2_connection,
        'mysql': test_mysql,
        'full': test_full_invocation,
        'all': run_all_tests,
    }
    
    test_func = test_map.get(args.test, run_all_tests)
    
    try:
        if args.test == 'all':
            success = test_func()
        else:
            success = test_func()
        
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
