#!/usr/bin/env python3
"""
Test script for DigitalOcean MySQL connection and state management

Usage:
    python test_mysql_state.py
    python test_mysql_state.py --test-full
    python test_mysql_state.py --create-table
    python test_mysql_state.py --cleanup
    python test_mysql_state.py --test-full --cleanup

This script tests:
- MySQL connection and credentials using SQLAlchemy
- State table creation (ticker_progress)
- State save/retrieve operations
- R2 bucket cleanup (when using --cleanup flag)
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
import boto3
from botocore.config import Config

from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

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


def get_engine():
    """Create SQLAlchemy engine for MySQL connection"""
    try:
        db_host = os.getenv('DB_HOST')
        db_port = int(os.getenv('DB_PORT', '3306'))
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        db_name = os.getenv('DB_NAME', 'news_collector')
        
        # Create database connection string
        db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Create engine with NullPool for Lambda environment
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            poolclass=NullPool,
            connect_args={'connect_timeout': 10}
        )
        return engine
    except Exception as e:
        print(f"✗ Engine creation failed: {e}")
        return None


def get_session(engine):
    """Create SQLAlchemy session"""
    try:
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal()
    except Exception as e:
        print(f"✗ Session creation failed: {e}")
        return None


def get_r2_client():
    """Create Cloudflare R2 client"""
    try:
        account_id = os.getenv('R2_ACCOUNT_ID')
        access_key = os.getenv('R2_ACCESS_KEY')
        secret_key = os.getenv('R2_SECRET_KEY')
        
        if not all([account_id, access_key, secret_key]):
            print("✗ Missing R2 credentials: R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY")
            return None
        
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto',
            config=Config(signature_version='s3v4')
        )
        return client
    except Exception as e:
        print(f"✗ R2 client initialization failed: {e}")
        return None


def test_connection():
    """Test basic MySQL connection"""
    print("\n" + "="*60)
    print("Testing MySQL Connection")
    print("="*60)
    
    print(f"Host: {os.getenv('DB_HOST')}")
    print(f"Port: {os.getenv('DB_PORT', '3306')}")
    print(f"User: {os.getenv('DB_USER')}")
    print(f"Database: {os.getenv('DB_NAME', 'news_collector')}")
    
    engine = get_engine()
    if engine is None:
        return False
    
    try:
        # Test connection
        with engine.connect() as connection:
            # Get MySQL version
            result = connection.execute("SELECT VERSION()")
            version = result.fetchone()[0]
            print(f"\n✓ Connected successfully!")
            print(f"✓ MySQL Version: {version}")
            
            # Get current database
            result = connection.execute("SELECT DATABASE()")
            db = result.fetchone()[0]
            print(f"✓ Current Database: {db}")
            
            # Get current user
            result = connection.execute("SELECT USER()")
            user = result.fetchone()[0]
            print(f"✓ Current User: {user}")
            
            # List tables
            result = connection.execute("SHOW TABLES")
            tables = [row[0] for row in result.fetchall()]
            print(f"✓ Tables in database: {tables}")
        
        return True
        
    except SQLAlchemyError as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        engine.dispose()


def create_table():
    """Create ticker_progress table if it doesn't exist using SQLAlchemy"""
    print("\n" + "="*60)
    print("Creating ticker_progress Table")
    print("="*60)
    
    engine = get_engine()
    if engine is None:
        return False
    
    try:
        # Create all tables defined in Base
        Base.metadata.create_all(engine)
        print("✓ Table 'ticker_progress' created or already exists")
        
        # Check table structure
        with engine.connect() as connection:
            result = connection.execute("DESCRIBE ticker_progress")
            columns = result.fetchall()
            print("\nTable structure:")
            for col in columns:
                print(f"  - {col[0]:<25} {col[1]}")
        
        return True
        
    except SQLAlchemyError as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        engine.dispose()


def test_state_operations():
    """Test state save and retrieve operations using SQLAlchemy ORM"""
    print("\n" + "="*60)
    print("Testing State Operations")
    print("="*60)
    
    engine = get_engine()
    if engine is None:
        return False
    
    session = get_session(engine)
    if session is None:
        return False
    
    try:
        # Test data
        test_ticker_index = 42
        test_ticker = "AAPL"
        test_articles = 15
        
        print(f"\nTest Data:")
        print(f"  Ticker Index: {test_ticker_index}")
        print(f"  Ticker: {test_ticker}")
        print(f"  Articles: {test_articles}")
        
        # Save progress (upsert using ORM)
        print("\n1. Saving progress...")
        
        existing = session.query(TickerProgress).filter(
            TickerProgress.state_id == 'ticker_progress'
        ).first()
        
        if existing:
            existing.last_index = test_ticker_index
            existing.last_ticker = test_ticker
            existing.articles_processed += test_articles
            existing.last_update = datetime.utcnow()
        else:
            new_record = TickerProgress(
                state_id='ticker_progress',
                last_index=test_ticker_index,
                last_ticker=test_ticker,
                articles_processed=test_articles
            )
            session.add(new_record)
        
        session.commit()
        print("✓ Progress saved successfully")
        
        # Retrieve progress
        print("\n2. Retrieving progress...")
        result = session.query(TickerProgress).filter(
            TickerProgress.state_id == 'ticker_progress'
        ).first()
        
        if result:
            print("✓ Progress retrieved successfully")
            print(f"  State ID: {result.state_id}")
            print(f"  Last Index: {result.last_index}")
            print(f"  Last Ticker: {result.last_ticker}")
            print(f"  Articles Processed: {result.articles_processed}")
            print(f"  Last Update: {result.last_update}")
        else:
            print("✗ No progress found")
            return False
        
        # Test update
        print("\n3. Testing update (add more articles)...")
        
        existing = session.query(TickerProgress).filter(
            TickerProgress.state_id == 'ticker_progress'
        ).first()
        
        if existing:
            existing.last_index = test_ticker_index + 1
            existing.last_ticker = "GOOGL"
            existing.articles_processed += 10
            existing.last_update = datetime.utcnow()
            session.commit()
        
        result = session.query(TickerProgress).filter(
            TickerProgress.state_id == 'ticker_progress'
        ).first()
        
        if result:
            print("✓ Update successful")
            print(f"  New Last Index: {result.last_index}")
            print(f"  New Last Ticker: {result.last_ticker}")
            print(f"  Total Articles: {result.articles_processed}")
        
        # View all records
        print("\n4. All records in table:")
        records = session.query(TickerProgress).order_by(TickerProgress.last_update.desc()).all()
        
        if records:
            for record in records:
                print(f"  - {record.state_id}: Index={record.last_index}, "
                      f"Ticker={record.last_ticker}, Articles={record.articles_processed}")
        else:
            print("  No records found")
        
        return True
        
    except SQLAlchemyError as e:
        print(f"✗ Error: {e}")
        session.rollback()
        return False
    finally:
        session.close()
        engine.dispose()


def cleanup_test_data():
    """Clean up test data from both MySQL and R2 storage using SQLAlchemy"""
    print("\n" + "="*60)
    print("Cleaning Up Test Data")
    print("="*60)
    
    success = True
    
    # Clean up MySQL data
    print("\n1. Cleaning MySQL database...")
    engine = get_engine()
    if engine is None:
        print("✗ Failed to create engine")
        success = False
    else:
        session = get_session(engine)
        if session is None:
            print("✗ Failed to create session")
            success = False
        else:
            try:
                # Delete test record using ORM
                session.query(TickerProgress).filter(
                    TickerProgress.state_id == 'ticker_progress'
                ).delete()
                session.commit()
                print("✓ MySQL test data cleaned up")
            except SQLAlchemyError as e:
                print(f"✗ MySQL cleanup error: {e}")
                session.rollback()
                success = False
            finally:
                session.close()
                engine.dispose()
    
    # Clean up R2 news files
    print("\n2. Cleaning R2 news bucket...")
    bucket_name = os.getenv('NEWS_BUCKET')
    if not bucket_name:
        print("⚠ NEWS_BUCKET environment variable not set, skipping R2 cleanup")
    else:
        r2_client = get_r2_client()
        if r2_client is None:
            print("✗ Failed to initialize R2 client")
            success = False
        else:
            try:
                print(f"Listing objects in bucket: {bucket_name}")
                
                # List all objects in the bucket
                paginator = r2_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name)
                
                deleted_count = 0
                total_objects = 0
                
                for page in pages:
                    if 'Contents' not in page:
                        continue
                    
                    # Delete objects in batches
                    for obj in page['Contents']:
                        total_objects += 1
                        key = obj['Key']
                        try:
                            r2_client.delete_object(Bucket=bucket_name, Key=key)
                            deleted_count += 1
                            print(f"  Deleted: {key}")
                        except Exception as e:
                            print(f"  ✗ Failed to delete {key}: {e}")
                            success = False
                
                if deleted_count > 0:
                    print(f"✓ R2 cleanup complete: {deleted_count} files deleted out of {total_objects}")
                else:
                    print(f"✓ R2 bucket is empty or no objects found")
                    
            except Exception as e:
                print(f"✗ R2 cleanup error: {e}")
                success = False
    
    return success


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Test DigitalOcean MySQL connection and R2 storage')
    parser.add_argument('--test-full', action='store_true', help='Run all tests including state operations')
    parser.add_argument('--create-table', action='store_true', help='Create ticker_progress table')
    parser.add_argument('--cleanup', action='store_true', help='Clean up test data from MySQL and R2 bucket')
    
    args = parser.parse_args()
    
    print(f"\n🧪 MySQL Connection Test Utility")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {
        'Connection': test_connection(),
    }
    
    if args.create_table or args.test_full:
        results['Create Table'] = create_table()
    
    if args.test_full:
        results['State Operations'] = test_state_operations()
    
    if args.cleanup:
        results['Cleanup'] = cleanup_test_data()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Check configuration and logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
