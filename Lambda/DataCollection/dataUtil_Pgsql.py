import os
import logging
import pandas as pd
from dotenv import load_dotenv
from os import environ
from sqlalchemy import create_engine,text

# Load environment variables
load_dotenv(dotenv_path='/Users/huangjunyi/Documents/Github/Fin-Agents/.env')

dbconn = None
BASE_DIR        = os.getenv('DATA_DIR', './DATA')


def get_DBengine():
    global dbconn
    if dbconn is None:
        hostname= environ.get("RHOST")
        uname=environ.get("DBUSER")
        pwd=environ.get("DBPWD")
        DB = environ.get("DB")
        DBPORT = environ.get("PORT")

        dbpath = "postgresql://{user}:{pw}@{host}:{port}/{db}".format(host=hostname, db=DB, user=uname, pw=pwd,port=DBPORT)
        logging.info(f'setup DBengine to {dbpath}')
        # Create SQLAlchemy engine to connect to MySQL Database
        dbconn = create_engine(dbpath)
        logging.info(f'dbconn=>{dbconn}')
    return dbconn

def save_news_to_postgres(news_items: list[dict], table_name: str):
    engine = get_DBengine()
    df = pd.DataFrame(news_items)
    sql = text(f"""
        INSERT INTO {table_name} (date, ticker, title, link, content)
        VALUES (:date, :ticker, :title, :link, :content)
        ON CONFLICT (ticker, title)
        DO UPDATE SET
            date=EXCLUDED.date,
            link=EXCLUDED.link,
            content=EXCLUDED.content
    """)
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(sql, {
                "date": row.get('date'),
                "ticker": row.get('ticker'),
                "title": row.get('title'),
                "link": row.get('link'),
                "content": row.get('content')
            })
    logging.info(f"Saved {len(df)} news items to PostgreSQL table '{table_name}' (upsert by ticker+title)")


def save_quarterly_financials_to_postgres(ticker, data_dict, fin_table_name):
    rows = []
    for date, items in data_dict.items():
        for item, value in items.items():
            rows.append({
                "ticker": ticker,
                "date": pd.to_datetime(date).date(),
                "item": item,
                "value": None if pd.isna(value) else float(value) if value is not None else None
            })
    engine = get_DBengine()
    sql = text(f"""
        INSERT INTO {fin_table_name} (ticker, date, item, value)
        VALUES (:ticker, :date, :item, :value)
        ON CONFLICT (ticker, date, item)
        DO UPDATE SET
            value = EXCLUDED.value
    """)
    with engine.begin() as conn:
        for row in rows:
            conn.execute(sql, row)


def save_statistics_to_postgres(stats: dict, stat_table_name):
    engine = get_DBengine()
    with engine.begin() as conn:
        sql = text(f"""
            INSERT INTO {stat_table_name} (ticker, date, "Market Cap", "Enterprise Value", "Trailing P/E", "Forward P/E", "PEG Ratio", "Price/Sales (ttm)", "Price/Book", "Enterprise/Revenue", "Enterprise/EBITDA", "EBITDA")
            VALUES (:ticker, :date, :MarketCap, :EnterpriseValue, :TrailingPE, :ForwardPE, :PEGRatio, :PriceSales, :PriceBook, :EnterpriseRevenue, :EnterpriseEBITDA, :EBITDA)
            ON CONFLICT (ticker, date)
            DO UPDATE SET
                "Market Cap" = EXCLUDED."Market Cap",
                "Enterprise Value" = EXCLUDED."Enterprise Value",
                "Trailing P/E" = EXCLUDED."Trailing P/E",
                "Forward P/E" = EXCLUDED."Forward P/E",
                "PEG Ratio" = EXCLUDED."PEG Ratio",
                "Price/Sales (ttm)" = EXCLUDED."Price/Sales (ttm)",
                "Price/Book" = EXCLUDED."Price/Book",
                "Enterprise/Revenue" = EXCLUDED."Enterprise/Revenue",
                "Enterprise/EBITDA" = EXCLUDED."Enterprise/EBITDA",
                "EBITDA" = EXCLUDED."EBITDA"
        """)
        params = {
            "ticker": stats.get("ticker"),
            "date": stats.get("date"),
            "MarketCap": stats.get("Market Cap"),
            "EnterpriseValue": stats.get("Enterprise Value"),
            "TrailingPE": stats.get("Trailing P/E"),
            "ForwardPE": stats.get("Forward P/E"),
            "PEGRatio": stats.get("PEG Ratio"),
            "PriceSales": stats.get("Price/Sales (ttm)"),
            "PriceBook": stats.get("Price/Book"),
            "EnterpriseRevenue": stats.get("Enterprise/Revenue"),
            "EnterpriseEBITDA": stats.get("Enterprise/EBITDA"),
            "EBITDA": stats.get("EBITDA"),
        }
        conn.execute(sql, params)