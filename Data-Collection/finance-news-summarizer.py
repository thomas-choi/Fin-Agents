from bs4 import BeautifulSoup
from datetime import datetime
from types import SimpleNamespace
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import OpenAI
from supabase import create_client
import os
import json
from curl_cffi import requests

load_dotenv()

#Gordon's Azure OpenAI configuration
#llm = AzureOpenAI(
    #azure_endpoint="https://gordo-m82vumaa-eastus2.cognitiveservices.azure.com/",
    #azure_deployment=os.environ.get("AZURE_DEPLOYMENT"),
    #api_version=os.environ.get("AZURE_API_VERSION"),
    #api_key=os.environ.get("AZURE_API_KEY")
#)
#supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

llm = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    #model="gpt-4o-mini",
    )

#Gordon's Supabase configuration
#supabase = create_client("https://kydgdecdmsdwdvgytqqp.supabase.co", os.environ.get("SUPABASE_KEY"))

#Jeri's Supabase configuration
supabase = create_client("https://mprqvvcbtqnltjnyznsd.supabase.co", os.environ.get("SUPABASE_KEY"))


FEEDTYPES = {
    'NASDAQ': 'API'
}
FEEDS = {
    'NASDAQ': {
        'AAPL': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=AAPL|STOCKS&offset=0&limit=10&fallback=true',
        'MSFT': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=MSFT|STOCKS&offset=0&limit=10&fallback=true',
        'GOOGL': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=GOOGL|STOCKS&offset=0&limit=10&fallback=true',
        'AMZN': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=AMZN|STOCKS&offset=0&limit=10&fallback=true',
        'TSLA': 'https://www.nasdaq.com/api/news/topic/articlebysymbol?q=TSLA|STOCKS&offset=0&limit=10&fallback=true'
    }
}

system_prompt = """You are an expert in analyzing financial news."""
user_prompt_base = """
Given the ticker and a piece of text about the company represented by that ticker in stock market,
determine whether that piece of text are relevant in predicting the one week ahead price movement of stock represented by that ticker.
Relevancy should be answered with either a Yes or No.
Consider the following text content to be irrelevant:
1. About general stock market and describing price performances of a number of tickers
2. Discussing after-hour market performance
3. Discussing stock option pf the ticker 

If the piece of text is determined to be relevant, then summarize that piece of text in less than 200 words, 
otherwise give it an empty string as summary.
Do not reference the text in the summary by saying 'the text describes', 'the report describes' and so on.
Write the summary as if it is a brand new article.
Respond in JSON as in this example:
{ 
    "result": {
        "relevant": "Yes"，
        "summary: "Summary of the given text....."
    }
}"""

def get_message(ticker, text):
    user_prompt = user_prompt_base + f"\nTicker symbol: {ticker}\n\nText: {text}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def get_llm_response(message):
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=message,
        response_format={"type": "json_object"},
        max_completion_tokens=1500
    )
    return response.choices[0].message

def get_all_articles():
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
    for article in all_articles:
        current[article[2]] = False
    for article in all_articles:
        link = article[1].links[0].content
        if link in visited:
            continue
        visited.append(link)
        article_content = get_article_content(link, article[0])
        message = get_message(article[2], article_content)
        response = get_llm_response(message)
        result = json.loads(response.content)
        if result and result["result"] and result["result"]["relevant"] and result["result"]["relevant"] == "Yes":
            print(f"Found relevant article about {article[2]}")
            article[4] = True
            if not current[article[2]]:
                try:
                    # 先检查是否有该ticker的记录
                    existing = supabase.table("finance_news").select("id").eq("ticker", article[2]).execute()
                    if existing.data and len(existing.data) > 0:
                        supabase.table("finance_news").update({"current":False}).eq("ticker", article[2]).execute()
                    current[article[2]] = True
                except Exception as e:
                    print(f"Update current=False failed for {article[2]}: {e}")

                supabase.table("finance_news").update({"current":False}).eq("ticker", article[2]).execute()
                current[article[2]] = True
            supabase_response = (
                supabase.table("finance_news")
                .insert({
                    "link": link,
                    "ticker": article[2],
                    "title": article[1].title.content,
                    "summary": result.get("result", {}).get("summary", "Summary not available"),
                    "date": datetime.now().strftime("%Y-%m-%d")
                })
                .execute()
            )
            if supabase_response.data and len(supabase_response.data) == 1:
                print(f"Inserted article about {article[2]} into Supabase")
            else:
                print(f"Failed to insert article about {article[2]} into Supabase")
        else:
            print(f"Found irrelevant article about {article[2]}")
        article[3] = result.get("result", {}).get("summary", "")
    
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
                "pub_date": {"content":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
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

if __name__ == '__main__':
    get_all_articles()


