import os
import json
import tiktoken
import typer
import sys
import httpx
import pymysql.cursors
import logging
import lunary
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, retry_if_result

load_dotenv()
app = typer.Typer(name="predictor")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}
httpxClient = httpx.Client(http2=True, timeout=30, follow_redirects=True, headers=headers)

console_output_handler = logging.StreamHandler(sys.stdout)
console_output_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

logger = logging.getLogger("Stock Trend Predictor")
logger.setLevel(logging.INFO)
logger.addHandler(console_output_handler)
        
link_system_prompt = "You are provided with a list of links found on a webpage. \
You can decide which of the links would be most relevant to include in the financial information about the stock. \
such as links to a News page, Statistics page, or Analysis page.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "News page", "url": "https://full.url/goes/here/news"},
        {"type": "Analysis page": "url": "https://another.full.url/analysis"}
    ]
}
"""
system_prompt="You are an assistant that analyzes the contents of a website about a stock \
and provides a prediction on the up-trend, down-trend, or stay flat in the next 5 days, ignoring text that might be navigation-related. \
If it includes news or announcements, then summarize these too in markdown. \
You should respond in JSON as in this example:"
system_prompt += """
{ 
    "direction": "Either UP, DOWN, FLAT or UNKNOWN", 
    "report": "A markdown summary of the news or announcements"
}
"""
# Define the conditions for retrying based on exception types
def is_retryable_exception(exception: Exception) -> bool:
    # Retry on specific HTTPX exceptions

    return isinstance(exception, (httpx.TimeoutException, httpx.ConnectError, httpx.StreamError))

# Define the conditions for retrying based on HTTP status codes
def is_retryable_status_code(response: httpx.Response) -> bool:
    # Retry on specific HTTP status codes

    return response.status_code in [500, 502, 503, 504]

# Define the conditions for retrying based on response content
def is_retryable_content(response: httpx.Response) -> bool:
    # Retry if the response contains specific content indicating a block

    return "you are blocked" in response.text.lower()

@retry(
    retry=(retry_if_exception_type(is_retryable_exception) | retry_if_result(is_retryable_status_code) | retry_if_result(is_retryable_content)),
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
)
def fetch_url(url: str) -> httpx.Response:
    """
    Fetch a URL with retries for specific exceptions and status codes.
    Args:
        url (str): The URL to fetch.
    Returns:
        httpx.Response: The HTTP response object.
    """
    try:
        response = httpxClient.get(url)
        response.raise_for_status()
        return response
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise e

class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """
    def __init__(self, url: str):
        
        self.url = url
        self.body = ""
        self.title = "No title found"
        self.text = ""
        self.links = []
        
        try:
            response = fetch_url(url)
            response.raise_for_status()
            self.body = response.text
            soup = BeautifulSoup(self.body, 'html.parser')
            self.title = soup.title.string if soup.title else "No title found"
            if soup.body:
                for irrelevant in soup.body(["script", "style", "img", "input"]):
                    irrelevant.decompose()
                self.text = soup.body.get_text(separator="\n", strip=True)
            links = [link.get('href') for link in soup.find_all('a')]
            self.links = [link for link in links if link]
        except httpx.TimeoutException:
            logger.error(f"Error: Request to {url} timed out.")
        except httpx.RequestError as e:
            logger.error(f"Error: Failed to fetch {url}. Reason: {e}")

    def get_contents(self) -> str:
        # Return the text contents of the webpage.

        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
    
class PredictAgent:

    def __init__(self, llm: str, site: str):
        self.llm = llm
        self.site = site
        self.encoding = tiktoken.get_encoding('o200k_base')
        self.max_input_tokens = 50000
        self.system_prompt = system_prompt
        self.link_system_prompt = link_system_prompt
        self.df = pd.read_csv("symbol-list.csv")
        self.tickers = self.df[self.site].tolist()
        logger.info(f"Tickers: {self.tickers}")
        
        if self.llm == "gemini":
            self.openai = OpenAI(
                api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.model = "gemini-2.0-flash"
        elif self.llm == "xai":
            self.openai = OpenAI(
                api_key=os.getenv("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
            self.model = "grok-2-latest"
        else:
            self.openai = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.model = "gpt-4o-mini"

        lunary.monitor(self.openai)

    def get_links_user_prompt(self, website: Website, ticker: str) -> str:
        # Generate the user prompt for getting relevant links.

        user_prompt = f"Here is the list of links on the website of {website.url} about the stock code \'{ticker}\' - "
        user_prompt += "please decide which of these are relevant web links for all related information about the stock code, respond with the full https URL in JSON format. \
    Do not include Terms of Service, Privacy, email links.\n"
        user_prompt += "Links (some might be relative links):\n"
        user_prompt += "\n".join(website.links)
        return user_prompt

    def get_links(self, website: Website, ticker: str) -> dict:
        # Get relevant links from the website.

        links_prompt = self.get_links_user_prompt(website, ticker)
        logger.info(f"Getting links for {website.url}")
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": link_system_prompt},
                {"role": "user", "content": links_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=5000,
            tags = [ticker, self.llm, self.site, 'getlinks']
        )
        result = response.choices[0].message.content
        return json.loads(result)

    def get_all_details(self, website: Website, ticker: str) -> str: 
        # Get all details from the website and its links.

        links = self.get_links(website, ticker)
        logger.info(f"Finish getting links for {website.url}")
        link_count = 0
        result = "Landing page:\n"
        result += website.get_contents()
        for link in links["links"]:
            prev_result = result
            result += f"\n\n{link['type']}\n"
            result += Website(link["url"]).get_contents()
            if len(self.encoding.encode(result)) > self.max_input_tokens:
                result = prev_result
                break
            link_count += 1
        logger.info(f"Links used: {link_count}")
        return result

    def user_prompt_for(self, website: Website, ticker: str) -> str:
        # Generate the user prompt for the OpenAI API call.

        u_prompt = f"You are looking at a website titled {website.title}"
        u_prompt += f"\nThe contents of this website is as follows; "
        u_prompt += self.get_all_details(website, ticker)

        logger.info(f"Token length: {len(self.encoding.encode(u_prompt))}")
        return u_prompt

    def messages_for(self, website: Website, ticker: str) -> list[ChatCompletionMessageParam]:
        # Generate the messages for the OpenAI API call.

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_for(website, ticker)}
        ]

    def predict(self, url: str, ticker: str) -> ChatCompletionMessage:
        # Make a prediction of stock trend by scraping info from the website

        logger.info(f"Using {url} for {ticker}")
        website = Website(url)
        try:
            response = self.openai.chat.completions.create(
                model = self.model,
                messages = self.messages_for(website, ticker),
                response_format={"type": "json_object"},
                max_tokens=5000,
                tags = [ticker, self.llm, self.site, 'predict']
            )
            logger.info(f"Token usgae stat: {response.usage}")
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"direction": "UNKNOWN", "report": "No prediction available"}
        return response.choices[0].message
    
    def run(self) -> None:
        # Run the prediction for each ticker

        cur_date = datetime.now().strftime("%Y-%m-%d")
        for index, ticker in enumerate(self.tickers):
            if ticker == "[PLACEHOLDER]":
                continue
            site_urls = {
                "cnbc": f"https://www.cnbc.com/quotes/{ticker}",
                "investopedia": f"https://www.investopedia.com/search?q={ticker}",
                "yahoo": f"https://finance.yahoo.com/quote/{ticker}/",
                "morningstar": f"https://www.morningstar.com/stocks/{ticker}/quote",
            }
            url = site_urls[self.site]
            root_ticker = self.df.iloc[index]["yahoo"]
            result = self.predict(url, ticker)
            parsed_result = {"direction": "UNKNOWN", "report": "No prediction available"}
            try:
                if result.content:
                    parsed_result = json.loads(result.content)
                    logger.info(f"Prediction for {ticker} from {url}: {parsed_result['direction']}")
                    db_connection = pymysql.connect(
                        host=os.getenv("MYSQL_HOST"),
                        user=os.getenv("MYSQL_USER"),
                        password=os.getenv("MYSQL_PASSWORD"),
                        port=int(os.getenv("MYSQL_PORT")),
                        database=os.getenv("MYSQL_DATABASE"),
                        cursorclass=pymysql.cursors.DictCursor)
                    with db_connection:
                        with db_connection.cursor() as cursor:
                            predict_sql = "INSERT INTO `trend_predict` (`Date`, `model`, `websites`, `symbol`, `trend`) VALUES (%s, %s, %s, %s, %s)"
                            cursor.execute(predict_sql, (cur_date, self.llm, url, root_ticker, parsed_result["direction"]))
                        db_connection.commit()
            except Exception as e:
                logger.error(f"Error: {e}")

@app.command()
def run_predict(llm: str, site: str) -> None:
    
    if site not in ["cnbc", "investopedia", "yahoo", "morningstar"]:
        logger.error("Invalid site. Choose from cnbc, investopedia, yahoo, morningstar.")
        raise typer.Exit(code=1)
    if llm not in ["openai", "gemini", "xai"]:
        logger.error("Invalid LLM. Choose from openai, gemini, xai.")
        raise typer.Exit(code=1)
    agent = PredictAgent(llm, site)
    agent.run()

if __name__ == "__main__":
    app()