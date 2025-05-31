# Financial Agents for market prediction, analysis

## YFinance Data Collection (by ticker, date, data-type)

- **Implementation** by the AWS Lambda function
- Store the data on S3 organized by "Data Type", "Sub-Type", "Update period" and "Date"

| Data Type      | Source URL                                                                 | Sub-Type                                                                 | Update Period |
|----------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|---------------|
| News           | [https://finance.yahoo.com/quote/TSLA/news/](https://finance.yahoo.com/quote/TSLA/news/) |                                                                           | Daily         |
| Community      | [https://finance.yahoo.com/quote/TSLA/community/](https://finance.yahoo.com/quote/TSLA/community/) |                                                                           | Daily         |
| Statistics     | [https://finance.yahoo.com/quote/TSLA/key-statistics/](https://finance.yahoo.com/quote/TSLA/key-statistics/) | - Valuation Measure (Quarterly free)<br>- Financial Highlights<br>- Trading Information | Quarterly     |
| Financials     | [https://finance.yahoo.com/quote/TSLA/financials/](https://finance.yahoo.com/quote/TSLA/financials/) | - Income Statement                                                        | Annual, Quarterly |
|                | [https://finance.yahoo.com/quote/TSLA/balance-sheet/](https://finance.yahoo.com/quote/TSLA/balance-sheet/) | - Balance Sheet                                                           | Annual, Quarterly |
|                | [https://finance.yahoo.com/quote/TSLA/cash-flow/](https://finance.yahoo.com/quote/TSLA/cash-flow/) | - Cash Flow                                                               | Annual, Quarterly |
| Analysis       | [https://finance.yahoo.com/quote/TSLA/analysis/](https://finance.yahoo.com/quote/TSLA/analysis/) | Earnings Estimate, Revenue Estimate, Earnings History, EPS Trend, EPS Revisions, Growth Estimates, Top Analysts (%), Upgrades & Downgrades | Daily         |
| Holders        | [https://finance.yahoo.com/quote/TSLA/holders/](https://finance.yahoo.com/quote/TSLA/holders/) |                                                                           | Daily         |
| Sustainability | [https://finance.yahoo.com/quote/TSLA/sustainability/](https://finance.yahoo.com/quote/TSLA/sustainability/) |                                                                           | Daily         |


### File Structure Recommendation for S3

To make the data easily accessible for fine-tuning an LLM model, we should organize the S3 bucket with a hierarchical structure that reflects the key dimensions: Ticker, Date, and Data Type. Additionally, we’ll use a format that’s efficient for ML workflows, such as JSON or Parquet, since these formats are widely supported by data processing libraries like Pandas and are suitable for large-scale model training.

s3://\<bucket-name\>/\<ticker\>/\<date\>/\<data-type\>/\<sub-type\>.json

* \<bucket-name\>: The S3 bucket name (e.g., financial-data-for-llm).
* \<ticker\>: The stock ticker symbol (e.g., TSLA, AAPL).
* \<date\>: The date of the data in YYYY-MM-DD format (e.g., 2025-05-29).
* \<data-type\>: The type of data (e.g., news, community, statistics, financials, analysis, holders, sustainability).
* \<sub-type\>: The subtype of the data, if applicable (e.g., valuation-measure, income-statement, balance-sheet, etc.). If there’s no subtype, this can be omitted or set to a default like data.
.json: Store the data in JSON format for simplicity and compatibility. Alternatively, Parquet could be used for better performance in ML workflows, but JSON is more straightforward for this initial setup.

### Example File Structure
For the ticker TSLA on May 29, 2025, the structure might look like:

s3://financial-data-for-llm/TSLA/2025-05-29/news/data.json  
s3://financial-data-for-llm/TSLA/2025-05-29/community/data.json  
s3://financial-data-for-llm/TSLA/2025-05-29/statistics/valuation-measure.json  
s3://financial-data-for-llm/TSLA/2025-05-29/statistics/financial-highlights.json  
s3://financial-data-for-llm/TSLA/2025-05-29/statistics/trading-information.json  
s3://financial-data-for-llm/TSLA/2025-05-29/financials/income-statement.json  
s3://financial-data-for-llm/TSLA/2025-05-29/financials/balance-sheet.json  
s3://financial-data-for-llm/TSLA/2025-05-29/financials/cash-flow.json  
s3://financial-data-for-llm/TSLA/2025-05-29/analysis/earnings-estimate.json  
s3://financial-data-for-llm/TSLA/2025-05-29/holders/data.json  
s3://financial-data-for-llm/TSLA/2025-05-29/sustainability/data.json  

### Specification for the Python Scraping Process
Below is a detailed specification for a Python developer to implement the data scraping and storage process. The process will use the yfinance library to fetch data (since the URLs are from Yahoo Finance), boto3 to interact with S3, and will handle the specified data types and update periods.

### Process Overview
1. Input: A list of tickers (e.g., ["TSLA", "AAPL", "GOOGL"]).
2. Scrape Data: For each ticker, scrape the required data types (News, Community, Statistics, Financials, Analysis, Holders, Sustainability) using the yfinance library.
3. Organize Data: Structure the scraped data according to the data type and subtype, and format it as JSON.
4. Store in S3: Upload the data to the S3 bucket with the specified file structure, using the current date (2025-05-29 in this case).
5. Handle Update Periods: Only scrape data if it needs to be updated based on its update period (Daily, Quarterly, Annual).

### Logging:
* Imported the logging module and configured it to log to both a file and the console.
* The log file is named with a timestamp (e.g., scrape_financial_data_20250529_151745.log).
* Log levels used:
    * INFO: For general progress messages (e.g., starting the process, saving to S3).
    * DEBUG: For detailed steps (e.g., fetching data, checking last updated dates).
    * ERROR: For error messages (e.g., failed to fetch or save data).
    * WARNING: For unexpected scenarios (e.g., unknown update period).
    
### Dependencies
The Python script requires the following libraries:
* yfinance: For scraping Yahoo Finance data.
* boto3: For interacting with AWS S3.
* python-dateutil: For date handling (optional, if not already included).

Install them using:

`
pip install yfinance boto3 python-dateutil
`

### Environment Setup
* **S3 Bucket**: Create an S3 bucket named financial-data-for-llm (or your chosen name) in the specified region.

### Notes for the Developer
* Error Handling: The script includes basic error handling, but you may need to add more robust logging or retry logic for production use.
* Community Data: The yfinance library doesn’t directly support scraping community data from Yahoo Finance. You’ll need to implement custom web scraping (e.g., using requests and beautifulsoup4) to fetch this data.
* Analysis Subtypes: The yfinance library may not directly map to all analysis subtypes (e.g., "earnings-estimate", "revenue-estimate"). You may need to extend the fetch_data function to handle these cases, possibly by scraping the raw HTML pages or finding alternative APIs.
* Rate Limiting: Yahoo Finance may impose rate limits. Add delays or retry logic if needed to avoid being blocked.
* Scalability: For large ticker lists, consider parallelizing the scraping process using multiprocessing or a task queue like Celery.

### How This Supports LLM Fine-Tuning
* Data Access: The S3 structure allows easy retrieval of data by ticker, date, or data type, which is useful for creating training datasets (e.g., all news data for TSLA in 2025).
* Format: JSON is easily parsed by Python libraries like Pandas, which can be used to preprocess data for LLM fine-tuning.
* Granularity: The separation by subtype ensures that different data aspects (e.g., financials vs. news) can be selectively used for training, depending on the model’s needs.

