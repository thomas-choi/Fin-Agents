import pandas as pd
import schedule
import time
from datetime import datetime, timedelta
import pytz
import logging
import threading
import socket
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

from dotenv import load_dotenv
from os import environ

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
load_dotenv()
DEBUG=environ.get("DEBUG")
if DEBUG == "debug":
    logger.setLevel(logging.DEBUG)
    print("Logging is DEBUG.")

IBHOST=environ.get("IBHOST")
IBPORT=int(environ.get("IBPORT"))

# Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define time zones
HKT = pytz.timezone('Asia/Hong_Kong')
MST = pytz.timezone('America/Denver')
NYT = pytz.timezone('America/New_York')

# HSI futures trading hours (HKT)
TRADING_HOURS = [
    ("08:45", "12:00"),  # Pre-open (8:45) to morning session
    ("13:00", "16:30"),  # Afternoon session
    ("17:15", "03:00")   # After-hours (T+1)
]

# Check if current time is within HSI trading hours
def is_trading_time():
    now_hkt = datetime.now(HKT)
    current_time = now_hkt.strftime("%H:%M")
    
    for start, end in TRADING_HOURS:
        if start <= current_time <= end:
            return True
        # Handle after-hours crossing midnight
        if end < start:
            end_time = (datetime.strptime(end, "%H:%M") + timedelta(days=1)).time()
            if now_hkt.time() <= end_time or now_hkt.time() >= datetime.strptime(start, "%H:%M").time():
                return True
    return False

# Get start of current trading day (8:45 AM HKT)
def get_trading_day_start():
    now_hkt = datetime.now(HKT)
    # If after 3:00 AM HKT, use today's 8:45 AM; else use yesterday's
    if now_hkt.hour < 3:
        trading_day = now_hkt.date() - timedelta(days=1)
    else:
        trading_day = now_hkt.date()
    
    # Set to 8:45 AM HKT
    start_time = datetime.combine(trading_day, datetime.strptime("08:45", "%H:%M").time(), tzinfo=HKT)
    return start_time

# IB API Wrapper
class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.error_occurred = False
        self.error_msg = ""
        self.data_ready = threading.Event()
        self.connected = threading.Event()
        self.contract_details = None
        self.contract_resolved = threading.Event()
        
    def connectAck(self):
        logging.info("Connected to TWS")
        self.connected.set()
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        logging.error(f"Error {errorCode}: {errorString}")
        if errorCode in [200, 162, 165]:  # Contract not found, pacing violation
            self.error_occurred = True
            self.error_msg = errorString
            self.data_ready.set()
        elif errorCode == 502:  # Not connected
            self.error_occurred = True
            self.error_msg = "TWS not connected"
            self.connected.set()
        
    def contractDetails(self, reqId, contractDetails):
        self.contract_details = contractDetails.contract
        self.contract_resolved.set()
        
    def contractDetailsEnd(self, reqId):
        self.contract_resolved.set()
        
    def historicalData(self, reqId, bar: BarData):
        logging.debug(f"bar.date is {bar.date}")
        date_str = bar.date.split(' ')[0:2]  # Take YYYYMMDD HH:MM:SS
        date_str = ' '.join(date_str)  # Rejoin as YYYYMMDD HH:MM:SS
        timestamp = datetime.strptime(date_str, '%Y%m%d %H:%M:%S')
        timestamp = NYT.localize(timestamp)
        timestamp = timestamp.astimezone(HKT)
        logging.debug(f"saved timestamp:  {timestamp}")
        self.data.append({
            'Timestamp': timestamp,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })
        
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        logging.info(f"Historical data fetch completed: {start} to {end}")
        self.data_ready.set()

# Save data to CSV with duplicate prevention
def save_to_csv(data, csv_file='hsi_futures_realtime.csv'):
    if data:
        df = pd.DataFrame(data)
        logging.debug("==========    data      ===========")
        logging.debug(df)
        # Format Timestamp column to yyyy-mm-dd hh:mm:ss without time zone
        df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if pd.io.common.file_exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            existing_df['Timestamp'] = pd.to_datetime(existing_df['Timestamp'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df[~df['Timestamp'].isin(existing_df['Timestamp'])]
            if not df.empty:
                df.to_csv(csv_file, mode='a', index=False, header=False)
        else:
            df.to_csv(csv_file, mode='w', index=False)
        logging.info(f"Saved {len(df)} bars: {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")
        print(df)
    else:
        logging.info("No data to save")

# Initial fetch using ibapi
def initial_fetch2():
    logging.info("Performing initial fetch from start of trading day...")
    now_hkt = datetime.now(HKT)
    logging.debug(f"now_hkt: {now_hkt}")
       
    # Create IB app
    app = IBApp()
    try:
        # Retry connection
        for attempt in range(3):
            try:
                socket.setdefaulttimeout(10)
                app.connect(IBHOST, IBPORT, clientId=678)
                print("*** After app.connect()   ***")

                api_thread = threading.Thread(target=app.run)
                api_thread.start()
                
                if not app.connected.wait(timeout=15):
                    logging.warning(f"Connection attempt {attempt + 1} timed out")
                    raise TimeoutError("Connection to TWS timed out")
                
                if app.error_occurred:
                    raise Exception(app.error_msg)
                
                time.sleep(5)
                break
            except Exception as e:
                logging.warning(f"Connection attempt {attempt + 1} failed: {e}")
                app.disconnect()
                if api_thread.is_alive():
                    api_thread.join(timeout=5)
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise e
        
        # Define HSI futures contract
        contract = Contract()
        contract.symbol = 'HSI'
        contract.secType = 'FUT'
        contract.exchange = 'HKFE'
        contract.currency = 'HKD'
        # contract_str = environ.get("contract_month")
        contract_str = now_hkt.strftime('%Y%m')
        logging.info(f"Setting contract_str :  {contract_str}")
        contract.lastTradeDateOrContractMonth = now_hkt.strftime('%Y%m')
        logging.info(f"* contract month:  {contract}")

        # Resolve contract
        app.contract_details = None
        app.contract_resolved.clear()
        app.reqContractDetails(reqId=1, contract=contract)
        if not app.contract_resolved.wait(timeout=10):
            logging.warning("Timeout resolving contract")
            raise TimeoutError("Contract resolution timed out")
        if not app.contract_details:
            raise Exception("Failed to resolve contract")
        contract = app.contract_details
        
        # Calculate duration from start of trading day
        start_time = get_trading_day_start()
        duration_seconds = int((now_hkt - start_time).total_seconds())
        duration_str = f"{max(duration_seconds, 1800)} S"
        duration_str=environ.get("duration_str")
        logging.info(f"start_time:{start_time}, duration_str:{duration_str}")
        
        # Format endDateTime with HKT time zone
        end_date_time = now_hkt.strftime('%Y%m%d %H:%M:%S Asia/Hong_Kong')
        end_date_time=environ.get("end_date_time")
        # edt_time = datetime.strptime(end_date_time, '%Y%m%d %H:%M:%S')
        # edt_time = HKT.localize(edt_time)
        logging.info(f"** reqHistoricalData.endDateTime={end_date_time}")
        
        barSize=environ.get("barSize")
        logging.info(f"** reqHistoricalData.barSize={barSize}")

        # Request historical data
        app.data = []
        app.data_ready.clear()
        for attempt in range(3):
            try:
                app.reqHistoricalData(
                    reqId=2,  # Use different reqId to avoid conflict
                    contract=contract,
                    endDateTime=end_date_time,
                    durationStr=duration_str,
                    barSizeSetting=barSize,
                    whatToShow='TRADES',
                    useRTH=0,
                    formatDate=1,
                    keepUpToDate=False,
                    chartOptions=[]
                )
                if not app.data_ready.wait(timeout=60):
                    logging.warning("Timeout waiting for historical data")
                    raise TimeoutError("Historical data request timed out")
                if app.error_occurred:
                    raise Exception(app.error_msg)
                break
            except Exception as e:
                logging.warning(f"Data request attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)
                else:
                    raise e
        
        # Save data
        save_to_csv(app.data)
        
    except Exception as e:
        logging.error(f"Initial Fetch Error: {e}")
    finally:
        app.disconnect()
        if api_thread.is_alive():
            api_thread.join(timeout=5)
        socket.setdefaulttimeout(None)

# Real-time fetch using ibapi
def fetch_data():
    now_mst = datetime.now(MST)
    now_hkt = datetime.now(HKT)
    logging.info(f"Fetching data at {now_mst} MST ({now_hkt} HKT)")
    
    if is_trading_time():
        logging.info("Market is open, fetching data...")
        app = IBApp()
        try:
            socket.setdefaulttimeout(10)
            app.connect('192.168.11.111', 7497, clientId=1)
            api_thread = threading.Thread(target=app.run)
            api_thread.start()
            if not app.connected.wait(timeout=15):
                raise TimeoutError("Connection to TWS timed out")
            time.sleep(5)
            
            contract = Contract()
            contract.symbol = 'HSI'
            contract.secType = 'FUT'
            contract.exchange = 'HKFE'
            contract.currency = 'HKD'
            contract.lastTradeDateOrContractMonth = now_hkt.strftime('%Y%m')
            
            app.contract_details = None
            app.contract_resolved.clear()
            app.reqContractDetails(reqId=1, contract=contract)
            if not app.contract_resolved.wait(timeout=10):
                raise TimeoutError("Contract resolution timed out")
            if not app.contract_details:
                raise Exception("Failed to resolve contract")
            contract = app.contract_details
            
            app.data = []
            app.data_ready.clear()
            app.reqHistoricalData(
                reqId=2,
                contract=contract,
                endDateTime='',
                durationStr='1800 S',
                barSizeSetting='30 mins',
                whatToShow='TRADES',
                useRTH=0,
                formatDate=1,
                keepUpToDate=True,
                chartOptions=[]
            )
            
            if not app.data_ready.wait(timeout=60):
                logging.warning("Timeout waiting for real-time data")
                raise TimeoutError("Real-time data request timed out")
            if app.error_occurred:
                raise Exception(app.error_msg)
            
            save_to_csv(app.data)
            
        except Exception as e:
            logging.error(f"Real-time Fetch Error: {e}")
        finally:
            app.disconnect()
            if api_thread.is_alive():
                api_thread.join(timeout=5)
            socket.setdefaulttimeout(None)
    else:
        logging.info("Market is closed, skipping real-time fetch.")

# Schedule real-time fetches at :00 and :30
schedule.every().hour.at(":00").do(fetch_data)
schedule.every().hour.at(":30").do(fetch_data)

# Main execution
if __name__ == "__main__":
    # Run initial fetch
    initial_fetch2()
    
    # Run scheduler
    # logging.info("Starting HSI futures real-time data fetcher...")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)