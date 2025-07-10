import yfinance as yf
import pandas as pd
import pdb

def get_stock_splits(symbols):
    try:
        split_data_list = []
        for ticker in symbols:
        # Create a Ticker object
            stock = yf.Ticker(ticker)
        
        # Get historical stock splits
            splits = stock.splits
            if not splits.empty:
                splits_df = splits.reset_index()
                splits_df.columns = ['Date', 'SplitRatio']
                splits_df['ticker'] = ticker
                split_data_list.append(splits_df[['ticker', 'Date', 'SplitRatio']])
            else:
                print(f"No stock splits found for {ticker}")
                continue
            
        # Create a DataFrame with split ratios
        if split_data_list:
            all_splits = pd.concat(split_data_list, ignore_index=True)
            all_splits = all_splits.sort_values(['ticker', 'Date'])# Sort by date
        
        return all_splits
    
    except Exception as e:
        print(f"Error retrieving stock splits for {ticker}: {str(e)}")
        return None

def main():
    df = pd.read_csv('stock_list-2.csv')
    symbols = df['Symbol'].dropna().unique()
    
    # Get stock splits
    splits_df = get_stock_splits(symbols)
    
    #pdb.set_trace()

    if splits_df is not None:
        print(splits_df)
        # Optionally save to CSV
        splits_df.to_csv(f"all_stock_splits.csv", index=False)
        print(f"\nData saved to all_stock_splits.csv")

if __name__ == "__main__":
    main()
    