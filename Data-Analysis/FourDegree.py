import pandas as pd
from datetime import time

# Load CSV file into DataFrame and add time column
def load_and_split_csv(file_path):
    # Read CSV with timestamp parsing
    df = pd.read_csv(file_path, parse_dates=['Timestamp'], date_format='%Y-%m-%d %H:%M:%S')
    
    # Ensure columns are in correct format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter for trading hours (09:00:00 to 16:30:00)
    df = df[(df['Timestamp'].dt.time >= time(9, 0)) & (df['Timestamp'].dt.time <= time(16, 30))]
    
    # Add new time column in %H%M format
    df['Time'] = df['Timestamp'].dt.strftime('%H%M')
    
    # Group by date and create separate DataFrames
    daily_dfs = {date: group for date, group in df.groupby(df['Timestamp'].dt.date)}
    
    return daily_dfs

# Create virtual table with price ranges and time tags
def fourDegree(df, date):
    # Find min and max prices from High and Low columns
    min_price = df['Low'].min()
    max_price = df['High'].max()
    
    # Round min_price down and max_price up to nearest 10
    min_price = int(min_price // 10 * 10)
    max_price = int((max_price // 10 + 1) * 10)
    
    # Create price range for virtual table
    price_range = range(min_price, max_price + 10, 10)
    
    # Initialize virtual table as a dictionary with prices as keys and empty lists for time tags
    virtual_table = {price: [] for price in price_range}
    
    # Scan each row of the input DataFrame
    for _, row in df.iterrows():
        low = row['Low']
        high = row['High']
        time_tag = row['Time']
        
        # Round low down and high up to nearest 10
        low_rounded = int(low // 10 * 10)
        high_rounded = int((high // 10 + 1) * 10)
        
        # Add time tag to all price levels within the rounded Low-High range
        for price in range(low_rounded, high_rounded + 10, 10):
            if price in virtual_table:
                virtual_table[price].append(time_tag)
    
    # Convert virtual table to DataFrame with date as table name
    virtual_df = pd.DataFrame({
        'Price': list(virtual_table.keys()),
        'Times': [', '.join(tags) if tags else '' for tags in virtual_table.values()]
    }, index=[str(date)] * len(virtual_table))
    
    # Display the virtual table
    display_virtual_table(virtual_df, date)
    
    return virtual_df

# Display a single virtual table in a formatted way
def display_virtual_table(virtual_df, table_name):
    # Sort by Price in descending order
    sorted_df = virtual_df.sort_values(by='Price', ascending=False)
    
    # Determine the maximum length of the Times column for padding
    max_times_length = max(len(str(times)) for times in sorted_df['Times'])
    
    # Format Times column to be left-aligned with spaces padded to the right
    sorted_df['Times'] = sorted_df['Times'].apply(lambda x: f"{str(x):<{max_times_length}}")
    
    print(f"\nVirtual Table: {table_name}")
    print(sorted_df[['Price', 'Times']].to_string())

# Save all virtual tables aligned by price to a CSV file
def display_all_virtual_tables(virtual_tables, dates):
    # Sort dates from oldest to latest
    sorted_date_pairs = sorted(zip(dates, virtual_tables), key=lambda x: x[0])
    sorted_dates, sorted_tables = zip(*sorted_date_pairs)
    
    # Find the complete set of prices across all tables
    all_prices = set()
    for table in sorted_tables:
        all_prices.update(table['Price'])
    all_prices = sorted(all_prices, reverse=True)  # Sort prices high to low
    
    # Create a combined DataFrame
    combined_df = pd.DataFrame({'Price': all_prices})
    
    # Add Times columns for each date
    max_times_lengths = {}
    for date, table in zip(sorted_dates, sorted_tables):
        # Convert table to a dictionary for quick lookup
        price_to_times = dict(zip(table['Price'], table['Times']))
        # Add Times for this date, defaulting to empty string if price not present
        combined_df[str(date)] = [price_to_times.get(price, '') for price in all_prices]
        # Calculate max length for padding
        max_times_lengths[str(date)] = max(len(str(times)) for times in combined_df[str(date)])
    
    # Format each Times column to be left-aligned with spaces padded to the right
    for date in sorted_dates:
        combined_df[str(date)] = combined_df[str(date)].apply(
            lambda x: f"{str(x):<{max_times_lengths[str(date)]}}"
        )
    
    # Save to CSV file
    output_file = 'combined_virtual_tables.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined virtual tables saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with your CSV file path
    file_path = '../Data/HSI-futures.csv'
    file_path = "../Data/HSI-futures-30min.csv"
    
    try:
        daily_dfs = load_and_split_csv(file_path)
        
        # Collect all virtual tables
        virtual_tables = []
        dates = []
        
        # Process each day's DataFrame
        for date, day_df in daily_dfs.items():
            print(f"\nProcessing DataFrame for {date}:")
            print(f"Number of rows: {len(day_df)}")
            print(day_df.head())
            
            # Apply fourDegree function to the daily DataFrame with date
            virtual_table = fourDegree(day_df, date)
            virtual_tables.append(virtual_table)
            dates.append(date)
        
        # Save all virtual tables to CSV
        if virtual_tables:
            display_all_virtual_tables(virtual_tables, dates)
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")