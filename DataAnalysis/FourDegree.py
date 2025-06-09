import pandas as pd
from datetime import time
import matplotlib.pyplot as plt
from matplotlib.table import Table
import os
from pathlib import Path

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

# Save all virtual tables aligned by price to a CSV file and generate monthly images and CSVs
def display_all_virtual_tables(virtual_tables, dates, output_file):
    # Set Matplotlib style to default for light background
    plt.style.use('default')
    
    # Extract ticker from output file name
    ticker = Path(output_file).stem.split('_')[0]
    
    # Create directory for ticker if it doesn't exist
    ticker_dir = Path(ticker)
    ticker_dir.mkdir(exist_ok=True)
    
    # Sort dates from oldest to latest
    sorted_date_pairs = sorted(zip(dates, virtual_tables), key=lambda x: x[0])
    sorted_dates, sorted_tables = zip(*sorted_date_pairs)
    
    # Find the complete set of prices across all tables
    all_prices = set()
    for table in sorted_tables:
        all_prices.update(table['Price'])
    all_prices = sorted(all_prices, reverse=False)  # Sort prices low to high for Y-axis
    
    # Create a combined DataFrame
    combined_df = pd.DataFrame({'Price': all_prices})
    
    # Collect all date columns as a list of DataFrames
    date_columns = []
    max_times_lengths = {}
    for date, table in zip(sorted_dates, sorted_tables):
        # Convert table to a dictionary for quick lookup
        price_to_times = dict(zip(table['Price'], table['Times']))
        # Create a Series for this date
        times_series = pd.Series([price_to_times.get(price, '') for price in all_prices], name=str(date))
        date_columns.append(times_series)
        # Calculate max length for padding
        max_times_lengths[str(date)] = max(len(str(times)) for times in times_series)
    
    # Concatenate all date columns at once
    date_df = pd.concat(date_columns, axis=1)
    combined_df = pd.concat([combined_df, date_df], axis=1)
    
    # Format each Times column to be left-aligned with spaces padded to the right
    for date in sorted_dates:
        combined_df[str(date)] = combined_df[str(date)].apply(
            lambda x: f"{str(x):<{max_times_lengths[str(date)]}}"
        )
    
    # Save to main CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined virtual tables saved to {output_file}")
    
    # Group dates by month
    monthly_groups = {}
    for date in sorted_dates:
        month_key = date.strftime('%Y-%m')
        if month_key not in monthly_groups:
            monthly_groups[month_key] = []
        monthly_groups[month_key].append(str(date))
    
    # Generate an image and CSV for each month
    for month_key, date_columns in monthly_groups.items():
        # Sort date columns chronologically for X-axis (first to last day of the month)
        date_columns = sorted(date_columns)
        # Create a DataFrame for this month with Price as Y-axis
        month_df = combined_df[['Price'] + date_columns]
        
        # Filter rows where at least one date column has non-empty content (excluding whitespace)
        month_df = month_df[month_df[date_columns].apply(lambda x: x.str.strip()).ne('').any(axis=1)]
        
        # Skip if month_df is empty after filtering
        if month_df.empty:
            print(f"No data for {month_key}, skipping CSV and image generation.")
            continue
        
        # Save monthly CSV
        output_csv = ticker_dir / f"{ticker}_{month_key}.csv"
        month_df.to_csv(output_csv, index=False)
        print(f"Monthly virtual table CSV saved to {output_csv}")
        
        # Generate image with Price on Y-axis (rows) and dates on X-axis (columns)
        output_image = ticker_dir / f"{ticker}_{month_key}.png"
        max_content_length = max(max_times_lengths.get(date, 0) for date in date_columns)
        col_width = max(1, max_content_length * 0.1)  # Increased minimum width to 0.5, adjusted scaling to 0.02
        col_width = max(0.5, max_content_length * 0.08)  # Increased minimum width to 0.5, adjusted scaling to 0.02

        # col_width = 1 / len(date_columns)
        fig_width=max(len(date_columns) * 1.5, 10)
        fig_width = max(3, len(date_columns) * col_width + 2.0)  # Minimum width of 3, added 2.0 units padding
        print(f"max_content_length={max_content_length}, col_width={col_width}, fig_width={fig_width}")
        fig, ax = plt.subplots(figsize=(fig_width, len(month_df) * 0.3 + 2))
        
        # Create table with Price as row headers (Y-axis) and dates as column headers (X-axis)
        table = Table(ax, bbox=[0, 0, 1, 1])
        
        # Add column headers (dates for X-axis) with left alignment
        table.add_cell(0, 0, col_width * 0.2, 1, text='Price', loc='center')  # Price header
        for col_idx, col_name in enumerate(date_columns, start=1):
            table.add_cell(0, col_idx, col_width, 1, text=col_name, loc='center')
        
        # Add data rows (Price for Y-axis) with left alignment
        for row_idx, row in month_df.iterrows():
            table.add_cell(row_idx + 1, 0, col_width * 0.2, 1, text=str(row['Price']), loc='center')  # Price value
            for col_idx, value in enumerate(row[1:], start=1):
                table.add_cell(row_idx + 1, col_idx, col_width, 1, text=value, loc='left')
        
        # Styling
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Set table properties for visibility
        for (row, col), cell in table.get_celld().items():
            cell.set_text_props(color='black')
            cell.set_edgecolor('black')
            cell.set_facecolor('white')
        
        ax.add_table(table)
        plt.title(f"{ticker} Virtual Table - {month_key}", fontsize=10, pad=15)
        plt.axis('off')  # Hide axes
        plt.savefig(output_image, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Monthly virtual table image saved to {output_image}")

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with your CSV file path
    file_path = "../Data/HSI-futures-30min.csv"
    output_file = 'HSI_virtual_tables.csv'
    
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
        
        # Save all virtual tables to CSV and generate monthly images and CSVs
        if virtual_tables:
            display_all_virtual_tables(virtual_tables, dates, output_file)
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")