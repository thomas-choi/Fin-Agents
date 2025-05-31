import pandas as pd
from pandas.tseries.offsets import Minute

def convert_to_30min(input_file, output_file, timestamp_column='Timestamp'):
    # Read the CSV file with explicit encoding
    try:
        df = pd.read_csv(input_file, parse_dates=[timestamp_column], encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except ValueError as e:
        print(f"Error: Issue with CSV parsing - {e}")
        return
    except Exception as e:
        print(f"Error: Failed to read CSV - {e}")
        return

    # Print column names for debugging
    print("Columns in CSV:", list(df.columns))

    # Verify required columns
    required_columns = [timestamp_column, 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Error: Missing columns in CSV: {missing}")
        return

    # Check for whitespace or case issues in timestamp column
    if timestamp_column not in df.columns:
        print(f"Error: '{timestamp_column}' not found. Available columns: {list(df.columns)}")
        print("Possible issue: Check for extra whitespace or case sensitivity in column names.")
        return

    # Function to assign 30-minute bins (0 or 30 minutes)
    def assign_30min_bin(timestamp):
        # Round up to the next 30-minute mark (00 or 30)
        if timestamp.minute <= 30:
            return timestamp.replace(minute=30, second=0, microsecond=0)
        else:
            # Use pandas Timedelta to handle month-end cases correctly
            return (timestamp + Minute(30)).replace(minute=0, second=0, microsecond=0)

    # Apply binning to create a new column for the 30-minute intervals
    try:
        df['bin_timestamp'] = df[timestamp_column].apply(assign_30min_bin)
    except Exception as e:
        print(f"Error during timestamp binning: {e}")
        return

    # Group by the binned timestamp and aggregate
    try:
        df_30min = df.groupby('bin_timestamp').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return

    # Reset index to make timestamp a column again
    df_30min.reset_index(inplace=True)

    # Rename bin_timestamp to Timestamp for consistency with input
    df_30min.rename(columns={'bin_timestamp': 'Timestamp'}, inplace=True)

    # Save to new CSV file
    try:
        df_30min.to_csv(output_file, index=False)
        print(f"30-minute interval data saved to {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "../Data/HSI-futures-1min.csv"  # Replace with your input file path
    output_file = "../Data/HSI-futures-30min.csv"  # Replace with desired output file path
	
    timestamp_column = "Timestamp"  # Confirmed as 'timestamp' based on your input
    convert_to_30min(input_file, output_file, timestamp_column)