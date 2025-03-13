import pandas as pd
import os

def clean_and_filter_csv(input_csv, output_csv):
    """
    Reads a CSV file, reformats it to standard CSV formatting, and keeps only the 
    "Image name" and "Curl Diameter" columns. Any rows missing these values are dropped.
    The cleaned DataFrame is written to a new CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Remove leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Define the columns to keep
    columns_to_keep = ["Image name", "Twist Frequency", "Length"]
    
    # Verify that the required columns exist in the CSV
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}. Found columns: {df.columns.tolist()}")
    
    # Filter the DataFrame to only the desired columns and drop rows with missing values
    df_filtered = df[columns_to_keep].dropna()
    
    # Ensure the "Curl Diameter" column is of type float for consistency
    df_filtered["Twist Frequency"] = df_filtered["Twist Frequency"].astype(float)
    
    # Write the filtered DataFrame to a new CSV file
    df_filtered.to_csv(output_csv, index=False)
    print(f"Filtered CSV saved to: {output_csv}")

if __name__ == "__main__":
    # Update these file paths as needed
    input_csv = r"scripts/imagehairtype1.csv"
    output_csv = r"scripts/twistlength.csv"
    
    if os.path.exists(input_csv):
        clean_and_filter_csv(input_csv, output_csv)
    else:
        print(f"Input CSV file not found: {input_csv}")
