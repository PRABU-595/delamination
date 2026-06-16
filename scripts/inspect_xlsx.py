import pandas as pd
import os
import glob

def inspect_xlsx():
    # Find the file
    search_path = "data/raw/NASA_CFRP/2. Composites/L2_S11_F/*.xlsx"
    files = glob.glob(search_path)
    
    if not files:
        print("No XLSX files found.")
        return

    target_file = files[0]
    print(f"Inspecting: {target_file}")
    
    try:
        df = pd.read_excel(target_file)
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")
        print("\nFirst 10 rows of 'Remarks':")
        print(df['Remarks'].head(10))
        
        # Check for sheet names if multiple
        xl = pd.ExcelFile(target_file)
        print("\nSheet names:", xl.sheet_names)
        
    except Exception as e:
        print(f"Error reading excel: {e}")

if __name__ == "__main__":
    inspect_xlsx()
