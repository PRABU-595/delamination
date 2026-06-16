import sqlite3
import pandas as pd
from pathlib import Path

def explore_pridb(file_path):
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in {file_path.name}:")
    for table in tables:
        print(f" - {table[0]}")
    
    # Peek at 'AE_Data' or 'Hits' or similar if it exists
    for table_name in ['AE_Data', 'Hits', 'LocalData']:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
            print(f"\nPeek into {table_name}:")
            print(df)
        except Exception:
            pass
    
    conn.close()

if __name__ == "__main__":
    p = Path("data/raw/additional/F-MOC/extracted/ACOUSTIC/spec1.pridb")
    if p.exists():
        explore_pridb(p)
    else:
        print(f"File {p} does not exist yet.")
