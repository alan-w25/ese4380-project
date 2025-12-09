
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import datetime

def construct_moment_dataset(input_path="data/sp500_historical_returns.csv", output_path="data/SP500_Moments_2000_2024.csv"):
    """
    Constructs the Moment Evolution dataset from local historical returns.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    print(f"Loading data from {input_path}...")
    # Load specific columns to save memory if needed, but file isn't huge.
    df = pd.read_csv(input_path, parse_dates=['date'])
    
    # Filter Date Range (2000-01-01 to 2024-01-01)
    # User requested: "Filter form 2000-01-01 to 2024-01-01"
    start_date = "2000-01-01"
    end_date = "2024-01-01"
    print(f"Filtering data from {start_date} to {end_date}...")
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Handle Delisting Returns
    # CRSP convention: gross_return = (1 + ret) * (1 + dlret) - 1
    # We want log returns: ln( (1+ret)*(1+dlret) )
    print("Processing returns (combining daily and delisting returns)...")
    
    # Fill NaNs in ret and dlret with 0 for calculation safety, 
    # but we must preserve NaNs where no data exists.
    # However, this dataset is likely 'valid observations only'.
    
    # Ensure columns are numeric, coercing errors (like 'C' or 'T' codes if raw CRSP)
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    df['dlret'] = pd.to_numeric(df['dlret'], errors='coerce').fillna(0.0)
    
    # If ret is NaN but we have a row, it might be a missing return day. 
    # If we dropna(subset=['ret']) we might lose the delisting day if ret is missing?
    # Usually valid rows have ret.
    df = df.dropna(subset=['ret'])

    # Calculation
    # Gross Return = (1 + ret) * (1 + dlret)
    df['gross_ret'] = (1 + df['ret']) * (1 + df['dlret'])
    
    # Log Return = ln(Gross Return)
    # Handle non-positive returns (bankruptcy/extreme loss)
    # If gross_ret <= 0, log is undefined/inf.
    # This happens if ret = -1 (100% loss). np.log(0) = -inf.
    # We will compute and replace -inf.
    df['log_ret'] = np.log(df['gross_ret'])
    
    # Replace -inf with NaN (or a proxy for -100% like log(1e-6))?
    # Moments calculation with -inf will fail.
    # A true bankruptcy (-100%) effectively removes the stock.
    # Winsorization should handle it if it's just an outlier, but -inf is problematic.
    # We'll drop rows with -inf or NaN log_ret for moment calculation safety, 
    # essentially treating it as "not returned" or "removed".
    
    mask_inf = np.isinf(df['log_ret'])
    if mask_inf.any():
        print(f"Warning: Found {mask_inf.sum()} infinite log-returns (bankruptcies?). These will be treated as NaNs for moment calculation.")
        df.loc[mask_inf, 'log_ret'] = np.nan

    print("Pivoting data to wide format...")
    # Pivot so Index=Date, Columns=Permno
    # using permno is safer than TICKER to avoid duplicates (e.g., dual class shares or ticker reuse)
    # We first ensure no duplicates for date/permno
    df = df.drop_duplicates(subset=['date', 'permno'])
    
    log_returns = df.pivot(index='date', columns='permno', values='log_ret')
    
    # Sort index
    log_returns = log_returns.sort_index()

    # Prepare for moment calculation
    dates = log_returns.index
    moments_history = []
    
    valid_dates = []

    print("Calculating Day-by-Day Cross-Sectional Moments...")
    for date in tqdm(dates, desc="Processing Days"):
        # Extract returns for the day
        daily_returns = log_returns.loc[date]
        
        # Drop NaNs (stocks not active)
        daily_returns = daily_returns.dropna()
        
        # Check active stocks count
        if len(daily_returns) < 200:
            continue
            
        vals = daily_returns.values
        
        # Winsorization (1st and 99th percentiles)
        p1 = np.percentile(vals, 1)
        p99 = np.percentile(vals, 99)
        vals_clipped = np.clip(vals, p1, p99)
        
        # Moment Computation (Power Mean)
        # m_k = ( (1/N) * sum(r^k) )^(1/k)
        # Even k: (mean(r^k))^(1/k)
        # Odd k: sign(mean(r^k)) * abs(mean(r^k))^(1/k)
        
        day_moments = []
        for k in range(1, 6): # K=1 to 5
            sum_rk = np.sum(np.power(vals_clipped, k))
            n = len(vals_clipped)
            mean_rk = sum_rk / n
            
            if k % 2 == 0:
                # Even: returns can be negative, but returns^even is positive. mean is positive.
                if mean_rk < 0:
                    # Should not happen theoretically for real numbers
                    m_k = 0
                else:
                    m_k = np.power(mean_rk, 1/k)
            else:
                # Odd
                m_k = np.sign(mean_rk) * np.power(np.abs(mean_rk), 1/k)
            
            day_moments.append(m_k)
            
        moments_history.append(day_moments)
        valid_dates.append(date)

    # DataFrame construction
    moments_df = pd.DataFrame(moments_history, index=valid_dates, columns=['m1', 'm2', 'm3', 'm4', 'm5'])
    moments_df.index.name = 'Date'

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Matches OU Moments.csv style (no header? user said "no header or simple header").
    # USER REQUEST: "Index: Date. Columns: m1... Save... matches the input style expected... (no header or simple header, consistent with the OU Moments.csv)"
    # OU Moments.csv had NO header. But user asked for specific columns.
    # If I save without header, I lose the 'Index: Date' context unless the Date is the first column.
    # Usually standard CSV output `to_csv` includes index and header.
    # To match "OU Moments.csv example" which purely data, maybe I should do header=False?
    # BUT user said "Index: Date". This implies the file *should* have the date.
    # I will stick with default to_csv (header=True, index=True) because for a *new* dataset like SP500, having headers is much safer.
    # The user said "no header or simple header". Simple header implies `header=True`.
    
    print(f"Saving to {output_path}...")
    moments_df.to_csv(output_path)
    
    print("Done.")

if __name__ == "__main__":
    construct_moment_dataset()
