"""
Generate Synthetic Sales Data
Creates realistic sales data for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse

def generate_sales_data(n_days=1065, start_date='2023-01-01', seed=42):
    """
    Generate synthetic sales data
    
    Args:
        n_days: Number of days of data to generate
        start_date: Start date for the dataset
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with sales data
    """
    np.random.seed(seed)
    
    # Generate dates
    start = pd.to_datetime(start_date)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    
    # Generate features
    data = []
    
    for i, date in enumerate(dates):
        # Time-based features
        day_of_week = date.dayofweek
        month = date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Base sales with trend and seasonality
        base_sales = 100
        trend = i * 0.02  # Gradual upward trend
        
        # Seasonal patterns
        yearly_seasonality = 20 * np.sin(2 * np.pi * i / 365.25)  # Yearly cycle
        weekly_seasonality = 15 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly cycle
        
        # Random components
        advertising_spend = np.random.uniform(1000, 5000)
        promotions = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% promotion days
        
        # Sales calculation with various effects
        sales = (
            base_sales +
            trend +
            yearly_seasonality +
            weekly_seasonality +
            advertising_spend * 0.015 +  # Advertising effect
            promotions * 25 +  # Promotion boost
            is_weekend * 20 +  # Weekend boost
            np.random.gamma(2, 8)  # Random noise
        )
        
        # Ensure sales is positive
        sales = max(sales, 10)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'sales': round(sales, 2),
            'advertising_spend': round(advertising_spend, 2),
            'promotions': promotions,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend
        })
    
    df = pd.DataFrame(data)
    
    print(f"✓ Generated {len(df)} days of sales data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Sales range: ${df['sales'].min():.2f} to ${df['sales'].max():.2f}")
    print(f"  Sales mean: ${df['sales'].mean():.2f}")
    
    return df


def save_data(df, output_path):
    """Save data to CSV"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Data saved to: {output_path}")
    
    # Print file size
    file_size = Path(output_path).stat().st_size
    print(f"  File size: {file_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sales data")
    parser.add_argument("--n-days", type=int, default=1065,
                        help="Number of days of data (default: 1065 ~3 years)")
    parser.add_argument("--start-date", default="2023-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/raw/sales_data.csv",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Synthetic Sales Data")
    print("=" * 60)
    
    # Generate data
    df = generate_sales_data(
        n_days=args.n_days,
        start_date=args.start_date,
        seed=args.seed
    )
    
    # Save data
    save_data(df, args.output)
    
    print("=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Track with DVC: dvc add data/raw/sales_data.csv")
    print("  2. Commit: git add data/raw/sales_data.csv.dvc")
    print("  3. Preprocess: python src/preprocess.py")


if __name__ == "__main__":
    main()
