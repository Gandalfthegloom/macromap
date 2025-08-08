"""
MacroMap
A world trade data visualization project in Python.
"""
import pandas as pd
from raw_process import BASE_DIR
import os
import streamlit as st

TRADE_DATA = 'Data/bilateral_value_clean_23_withid.csv'  # Default data to be used
GDP_DATA = 'Data/world_bank_gdp/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_26433.csv'


def run_trade_dashboard(filename: str = TRADE_DATA, gdp: str = GDP_DATA) -> None:
    """Run the complete trade visualization dashboard.

    This function orchestrates the following steps:
    1. Load and process the trade data
    2. Build the trade network graph
    3. Perform network analysis
    4. Launch the integrated dashboard with multiple visualization options

    Preconditions:
        - filename is a valid CSV file
        - gdp is a valid CSV file
    """
    # Step 1: Load the (processed) trade data

    # Step 2: Build the trade network graph

    # Step 3: Perform network analysis

    # Step 4: Launch the dashboard
    

hs2_path = os.path.join(BASE_DIR, "parquet", "HS2")
hs2_data = pd.read_parquet(hs2_path)