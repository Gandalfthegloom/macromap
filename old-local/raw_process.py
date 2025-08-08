"""
A module for processing raw bilateral trade data from CEPII's Base pour l'Analyse du Commerce International (BACI).
Default file format is CSV.
Provides functions to convert HS6 to HS4, and HS4 to HS2 level aggregates.
"""

import os
import pandas as pd

# Check filepath
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory where this script lives

# Build absolute paths relative to the script
harmonized_path = os.path.join(BASE_DIR, "harmonized.csv")
data2023_path = os.path.join(BASE_DIR, "..", "Data", "trade_i_baci_a_12_2023.csv")

# Load CSV files
def load_data():
    print("1/5 ‒ Loading CSV files...")
    harmonized = pd.read_csv(harmonized_path).astype({
        'hs2_code': str,
        'hs4_code': str
    })
    data_2023 = pd.read_csv(data2023_path)
    print("   • Loaded harmonized.csv and trade_i_baci_a_12_2023.csv")
    return data_2023, harmonized


def convert_hs6_to_hs4(
    data: pd.DataFrame,
    harmonized_index: pd.DataFrame,
    hs_code_col: str = "hs_code",
    value_col: str = "value",
    quantity_col: str = "quantity",
    group_cols: list = None
) -> pd.DataFrame:
    print("2/5 ‒ Converting HS6 → HS4 aggregates...")
    df = data.copy()
    df[hs_code_col] = df[hs_code_col].astype(str)

    # derive HS4
    df['hs4_code'] = df[hs_code_col].str[:4]
    mask5 = df[hs_code_col].str.len() == 5
    df.loc[mask5, 'hs4_code'] = df.loc[mask5, hs_code_col].str[:3]

    # derive HS2
    df['hs2_code'] = df[hs_code_col].str[:2]
    mask3 = df[hs_code_col].str.len() == 3
    df.loc[mask3, 'hs2_code'] = df.loc[mask3, hs_code_col].str[:1]

    # default grouping
    if group_cols is None:
        group_cols = [
            'year',
            'exporter_name',
            'exporter_id',
            'importer_name',
            'importer_id',
            'hs2_code',
            'hs4_code'
        ]

    # aggregate
    df_val = (
        df
        .groupby(group_cols)[value_col]
        .sum()
        .reset_index(name='value_hs4')
    )
    df_qty = (
        df
        .groupby(group_cols)[quantity_col]
        .sum()
        .reset_index(name='quantity_hs4')
    )
    df_hs4 = df_val.merge(df_qty, on=group_cols)

    # merge in headings
    df_hs4 = df_hs4.merge(
        harmonized_index,
        on=['hs2_code', 'hs4_code'],
        how='left'
    )

    # reorder
    desired_order = [
        'year',
        'exporter_name',
        'exporter_id',
        'importer_name',
        'importer_id',
        'hs2_code',
        'hs2_heading',
        'hs4_code',
        'hs4_heading',
        'value_hs4',
        'quantity_hs4'
    ]
    print(f"   • Generated HS4 DataFrame with {len(df_hs4)} rows")
    return df_hs4[desired_order]


def convert_hs4_to_hs2(
    data_hs4: pd.DataFrame,
    harmonized_index: pd.DataFrame,
    value_hs4_col: str = 'value_hs4',
    quantity_hs4_col: str = 'quantity_hs4',
    group_cols: list = None
) -> pd.DataFrame:
    print("3/5 ‒ Converting HS4 → HS2 aggregates...")
    df = data_hs4.copy()

    if group_cols is None:
        group_cols = [
            'year',
            'exporter_name',
            'exporter_id',
            'importer_name',
            'importer_id',
            'hs2_code'
        ]

    # sum HS4 values into HS2 totals
    df_val = (
        df
        .groupby(group_cols)[value_hs4_col]
        .sum()
        .reset_index(name='value_hs2')
    )
    df_qty = (
        df
        .groupby(group_cols)[quantity_hs4_col]
        .sum()
        .reset_index(name='quantity_hs2')
    )
    df_hs2 = df_val.merge(df_qty, on=group_cols)

    # get unique HS2 headings
    hs2_headings = (
        harmonized_index[['hs2_code', 'hs2_heading']]
        .drop_duplicates()
    )

    # merge headings
    df_hs2 = df_hs2.merge(
        hs2_headings,
        on='hs2_code',
        how='left'
    )

    # reorder
    desired_order = [
        'year',
        'exporter_name',
        'exporter_id',
        'importer_name',
        'importer_id',
        'hs2_code',
        'hs2_heading',
        'value_hs2',
        'quantity_hs2'
    ]
    print(f"   • Generated HS2 DataFrame with {len(df_hs2)} rows")
    return df_hs2[desired_order]

def convert_hs2_to_all_goods(
    data_hs2: pd.DataFrame,
    value_hs2_col: str = 'value_hs2',
    quantity_hs2_col: str = 'quantity_hs2',
    group_cols: list = None
) -> pd.DataFrame:
    """
    Aggregates HS2‐level bilateral trade into a single all_goods row per exporter–importer–year.

    Parameters
    ----------
    data_hs2 : pd.DataFrame
        Output from convert_hs4_to_hs2(), with columns including
        ['year','exporter_name','exporter_id','importer_name','importer_id','value_hs2','quantity_hs2',…].
    value_hs2_col : str
        Name of the HS2 value column (default 'value_hs2').
    quantity_hs2_col : str
        Name of the HS2 quantity column (default 'quantity_hs2').
    group_cols : list of str, optional
        Columns to group by. By default:
          ['year','exporter_name','exporter_id','importer_name','importer_id'].

    Returns
    -------
    pd.DataFrame
        Columns: ['year','exporter_name','exporter_id','importer_name','importer_id',
                  'total_value','total_quantity']
    """
    print("3.5/5 ‒ Converting HS2 → all_goods aggregates...")
    df = data_hs2.copy()

    if group_cols is None:
        group_cols = [
            'year',
            'exporter_name',
            'exporter_id',
            'importer_name',
            'importer_id'
        ]

    # sum across all HS2 codes
    df_val = (
        df
        .groupby(group_cols)[value_hs2_col]
        .sum()
        .reset_index(name='total_value')
    )
    df_qty = (
        df
        .groupby(group_cols)[quantity_hs2_col]
        .sum()
        .reset_index(name='total_quantity')
    )

    df_all = df_val.merge(df_qty, on=group_cols)
    print(f"   • Generated all_goods DataFrame with {len(df_all)} rows")

    return df_all[group_cols + ['total_value', 'total_quantity']]



if __name__ == "__main__":
    # 1) Load
    DATA_2023, HARMONIZED_SYSTEM_INDEX = load_data()

    # 2) Convert to HS4
    data_2023_hs4 = convert_hs6_to_hs4(DATA_2023, HARMONIZED_SYSTEM_INDEX)

    # 3) Convert to HS2
    data_2023_hs2 = convert_hs4_to_hs2(data_2023_hs4, HARMONIZED_SYSTEM_INDEX)
    
    # 4) Convert HS2 → all_goods
    data_2023_all = convert_hs2_to_all_goods(data_2023_hs2)

    # 5) Prepare output directories
    print("4/5 ‒ Preparing output directories...")
    hs4_dir = os.path.join(BASE_DIR, "parquet", "HS4")
    hs2_dir = os.path.join(BASE_DIR, "parquet", "HS2")
    all_dir = os.path.join(BASE_DIR, "parquet", "ALL_GOODS")
    os.makedirs(hs4_dir, exist_ok=True)
    os.makedirs(hs2_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)
    print(f"   • Ensured directories:\n     - {hs4_dir}\n     - {hs2_dir}\n     - {all_dir}")

    # 5) Write to Parquet
    print("5/5 ‒ Writing Parquet files...")
    hs4_path = os.path.join(hs4_dir, "2023_hs4.parquet")
    hs2_path = os.path.join(hs2_dir, "2023_hs2.parquet")
    all_path = os.path.join(all_dir, "2023_all_goods.parquet")
    

    print(f"   • Writing HS4 data → {hs4_path}")
    data_2023_hs4.to_parquet(
        hs4_path,
        engine="fastparquet",
        compression="snappy",
        index=False
    )

    print(f"   • Writing HS2 data → {hs2_path}")
    data_2023_hs2.to_parquet(
        hs2_path,
        engine="fastparquet",
        compression="snappy",
        index=False
    )
    print(f"   • Writing all_goods data → {all_path}")
    data_2023_all.to_parquet(
        all_path,
        engine="fastparquet",
        compression="snappy",
        index=False
    )

    print("✅ All done!")
