"""
Graph Builder
Python script containing functions to convert parquet data into a geospatial graph in pydeck and display in Streamlit.
"""
# Standard library imports
import os

# Third-party imports
import pandas as pd        # Data manipulation
import networkx as nx      # Graph algorithms
import pydeck as pdk       # Geospatial visualization
import streamlit as st     # Web app framework

# Project-specific constant: base directory for data files
from raw_process import BASE_DIR

# Significance threshold for Serrano's disparity filter
DISP_FILTER_ALPHA_SIG = 1e-9


def graph_filter(trade_data: pd.DataFrame, type: str, key: int = -1) -> pd.DataFrame:
    """
    Filter the raw trade DataFrame to a single HS level or return all goods.

    Args:
        trade_data: Full DataFrame (all goods) loaded from Parquet.
        type: 'all_goods', 'hs2', or 'hs4'.
        key: HS code (e.g., 38) to filter when type != 'all_goods'.

    Returns:
        A filtered DataFrame containing only the selected commodity level.
    """
    print(f"[graph_filter] Starting filter: type={type}, key={key}, input_rows={len(trade_data)}")
    try:
        df = trade_data.copy()
        # No filtering for all goods
        if type == "all_goods":
            print(f"[graph_filter] Returning all {len(df)} rows for 'all_goods'.")
            return df

        # Filter by HS code column (e.g., 'hs2_code' or 'hs4_code')
        code_col = f"{type}_code"
        if code_col not in df.columns:
            raise ValueError(f"Column '{code_col}' not found in trade_data")
        df = df[df[code_col] == str(key)].reset_index(drop=True)
        print(f"[graph_filter] Filtered to {len(df)} rows where {code_col} == {key}")
        return df

    except Exception as e:
        print(f"[graph_filter] ERROR: {e}")
        raise


def edge_filter(trade_data: pd.DataFrame, type: str, alpha_sig: float = DISP_FILTER_ALPHA_SIG,) -> pd.DataFrame:
    """
    Reduce edges using Serrano's disparity filter and then add a global maximum spanning tree to preserve connectivity.

    Args:
        trade_data: DataFrame with columns ['exporter_id','importer_id',value].
        alpha_sig: Significance level for the disparity test.

    Returns:
        A reduced DataFrame of edges.
    """
    print(f"[edge_filter] Starting disparity + MST: input_rows={len(trade_data)}, alpha_sig={alpha_sig}")
    try:
        df = trade_data.copy()
        # Determine 'value' column
        if 'total_value' in df.columns:
            value_col = 'total_value'
        elif 'value_hs2' in df.columns:
            value_col = 'value_hs2'
        elif 'value_hs4' in df.columns:
            value_col = 'value_hs4'
        else:
            raise ValueError("No 'value' column found in trade_data")
        print(f"[edge_filter] Using '{value_col}' as weight column.")

        # Compute out-strength and degree per exporter
        s = df.groupby('exporter_id')[value_col].transform('sum')
        k = df.groupby('exporter_id')[value_col].transform('count')
        print(f"[edge_filter] Computed strength and degree for {df['exporter_id'].nunique()} exporters.")

        # Disparity filter p_ij and alpha_ij
        p = df[value_col] / s
        alpha = 1 - (1 - p) ** (k - 1)
        df_filtered = df[(k <= 1) | (alpha < alpha_sig)].copy()
        print(f"[edge_filter] After disparity filter: {len(df_filtered)} edges retained.")

        # Build undirected graph for MST
        G = nx.Graph()
        for _, row in df.iterrows():
            u, v, w = row['exporter_id'], row['importer_id'], row[value_col]
            G.add_edge(u, v, weight=G[u][v]['weight'] + w if G.has_edge(u, v) else w)
        print(f"[edge_filter] Built undirected graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

        # Maximum spanning tree
        T = nx.maximum_spanning_tree(G, weight='weight')
        print(f"[edge_filter] MST extracted: {T.number_of_edges()} edges.")

        # Add MST edges back\ n        
        mst_list = []
        for u, v in T.edges():
            for src, dst in [(u, v), (v, u)]:
                subset = df[(df['exporter_id'] == src) & (df['importer_id'] == dst)]
                if not subset.empty:
                    mst_list.append(subset)
        if mst_list:
            mst_df = pd.concat(mst_list, ignore_index=True)
            df_filtered = pd.concat([df_filtered, mst_df], ignore_index=True).drop_duplicates()
            print(f"[edge_filter] After MST merge: {len(df_filtered)} edges.")

        df_filtered = df_filtered.reset_index(drop=True)
        
        if type != "all_goods":
            df_filtered = df_filtered.sort_values(f'value_{type}', ascending=False)[:100]
        st.write("filtered")
        st.write(df_filtered.head())
        return df_filtered
        
    except Exception as e:
        print(f"[edge_filter] ERROR: {e}")
        raise


def graph_builder(graph: pd.DataFrame, iso_convert: pd.DataFrame, type: str, key: int = -1) -> pdk.Deck:
    """
    Convert edge DataFrame into a PyDeck Deck object with ArcLayer and ScatterplotLayer.

    Args:
        graph: DataFrame after edge filtering, using ISO3 codes directly.

    Returns:
        A pdk.Deck instance for visualization.
    """
    print(f"[graph_builder] Building deck from {len(graph)} edges.")
    try:
        df = graph.copy()
        # Identify weight column
        if 'total_value' in df.columns:
            value_col = 'total_value'
        elif 'value_hs2' in df.columns:
            value_col = 'value_hs2'
        elif 'value_hs4' in df.columns:
            value_col = 'value_hs4'
        else:
            raise ValueError("No 'value' column found in graph DataFrame")

        # Load country centroids (ISO3 alpha-2 codes)
        coords_fp = os.path.join(BASE_DIR, "countries.csv")
        country_coords = pd.read_csv(coords_fp)
        print(f"[graph_builder] Loaded {len(country_coords)} country centroids.")
        st.write(country_coords.head())
        

        # Add alpha-2 conversion of country codes to dataframe
        df['exporter_id'] = df['exporter_id'].str.upper().str.strip()
        df['importer_id'] = df['importer_id'].str.upper().str.strip()
        st.write(df.head())
        st.write(iso_convert.head())
        iso_convert_alpha2 = iso_convert[['alpha2', 'alpha3']]
        df = df.merge(iso_convert_alpha2, how='left', left_on='exporter_id', right_on='alpha3')
        df = df.rename(columns={"alpha2": "exporter_alpha2"})
        df = df.merge(iso_convert_alpha2, how='left', left_on='importer_id', right_on='alpha3')
        df = df.rename(columns={"alpha2": "importer_alpha2"})

        # Merge coords
        src = country_coords.rename(columns={'ISO': 'exporter_iso3', 'latitude': 'source_lat', 'longitude': 'source_lon'})
        tgt = country_coords.rename(columns={'ISO': 'importer_iso3', 'latitude': 'target_lat', 'longitude': 'target_lon'})
        df = df.merge(src[['exporter_iso3', 'source_lat', 'source_lon']], left_on='exporter_alpha2', right_on='exporter_iso3', how='left')
        df = df.merge(tgt[['importer_iso3', 'target_lat', 'target_lon']], left_on='importer_alpha2', right_on='importer_iso3', how='left')

        # Drop missing geo
        before = len(df)
        df = df.dropna(subset=['source_lat', 'source_lon', 'target_lat', 'target_lon'])
        print(f"[graph_builder] Dropped {before - len(df)} edges missing geo info.")

        # Compute arc thickness
        max_w = df[value_col].max()
        df['weight'] = 1 + (df[value_col] / max_w) ** 0.3 * 10

        # ArcLayer definition
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=df,
            get_source_position=["source_lon", "source_lat"],
            get_target_position=["target_lon", "target_lat"],
            get_source_color=[255, 0, 0, 180],
            get_target_color=[0, 0, 255, 180],
            get_width="weight",
            pickable=True,
            auto_highlight=True,
        )

        # Node scatter by export volume
        node_df = df.groupby(['exporter_iso3', 'source_lat', 'source_lon'])[value_col].sum().reset_index()
        max_e = node_df[value_col].max()
        node_df['radius'] = 1000000 * (node_df[value_col] / max_e) ** 0.5
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=node_df,
            get_position=["source_lon", "source_lat"],
            get_radius="radius",
            get_fill_color=[255, 255, 0, 180],
            pickable=True
        )

        # Combine into Deck
        view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)
        deck = pdk.Deck(layers=[arc_layer, scatter_layer], initial_view_state=view_state)
        print(f"[graph_builder] Deck ready: {len(df)} arcs, {len(node_df)} nodes.")
        return deck
    except Exception as e:
        print(f"[graph_builder] ERROR: {e}")
        raise

# Streamlit App
st.title("Global Trade Network Visualization")
st.write("Load trade data, choose HS level, and explore the network of trade flows scaled by volume.")

# Sidebar: HS level
selected_year = st.sidebar.selectbox("Select year:", ["2023"], index=0)
trade_type = st.sidebar.selectbox("Select HS level:", ["all_goods", "hs2", "hs4"], index=1)
if trade_type != "all_goods":
    if trade_type == "hs2":
        product_key = st.sidebar.number_input("HS code key:", value=38, min_value=0,
                                            help="Numeric code for HS2 or HS4 filter")
    if trade_type == "hs4":
        product_key = st.sidebar.number_input("HS code key:", value=3801, min_value=100,
                                            help="Numeric code for HS2 or HS4 filter")
else:
    product_key = -1
    
# Load parquet data
graph_fp = os.path.join(
    BASE_DIR,
    "parquet",
    trade_type,                     # one of "all_goods","hs2","hs4"
    f"{selected_year}_{trade_type}.parquet"
)
trade_df = pd.read_parquet(graph_fp)
print(f"[main] Loaded trade data: {len(trade_df)} rows")

# alpha2 to alpha3 conversion
alpha_fp = os.path.join(BASE_DIR, "alpha2_alpha3.csv")
alpha_conv = pd.read_csv(alpha_fp)

# Apply filters and reduction
filtered = graph_filter(trade_df, type=trade_type, key=product_key)
sparse = edge_filter(filtered, type=trade_type)

# Build and render Deck
deck = graph_builder(sparse, alpha_conv, type=trade_type, key=product_key) 
st.pydeck_chart(deck)
