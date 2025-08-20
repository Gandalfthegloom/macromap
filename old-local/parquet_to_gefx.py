"""
parquet_to_gefx
Converts bilateral trade data in parquet format into a GEFX multidigraph through networkx for network analysis.
"""


import pandas as pd
import networkx as nx
import os
from macromap.raw_process import BASE_DIR  # base path for data dirs

def hs2_parquet_to_mdgraph(pqt_address: str) -> nx.MultiDiGraph:
    """Convert HS2 parquet → MultiDiGraph (keyed by hs2_code)."""
    print(f"[HS2] Reading parquet file: {pqt_address}")
    try:
        df = pd.read_parquet(pqt_address).astype({'hs2_code': str})
        print(f"[HS2] DataFrame loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"[HS2] Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"[HS2] ERROR: Failed to read parquet - {e}")
        raise

    print("[HS2] Building MultiDiGraph...")
    try:
        graph = nx.from_pandas_edgelist(
            df,
            source="exporter_id",
            target="importer_id",
            edge_key="hs2_code",
            edge_attr=True,
            create_using=nx.MultiDiGraph
        )
        print(f"[HS2] Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    except Exception as e:
        print(f"[HS2] ERROR: Failed to build graph - {e}")
        raise

    return graph


def hs4_parquet_to_mdgraph(pqt_address: str) -> nx.MultiDiGraph:
    """Convert HS4 parquet → MultiDiGraph (keyed by hs4_code)."""
    print(f"[HS4] Reading parquet file: {pqt_address}")
    try:
        df = pd.read_parquet(pqt_address).astype({'hs4_code': str})
        print(f"[HS4] DataFrame loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"[HS4] Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"[HS4] ERROR: Failed to read parquet - {e}")
        raise

    print("[HS4] Building MultiDiGraph...")
    try:
        graph = nx.from_pandas_edgelist(
            df,
            source="exporter_id",
            target="importer_id",
            edge_key="hs4_code",
            edge_attr=True,
            create_using=nx.MultiDiGraph
        )
        print(f"[HS4] Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    except Exception as e:
        print(f"[HS4] ERROR: Failed to build graph - {e}")
        raise

    return graph


def all_goods_parquet_to_mdgraph(pqt_address: str) -> nx.MultiDiGraph:
    """Convert ALL_GOODS parquet → MultiDiGraph (aggregated over all goods)."""
    print(f"[ALL_GOODS] Reading parquet file: {pqt_address}")
    try:
        df = pd.read_parquet(pqt_address)
        print(f"[ALL_GOODS] DataFrame loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"[ALL_GOODS] Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"[ALL_GOODS] ERROR: Failed to read parquet - {e}")
        raise

    print("[ALL_GOODS] Building MultiDiGraph...")
    try:
        graph = nx.from_pandas_edgelist(
            df,
            source="exporter_id",
            target="importer_id",
            edge_attr=True,
            create_using=nx.MultiDiGraph
        )
        print(f"[ALL_GOODS] Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    except Exception as e:
        print(f"[ALL_GOODS] ERROR: Failed to build graph - {e}")
        raise

    return graph


def mdgraph_to_gexf(graph: nx.MultiDiGraph, title: str, type_: str) -> None:
    """Write graph to GEXF under BASE_DIR/gexf/{TYPE_}/{title}."""
    dirpath = os.path.join(BASE_DIR, "gexf", type_.upper())
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, title)

    print(f"[GEXF] Cleaning attributes and writing file: {filepath}")
    # Remove None values from edge attributes
    removed = 0
    for u, v, key, data in graph.edges(keys=True, data=True):
        for attr, val in list(data.items()):
            if val is None:
                del data[attr]
                removed += 1
    print(f"[GEXF] Removed {removed} None attributes from edges")

    try:
        nx.write_gexf(graph, filepath)
        print(f"[GEXF] Successfully wrote GEXF file: {filepath}")
    except Exception as e:
        print(f"[GEXF] ERROR: Failed to write GEXF - {e}")
        raise
    
def edge_filtering(graph: nx.MultiDiGraph, type: str, key) -> nx.MultiDiGraph:
    """_summary_

    Args:
        graph (nx.MultiDiGraph): _description_
        key: HS code

    Returns:
        nx.DiGraph: _description_
    """


if __name__ == "__main__":
    year = "2023"
    print(f"Starting GEXF export for year: {year}")

    # HS2
    print("\n--- Processing HS2 ---")
    hs2_fp = os.path.join(BASE_DIR, "parquet", "HS2", f"{year}_hs2.parquet")
    g2 = hs2_parquet_to_mdgraph(hs2_fp)
    mdgraph_to_gexf(g2, f"{year}_hs2.gexf", "hs2")
    print("--- HS2 processing complete ---\n")

    # HS4
    print("--- Processing HS4 ---")
    hs4_fp = os.path.join(BASE_DIR, "parquet", "HS4", f"{year}_hs4.parquet")
    g4 = hs4_parquet_to_mdgraph(hs4_fp)
    mdgraph_to_gexf(g4, f"{year}_hs4.gexf", "hs4")
    print("--- HS4 processing complete ---\n")

    # ALL_GOODS
    print("--- Processing ALL_GOODS ---")
    all_fp = os.path.join(BASE_DIR, "parquet", "all_goods", f"{year}_all_goods.parquet")
    ga = all_goods_parquet_to_mdgraph(all_fp)
    mdgraph_to_gexf(ga, f"{year}_all_goods.gexf", "all_goods")
    print("--- ALL_GOODS processing complete ---\n")

    print("GEXF export pipeline finished successfully.")

    
    ## Test
    # G = nx.read_gexf(os.path.join(BASE_DIR, "gexf", "HS2", "2023_hs2.gexf"))   # your full MultiDiGraph
    # desired_key = "38"  # e.g. the HS2 code you want to inspect

    # # 1) collect exactly the (u,v,key) triples you care about
    # edges = [(u, v, k) for u, v, k in G.edges(keys=True) if k == desired_key]

    # # 2) build the subgraph
    # H = G.edge_subgraph(edges)

    # # 3) draw it
    # pos = nx.spring_layout(H)
    # nx.draw(H, pos,
    #         with_labels=True,
    #         node_size=300,
    #         arrowsize=10)
    # plt.title(f"Only edges with hs2_code = {desired_key}")
    # plt.show()