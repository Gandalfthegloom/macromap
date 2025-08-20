import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List, Tuple

import pandas as pd
import networkx as nx

# ============================================================
# Trade graph construction
# ============================================================

def _cols_for_type(type_: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Return (value_col, quantity_col, code_col, heading_col) for a given granularity.
    type_ in {"hs4", "hs2", "all_goods"}
    """
    t = type_.lower()
    if t == "hs4":
        return "value_hs4", "quantity_hs4", "hs4_code", None
    if t == "hs2":
        return "value_hs2", "quantity_hs2", "hs2_code", "hs2_heading"
    if t == "all_goods":
        return "total_value", "total_quantity", None, None
    raise ValueError("type must be one of: hs4, hs2, all_goods")


def build_trade_digraph(
    df: pd.DataFrame,
    type: str,                      # "hs4" | "hs2" | "all_goods"
    year: Optional[int] = None,
    product: Optional[str] = None,  # exact code only: len==4 for hs4, len==2 for hs2
    min_value: Optional[float] = None,
    min_quantity: Optional[float] = None,
    drop_self_loops: bool = True,
) -> nx.DiGraph:
    """Build a weighted directed graph of trade flows.

    Nodes are country IDs (strings from exporter_id/importer_id).
    Edge weight is the trade value column for the granularity.

    Parameters
    ----------
    df : DataFrame at the specified granularity (columns must include exporter_id, importer_id, and value column)
    type : hs4 | hs2 | all_goods
    year : optional year filter
    product : optional exact product code (len 4 for hs4, len 2 for hs2)
    min_value : optional filter to drop edges below this trade value (after product/year filter)
    min_quantity : optional filter to drop edges below this quantity
    drop_self_loops : remove edges where exporter == importer
    """
    value_col, qty_col, code_col, _ = _cols_for_type(type)

    # Normalize column types
    data = df.copy()
    for col in ("exporter_id", "importer_id"):
        if col in data.columns:
            data[col] = data[col].astype(str)
        else:
            raise KeyError(f"Missing column '{col}' in dataframe for type={type}")
    if code_col and code_col in data.columns:
        data[code_col] = data[code_col].astype(str)

    # Filter by year (if present)
    if year is not None and "year" in data.columns:
        data = data[data["year"] == year]

    # Exact product filter (only when type has a code column)
    if product is not None:
        product = str(product).strip()
        if code_col is None:
            raise ValueError("'product' filter is not valid for type='all_goods'")
        if type.lower() == "hs4" and len(product) != 4:
            raise ValueError("For type='hs4', product must be a 4-digit string")
        if type.lower() == "hs2" and len(product) != 2:
            raise ValueError("For type='hs2', product must be a 2-digit string")
        data = data[data[code_col] == product]

    # Keep only necessary columns
    keep = ["exporter_id", "importer_id", value_col]
    if qty_col in data.columns:
        keep.append(qty_col)
    if code_col:
        keep.append(code_col)
    if "year" in data.columns:
        keep.append("year")
    data = data[keep]

    # Optional numeric filters
    if min_value is not None:
        data = data[data[value_col] >= float(min_value)]
    if min_quantity is not None and qty_col in data.columns:
        data = data[data[qty_col] >= float(min_quantity)]

    # Drop self-loops if requested
    if drop_self_loops:
        data = data[data["exporter_id"] != data["importer_id"]]

    # Build DiGraph
    G = nx.DiGraph()

    # Aggregate duplicates just in case (exporter, importer) may have multiple rows
    agg = data.groupby(["exporter_id", "importer_id"], as_index=False)[value_col].sum()

    for _, row in agg.iterrows():
        u = row["exporter_id"]
        v = row["importer_id"]
        w = float(row[value_col])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    # Node rollups for convenience in metrics table
    out_strength = dict(G.out_degree(weight="weight"))
    in_strength = dict(G.in_degree(weight="weight"))
    nx.set_node_attributes(G, out_strength, name="export_value")
    nx.set_node_attributes(G, in_strength, name="import_value")
    total_trade = {n: out_strength.get(n, 0.0) + in_strength.get(n, 0.0) for n in G.nodes}
    nx.set_node_attributes(G, total_trade, name="total_trade")

    # Remember metadata on the graph for later
    G.graph["granularity"] = type.lower()
    G.graph["value_col"] = value_col
    G.graph["year"] = year
    return G


# ============================================================
# Centrality & community metrics
# ============================================================

def _inverse_lengthify(G: nx.DiGraph, eps: float = 1e-9) -> nx.DiGraph:
    H = G.copy()
    for u, v, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["length"] = 1.0 / (w + eps)
    return H


def compute_trade_metrics(G: nx.DiGraph, seed: int = 42) -> pd.DataFrame:
    """Compute MVP metrics for the trade network and return a tidy node table."""
    out_strength = dict(G.out_degree(weight="weight"))
    in_strength = dict(G.in_degree(weight="weight"))

    pr = nx.pagerank(G, weight="weight", alpha=0.85)
    hubs, auths = nx.hits(G, max_iter=1000, tol=1e-08, normalized=True)

    H = _inverse_lengthify(G)
    btw = nx.betweenness_centrality(H, weight="length", normalized=True)

    # Weighted Louvain communities
    comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed)
    comm_label = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            comm_label[n] = cid

    nodes = list(G.nodes())
    df = pd.DataFrame({
        "node": nodes,
        "export_strength": [out_strength.get(n, 0.0) for n in nodes],
        "import_strength": [in_strength.get(n, 0.0) for n in nodes],
        "pagerank":        [pr.get(n, 0.0)          for n in nodes],
        "hub_score":       [hubs.get(n, 0.0)        for n in nodes],
        "authority_score": [auths.get(n, 0.0)       for n in nodes],
        "betweenness":     [btw.get(n, 0.0)         for n in nodes],
        "community":       [comm_label.get(n, -1)   for n in nodes],
        "export_value":    [G.nodes[n].get("export_value", float("nan")) for n in nodes],
        "import_value":    [G.nodes[n].get("import_value", float("nan")) for n in nodes],
        "total_trade":     [G.nodes[n].get("total_trade", float("nan"))  for n in nodes],
    })

    # Percentiles for comparability inside slice
    for col in ("pagerank", "betweenness", "export_strength", "import_strength"):
        s = df[col]
        df[col + "_pct"] = s.rank(pct=True, method="average")

    return df


# ============================================================
# Slice selection utilities
# ============================================================

def top_hs2_codes(df_hs2: pd.DataFrame, year: Optional[int], top_n: int = 12) -> List[str]:
    value_col, _, code_col, _ = _cols_for_type("hs2")
    d = df_hs2.copy()
    if year is not None and "year" in d.columns:
        d = d[d["year"] == year]
    order = (
        d.groupby(code_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    return order


def top_hs4_within_hs2(df_hs4: pd.DataFrame, hs2_code: str, year: Optional[int], top_n: int = 5) -> List[str]:
    value_col, _, code_col, _ = _cols_for_type("hs4")
    d = df_hs4.copy()
    if year is not None and "year" in d.columns:
        d = d[d["year"] == year]
    d[code_col] = d[code_col].astype(str)
    # Derive HS2 from HS4 prefix
    d["hs2_from_hs4"] = d[code_col].str.slice(0, 2)
    d = d[d["hs2_from_hs4"] == str(hs2_code)]
    order = (
        d.groupby(code_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    return order


# ============================================================
# Orchestrated analysis (baseline + HS2 + HS4 deep dives)
# ============================================================

@dataclass
class AnalysisConfig:
    year: Optional[int]
    top_hs2: int = 12
    top_hs4_per_hs2: int = 5
    min_edges_per_slice: int = 0  # set >0 to skip tiny slices
    out_dir: str = "metrics/centrality"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def analyze_all_levels(
    df_all_goods: Optional[pd.DataFrame],
    df_hs2: Optional[pd.DataFrame],
    df_hs4: Optional[pd.DataFrame],
    cfg: AnalysisConfig,
) -> Dict[str, pd.DataFrame]:
    """Run analysis for All-goods baseline, top HS2, and HS4 deep dives.

    Returns a dict with keys: 'all_goods', 'hs2', 'hs4', each mapping to a concatenated
    DataFrame of node metrics for all slices within that level.
    """
    out: Dict[str, pd.DataFrame] = {}

    # --------------------------------------------------------
    # Baseline: All-goods (compute if missing)
    # --------------------------------------------------------
    if df_all_goods is None:
        if df_hs2 is not None:
            # Aggregate HS2 -> All-goods
            val_col, qty_col, code_col, _ = _cols_for_type("hs2")
            gcols = ["year", "exporter_id", "importer_id"] if "year" in df_hs2.columns else ["exporter_id", "importer_id"]
            grouped = df_hs2.groupby(gcols, as_index=False).agg({val_col: "sum", qty_col: "sum"})
            grouped = grouped.rename(columns={val_col: "total_value", qty_col: "total_quantity"})
            df_all_goods = grouped
        elif df_hs4 is not None:
            val_col, qty_col, code_col, _ = _cols_for_type("hs4")
            gcols = ["year", "exporter_id", "importer_id"] if "year" in df_hs4.columns else ["exporter_id", "importer_id"]
            grouped = df_hs4.groupby(gcols, as_index=False).agg({val_col: "sum", qty_col: "sum"})
            grouped = grouped.rename(columns={val_col: "total_value", qty_col: "total_quantity"})
            df_all_goods = grouped
        else:
            raise ValueError("At least one of df_all_goods, df_hs2, or df_hs4 must be provided")

    G_all = build_trade_digraph(df_all_goods, type="all_goods", year=cfg.year)
    base = compute_trade_metrics(G_all)
    base["level"] = "all_goods"
    base["code"] = "ALL"
    base["year"] = cfg.year
    out["all_goods"] = base.copy()

    # Persist
    base_dir = os.path.join(cfg.out_dir, "all_goods", f"year={cfg.year}")
    _ensure_dir(base_dir)
    base.to_parquet(os.path.join(base_dir, f"nodes_all_goods_{cfg.year}.parquet"), index=False)

    # --------------------------------------------------------
    # HS2 slices
    # --------------------------------------------------------
    hs2_tables: List[pd.DataFrame] = []
    if df_hs2 is not None:
        codes = top_hs2_codes(df_hs2, year=cfg.year, top_n=cfg.top_hs2)
        for code in codes:
            d = df_hs2.copy()
            if cfg.year is not None and "year" in d.columns:
                d = d[d["year"] == cfg.year]
            d = d[d["hs2_code"].astype(str) == str(code)]

            G = build_trade_digraph(d, type="hs2", year=cfg.year, product=str(code))
            if cfg.min_edges_per_slice and G.number_of_edges() < cfg.min_edges_per_slice:
                continue
            met = compute_trade_metrics(G)

            # Join baseline for deltas
            merged = met.merge(base[["node", "pagerank", "betweenness", "export_strength", "import_strength"]],
                               on="node", suffixes=("", "_base"), how="left")
            merged["delta_pagerank"] = merged["pagerank"] - merged["pagerank_base"]
            merged["delta_betweenness"] = merged["betweenness"] - merged["betweenness_base"]
            merged["delta_export_strength"] = merged["export_strength"] - merged["export_strength_base"]
            merged["delta_import_strength"] = merged["import_strength"] - merged["import_strength_base"]

            merged["level"] = "hs2"
            merged["code"] = str(code)
            merged["year"] = cfg.year
            hs2_tables.append(merged)

        if hs2_tables:
            hs2_df = pd.concat(hs2_tables, ignore_index=True)
            out["hs2"] = hs2_df
            base_dir = os.path.join(cfg.out_dir, "hs2", f"year={cfg.year}")
            _ensure_dir(base_dir)
            hs2_df.to_parquet(os.path.join(base_dir, f"nodes_hs2_{cfg.year}.parquet"), index=False)

    # --------------------------------------------------------
    # HS4 deep dives within each HS2
    # --------------------------------------------------------
    hs4_tables: List[pd.DataFrame] = []
    if df_hs4 is not None and df_hs2 is not None:
        hs2_candidates = top_hs2_codes(df_hs2, year=cfg.year, top_n=cfg.top_hs2)
        for hs2_code in hs2_candidates:
            hs4_top = top_hs4_within_hs2(df_hs4, hs2_code=str(hs2_code), year=cfg.year, top_n=cfg.top_hs4_per_hs2)
            for hs4_code in hs4_top:
                d = df_hs4.copy()
                if cfg.year is not None and "year" in d.columns:
                    d = d[d["year"] == cfg.year]
                d = d[d["hs4_code"].astype(str) == str(hs4_code)]

                G = build_trade_digraph(d, type="hs4", year=cfg.year, product=str(hs4_code))
                if cfg.min_edges_per_slice and G.number_of_edges() < cfg.min_edges_per_slice:
                    continue
                met = compute_trade_metrics(G)

                merged = met.merge(base[["node", "pagerank", "betweenness", "export_strength", "import_strength"]],
                                   on="node", suffixes=("", "_base"), how="left")
                merged["delta_pagerank"] = merged["pagerank"] - merged["pagerank_base"]
                merged["delta_betweenness"] = merged["betweenness"] - merged["betweenness_base"]
                merged["delta_export_strength"] = merged["export_strength"] - merged["export_strength_base"]
                merged["delta_import_strength"] = merged["import_strength"] - merged["import_strength_base"]

                merged["level"] = "hs4"
                merged["code"] = str(hs4_code)
                merged["parent_hs2"] = str(hs2_code)
                merged["year"] = cfg.year
                hs4_tables.append(merged)

        if hs4_tables:
            hs4_df = pd.concat(hs4_tables, ignore_index=True)
            out["hs4"] = hs4_df
            base_dir = os.path.join(cfg.out_dir, "hs4", f"year={cfg.year}")
            _ensure_dir(base_dir)
            hs4_df.to_parquet(os.path.join(base_dir, f"nodes_hs4_{cfg.year}.parquet"), index=False)

    return out


# ============================================================
# File I/O helpers (optional, for local parquet layout)
# ============================================================

def _default_parquet_path(base_dir: str, type_: str, year: int) -> str:
    # Expected pattern: base_dir/<TYPE>/<year>_<type>.parquet
    fname = f"{year}_{type_.lower()}.parquet"
    return os.path.join(base_dir, type_.upper() if type_ != "all_goods" else "all_goods", fname)


def load_parquet_if_exists(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MacroMap trade graph analysis")
    p.add_argument("--base-dir", type=str, default="parquet", help="Base directory containing HS4/HS2/all_goods parquet files")
    p.add_argument("--year", type=int, required=True, help="Year to analyze")
    p.add_argument("--top-hs2", type=int, default=12)
    p.add_argument("--top-hs4-per-hs2", type=int, default=5)
    p.add_argument("--min-edges-per-slice", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="metrics/centrality")
    p.add_argument("--hs4-path", type=str, default=None, help="Override path to HS4 parquet for the year")
    p.add_argument("--hs2-path", type=str, default=None, help="Override path to HS2 parquet for the year")
    p.add_argument("--all-path", type=str, default=None, help="Override path to all_goods parquet for the year")
    return p.parse_args()


def main_cli():
    args = _parse_args()

    # Resolve paths (allow overrides)
    hs4_path = args.hs4_path or _default_parquet_path(args.base_dir, "hs4", args.year)
    hs2_path = args.hs2_path or _default_parquet_path(args.base_dir, "hs2", args.year)
    all_path = args.all_path or _default_parquet_path(args.base_dir, "all_goods", args.year)

    print("[INFO] Loading parquet files:")
    print("       HS4:", hs4_path)
    print("       HS2:", hs2_path)
    print("       ALL:", all_path)

    df_hs4 = load_parquet_if_exists(hs4_path)
    df_hs2 = load_parquet_if_exists(hs2_path)
    df_all = load_parquet_if_exists(all_path)

    missing = []
    if df_hs4 is None: missing.append("HS4")
    if df_hs2 is None: missing.append("HS2")
    if df_all is None: missing.append("ALL_GOODS")
    if missing:
        print(f"[WARN] Missing inputs: {', '.join(missing)} — will compute fallbacks where possible.")

    cfg = AnalysisConfig(
        year=args.year,
        top_hs2=args.top_hs2,
        top_hs4_per_hs2=args.top_hs4_per_hs2,
        min_edges_per_slice=args.min_edges_per_slice,
        out_dir=args.out_dir,
    )

    results = analyze_all_levels(df_all, df_hs2, df_hs4, cfg)

    # Pretty summary
    for lvl, df in results.items():
        slices = df[["level", "code"]].drop_duplicates().shape[0]
        print(f"[DONE] Level={lvl:10s} rows={len(df):6d} unique slices={slices}")
    print(f"[OUT] Parquets written under: {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main_cli()
