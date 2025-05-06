import os
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sqlalchemy import create_engine
from pandasql import sqldf
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import networkx as nx
import yaml

# --- App Configuration ---
st.set_page_config(page_title="Data Transformer Pro Plus", layout="wide")
st.title("🛠️ Data Transformer Pro Plus — Robust ETL Web App")

# --- Session State Init ---
for key, default in [('datasets', {}), ('current', None), ('steps', []), ('versions', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Snowflake Helper ---
def get_sf_conn():
    return snowflake.connector.connect(
        user=st.session_state.sf_user,
        password=st.session_state.sf_password,
        account=st.session_state.sf_account,
        warehouse=st.session_state.sf_warehouse,
        database=st.session_state.sf_database,
        schema=st.session_state.sf_schema
    )

# --- File Loader ---
def load_file(u):
    ext = u.name.split('.')[-1].lower()
    try:
        if ext == 'csv': return pd.read_csv(u)
        if ext in ['xls', 'xlsx']: return pd.read_excel(u, sheet_name=None)
        if ext == 'parquet': return pd.read_parquet(u)
        if ext == 'json': return pd.read_json(u)
    except Exception as e:
        st.error(f"Failed to load {u.name}: {e}")
    return None

# --- Transformation Engine ---
def apply_steps(df):
    for step in st.session_state.steps:
        t = step['type']
        if t == 'rename':
            df = df.rename(columns={step['old']: step['new']})
        elif t == 'filter':
            expr = step['expr']
            if expr:
                try: df = df.query(expr)
                except: st.warning(f"Invalid filter: {expr}")
        elif t == 'compute':
            expr = step['expr']
            if expr.lower().startswith('sql:'):
                q = expr[4:].strip()
                try: df = sqldf(q, {'df': df})
                except: st.warning(f"Invalid SQL: {q}")
            else:
                try: df[step['new']] = df.eval(expr)
                except: st.warning(f"Invalid formula: {expr}")
        elif t == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif t == 'join' and step.get('aux'):
            aux_df = st.session_state.datasets.get(step['aux'])
            if aux_df is not None:
                df = df.merge(aux_df, left_on=step['left'], right_on=step['right'], how=step['how'])
        elif t == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    if pd.api.types.is_numeric_dtype(df[c]): df[c].fillna(df[c].median(), inplace=True)
                    else: df[c].fillna(df[c].mode().iloc[0], inplace=True)
    return df

# --- UI Tabs ---
tabs = st.tabs([
    "📂 Datasets",
    "✏️ Transform",
    "📈 Profile",
    "💡 Insights",
    "⬇️ Export",
    "🕒 History",
    "⚙️ Snowflake",
    "📜 Pipeline",
    "🕸️ Social Graph"
])

# Datasets Tab

def datasets_tab():
    st.header("1. Upload & Manage Datasets")
    files = st.file_uploader("Upload files (CSV/Excel/Parquet/JSON)", type=['csv','xls','xlsx','parquet','json'], accept_multiple_files=True)
    if files:
        for u in files:
            data = load_file(u)
            if isinstance(data, dict):
                for sheet, sdf in data.items():
                    key = f"{u.name}:{sheet}"
                    st.session_state.datasets[key] = sdf
            elif data is not None:
                st.session_state.datasets[u.name] = data
        st.success(f"Loaded {len(files)} files into {len(st.session_state.datasets)} datasets.")
    if st.session_state.datasets:
        sel = st.selectbox("Select active dataset", list(st.session_state.datasets.keys()), key='sel_dataset')
        st.session_state.current = sel
        st.data_editor(st.session_state.datasets[sel], key=f"editor_{sel}", use_container_width=True)

# Transform Tab

def transform_tab():
    st.header("2. Build Transformations")
    key = st.session_state.current
    if not key:
        st.info("Upload and select a dataset first.")
        return
    df = st.session_state.datasets[key]
    for i, step in enumerate(st.session_state.steps):
        st.write(f"{i+1}. {step['type']} — {step.get('desc','')}")
    st.markdown("---")
    op = st.selectbox("Operation", ['rename','filter','compute','drop_const','onehot','join','impute'], key='op_transform')
    if op == 'rename':
        old = st.selectbox("Old column", df.columns)
        new = st.text_input("New column name")
        if st.button("Add Rename"):
            st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f"Rename {old}→{new}"})
    elif op == 'filter':
        expr = st.text_input("Filter expression")
        st.write("Columns:", df.columns.tolist())
        if st.button("Add Filter"):
            st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
    elif op == 'compute':
        newc = st.text_input("New column name", help="Name for the computed column.")
        st.write("Columns:", df.columns.tolist())
        st.write("Functions: np.log(), np.sqrt(); prefix SQL: for SQL queries")
        expr2 = st.text_input("Formula or SQL:", help="Enter formula or SQL: query.")
        if st.button("Add Compute"):
            st.session_state.steps.append({'type':'compute','new':newc,'expr':expr2,'desc':f"Compute {newc}"})
    elif op == 'drop_const':
        if st.button("Add Drop Constants"): st.session_state.steps.append({'type':'drop_const','desc':'Drop constants'})
    elif op == 'onehot':
        cols = st.multiselect("Columns to encode", df.select_dtypes('object').columns)
        if st.button("Add One-Hot"): st.session_state.steps.append({'type':'onehot','cols':cols,'desc':f"One-hot {cols}"})
    elif op == 'join':
        aux = st.selectbox("Aux dataset", [k for k in st.session_state.datasets if k != key])
        left = st.selectbox("Left key", df.columns)
        right = st.selectbox("Right key", st.session_state.datasets[aux].columns)
        how = st.selectbox("Join type", ['inner','left','right','outer'])
        if st.button("Add Join"): st.session_state.steps.append({'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':f"Join {aux}"})
    elif op == 'impute':
        if st.button("Add Auto-Impute"): st.session_state.steps.append({'type':'impute','desc':'Auto-impute'})
    if st.button("Apply Transformations"): st.session_state.datasets[key] = apply_steps(df); st.success("Applied steps.")
    st.data_editor(st.session_state.datasets[key], key=f"transformed_{key}", use_container_width=True)

# Profile Tab

def profile_tab():
    st.header("3. Data Profile")
    key = st.session_state.current
    if not key:
        st.info("Select a dataset first.")
        return
    df = st.session_state.datasets[key]
    stats = pd.DataFrame({'dtype': df.dtypes, 'nulls': df.isna().sum(), 'null_pct': df.isna().mean()*100})
    st.dataframe(stats, use_container_width=True)

# Insights Tab

def insights_tab():
    st.header("4. Auto Insights")
    key = st.session_state.current
    if not key:
        st.info("Select a dataset first.")
        return
    df = st.session_state.datasets[key]
    num = df.select_dtypes('number')
    if not num.empty:
        fig = px.imshow(num.corr(), text_auto=True, title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
        miss = df.isna()
        fig2 = px.imshow(miss, title='Missingness Heatmap')
        st.plotly_chart(fig2, use_container_width=True)

# Export Tab

def export_tab():
    st.header("5. Export & Writeback")
    key = st.session_state.current
    if not key:
        st.info("Select a dataset first.")
        return
    df = st.session_state.datasets[key]
    fmt = st.selectbox("Format", ['CSV','JSON','Parquet','Excel','Snowflake'], key='fmt')
    if st.button("Export"):
        if fmt == 'CSV': st.download_button("Download CSV", df.to_csv(index=False).encode(), "data.csv")
        elif fmt == 'JSON': st.download_button("Download JSON", df.to_json(orient='records'), "data.json")
        elif fmt == 'Parquet': st.download_button("Download Parquet", df.to_parquet(index=False), "data.parquet")
        elif fmt == 'Excel': out = BytesIO(); df.to_excel(out, index=False, engine='openpyxl'); st.download_button("Download Excel", out.getvalue(), "data.xlsx")
        else:
            with st.form("sf_wb"): tbl = st.text_input("Snowflake table"); sub = st.form_submit_button("Writeback")
            if sub: conn = get_sf_conn(); write_pandas(conn, df, tbl); conn.close(); st.success(f"Written to {tbl}")

# History Tab

def history_tab():
    st.header("6. History")
    for idx,(ts,snap) in enumerate(st.session_state['versions']):
        cols = st.columns([0.7,0.3])
        cols[0].write(f"{idx+1}. {ts}")
        if cols[1].button("Revert", key=f"h_{idx}"): st.session_state.datasets[st.session_state.current]=snap; st.experimental_rerun()

# Snowflake Tab

def snowflake_tab():
    st.header("7. Snowflake Settings")
    st.text_input("Account", key='sf_account')
    st.text_input("User", key='sf_user')
    st.text_input("Password", type='password', key='sf_password')
    st.text_input("Warehouse", key='sf_warehouse')
    st.text_input("Database", key='sf_database')
    st.text_input("Schema", key='sf_schema')

# Pipeline Tab

def pipeline_tab():
    st.header("8. Pipeline Configuration as YAML")
    yaml_steps = yaml.dump({'pipeline_steps': st.session_state.steps}, sort_keys=False)
    st.text_area("Pipeline YAML", yaml_steps, height=300)
    st.download_button("Download YAML", yaml_steps, file_name="pipeline.yaml", mime="text/yaml")

# Social Graph Tab

# Social Graph Tab

def social_graph_tab():
    st.header("9. Social Network Graph")
    key = st.session_state.current
    if not key:
        st.info("Select a dataset containing edge list first.")
        return
    df = st.session_state.datasets[key]
    cols = list(df.columns)
    source = st.selectbox("Source node column", cols, key='src_col')
    target = st.selectbox("Target node column", cols, key='tgt_col')
    weight_opt = [None] + cols
    weight = st.selectbox("Edge weight column (optional)", weight_opt, key='wt_col')
    if st.button("Generate Graph"):
        # Build graph with weights
        G = nx.Graph()
        for _, row in df.iterrows():
            u, v = row[source], row[target]
            w = float(row[weight]) if weight and weight in row and pd.notna(row[weight]) else 1.0
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        # Identify top-5 heaviest edges
        edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        top5 = {(u, v) for u, v, d in edges_sorted[:5]}
        # PyVis network
        net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
        net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=100, spring_strength=0.001)
        # Add nodes
        for n in G.nodes():
            deg = G.degree(n)
            net.add_node(n, label=str(n), title=f"Degree: {deg}", value=deg)
        # Add edges with styling
        for u, v, data in G.edges(data=True):
            w = data['weight']
            if (u, v) in top5 or (v, u) in top5:
                net.add_edge(u, v, value=w, width=4, color='red', title=f"Weight: {w}")
            else:
                net.add_edge(u, v, value=w, width=1, color='rgba(200,200,200,0.2)', title=f"Weight: {w}")
        # Show physics controls
        net.show_buttons(filter_=['physics'])
        html = net.generate_html()
        import streamlit.components.v1 as components
        components.html(html, height=750, scrolling=True)

# --- Render Tabs ---
funcs = [
    datasets_tab,
    transform_tab,
    profile_tab,
    insights_tab,
    export_tab,
    history_tab,
    snowflake_tab,
    pipeline_tab,
    social_graph_tab
]
for fn, tab in zip(funcs, tabs):
    with tab:
        fn()
