import os
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pandasql import sqldf
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import yaml
import networkx as nx
from pyvis.network import Network
import plotly.express as px

# --- App Configuration ---
st.set_page_config(page_title="Data Transformer Pro Plus", layout="wide")
st.title("🛠️ Data Transformer Pro Plus — Robust ETL Web App")

# --- Tabs Definition ---
tabs = st.tabs([
    "1. Datasets", 
    "2. Transform", 
    "3. Profile", 
    "4. Export", 
    "5. Preview", 
    "6. History", 
    "7. Snowflake Settings", 
    "8. Pipeline Configuration YAML", 
    "9. Social Network Graph"
])

# --- Session State Initialization ---
for key, default in [('datasets', {}), ('current', None), ('steps', []), ('versions', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helper Functions ---
def get_sf_conn():
    return snowflake.connector.connect(
        user=st.session_state.get('sf_account',''),
        password=st.session_state.get('sf_password',''),
        account=st.session_state.get('sf_account',''),
        warehouse=st.session_state.get('sf_warehouse',''),
        database=st.session_state.get('sf_database',''),
        schema=st.session_state.get('sf_schema','')
    )

def load_file(uploader_file):
    ext = uploader_file.name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            return pd.read_csv(uploader_file)
        elif ext in ['xls', 'xlsx']:
            return pd.read_excel(uploader_file, sheet_name=None)
        elif ext == 'parquet':
            return pd.read_parquet(uploader_file)
        elif ext == 'json':
            return pd.read_json(uploader_file)
    except Exception as e:
        st.error(f"Failed to load {uploader_file.name}: {e}")
    return None


def apply_steps(df):
    # Snapshot before transform
    ts = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.versions.append((ts, df.copy()))
    for step in st.session_state.steps:
        t = step['type']
        if t == 'rename':
            df = df.rename(columns={step['old']: step['new']})
        elif t == 'filter':
            df = df.query(step['expr'])
        elif t == 'compute':
            df[step['new']] = df.eval(step['expr'])
        elif t == 'drop_const':
            const_cols = [c for c in df.columns if df[c].nunique() == 1]
            df = df.drop(columns=const_cols)
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif t == 'join':
            aux_df = st.session_state.datasets[step['aux']]
            df = df.merge(aux_df, left_on=step['left'], right_on=step['right'], how=step['how'])
        elif t == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    fill = df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0]
                    df[c] = df[c].fillna(fill)
    return df

# --- 1. Datasets ---
with tabs[0]:
    st.header("1. Datasets")
    files = st.file_uploader(
        "Upload files (CSV/Excel/Parquet/JSON)",
        type=['csv','xls','xlsx','parquet','json'], accept_multiple_files=True
    )
    if files:
        for u in files:
            data = load_file(u)
            if data is not None:
                st.session_state.datasets[u.name] = data
        st.success("Files loaded into session.")
    if st.session_state.datasets:
        sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()), key='sel_dataset')
        st.session_state.current = sel
        st.data_editor(st.session_state.datasets[sel], key=f"editor_{sel}", use_container_width=True)

# --- 2. Transform ---
with tabs[1]:
    st.header("2. Transform")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        for i, step in enumerate(st.session_state.steps):
            st.write(f"{i+1}. {step['type']} — {step.get('desc','')}")
        op = st.selectbox("Operation", ['rename','filter','compute','drop_const','onehot','join','impute'], key='op')
        if op == 'rename':
            old = st.selectbox("Old column", df.columns)
            new = st.text_input("New column name")
            if st.button("Add Rename"):
                st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f"{old}→{new}"})
        elif op == 'filter':
            expr = st.text_input("Filter expression (pandas query)")
            if st.button("Add Filter"):
                st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
        elif op == 'compute':
            newc = st.text_input("New column name")
            st.write("Use pandas eval expressions")
            expr2 = st.text_input("Formula:")
            if st.button("Add Compute"):
                st.session_state.steps.append({'type':'compute','new':newc,'expr':expr2,'desc':newc})
        elif op == 'drop_const':
            if st.button("Add Drop Constants"):
                st.session_state.steps.append({'type':'drop_const','desc':'Drop constants'})
        elif op == 'onehot':
            cols = st.multiselect("Columns to encode", df.select_dtypes(include=['object','category']).columns)
            if st.button("Add One-Hot"):
                st.session_state.steps.append({'type':'onehot','cols':cols,'desc':','.join(cols)})
        elif op == 'join':
            aux = st.selectbox("Aux dataset", [k for k in st.session_state.datasets if k!=key])
            left = st.selectbox("Left key", df.columns)
            right = st.selectbox("Right key", st.session_state.datasets[aux].columns)
            how = st.selectbox("Join type", ['inner','left','right','outer'])
            if st.button("Add Join"):
                st.session_state.steps.append({'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':aux})
        elif op == 'impute':
            if st.button("Add Impute"):
                st.session_state.steps.append({'type':'impute','desc':'Auto-impute missing'})
        if st.button("Apply Transformations"):
            st.session_state.datasets[key] = apply_steps(df)
            st.success("Transformations applied.")
            st.data_editor(st.session_state.datasets[key], key=f"transformed_{key}", use_container_width=True)

# --- 3. Profile / Export ---
with tabs[2]:
    st.header("3. Profile & Export")
    df = st.session_state.datasets.get(st.session_state.current)
    if df is not None:
        fmt = st.selectbox("Format", ['CSV','JSON','Parquet','Excel','Snowflake'], key='fmt')
        if st.button("Export"):
            if fmt=='CSV':
                st.download_button("Download CSV", df.to_csv(index=False).encode(), "data.csv")
            elif fmt=='JSON':
                st.download_button("Download JSON", df.to_json(orient='records'), "data.json")
            elif fmt=='Parquet':
                st.download_button("Download Parquet", df.to_parquet(index=False), "data.parquet")
            elif fmt=='Excel':
                out = BytesIO()
                df.to_excel(out, index=False, engine='openpyxl')
                st.download_button("Download Excel", out.getvalue(), "data.xlsx")
            else:
                tbl = st.text_input("Snowflake table name", key='exp_tbl')
                if st.button("Write to Snowflake"):
                    conn = get_sf_conn()
                    write_pandas(conn, df, tbl)
                    conn.close()
                    st.success(f"Written to {tbl}")

# --- 6. History ---
with tabs[5]:
    st.header("6. History")
    if st.session_state.versions:
        for idx, (ts, snap) in enumerate(st.session_state.versions):
            cols = st.columns([0.7, 0.3])
            cols[0].write(f"{idx+1}. {ts}")
            if cols[1].button("Revert", key=f"hist_{idx}"):
                st.session_state.datasets[st.session_state.current] = snap
                st.experimental_rerun()

# --- 7. Snowflake Settings ---
with tabs[6]:
    st.header("7. Snowflake Settings")
    # Add Snowflake configuration inputs here (account, warehouse, etc.)

# --- 8. Pipeline Configuration YAML ---
with tabs[7]:
    st.header("8. Pipeline Configuration YAML")
    yaml_str = yaml.dump({'pipeline_steps': st.session_state.steps}, sort_keys=False)
    st.text_area("Pipeline YAML", yaml_str, height=300)
    st.download_button("Download YAML", yaml_str, "pipeline.yaml")

# --- 9. Social Network Graph ---
with tabs[8]:
    st.header("9. Social Network Graph")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        cols = df.columns.tolist()
        src = st.selectbox("Source column", cols, key='src_col')
        tgt = st.selectbox("Target column", cols, key='tgt_col')
        wt_opt = [None] + cols
        wt = st.selectbox("Weight column (optional)", wt_opt, key='wt_col')
        if st.button("Generate Graph"):
            G = nx.Graph()
            for _, row in df.iterrows():
                u, v = row[src], row[tgt]
                w = float(row[wt]) if wt and pd.notna(row[wt]) else 1.0
                if G.has_edge(u, v):
                    G[u][v]['weight'] += w
                else:
                    G.add_edge(u, v, weight=w)
            edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            top5 = {(u, v) for u, v, _ in edges_sorted[:5]}
            net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
            net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=100, spring_strength=0.001)
            for n in G.nodes():
                net.add_node(n, label=str(n), title=f"Degree: {G.degree(n)}", value=G.degree(n))
            for u, v, data in G.edges(data=True):
                color = 'red' if (u, v) in top5 or (v, u) in top5 else 'rgba(200,200,200,0.2)'
                width = 4 if (u, v) in top5 or (v, u) in top5 else 1
                net.add_edge(u, v, value=data['weight'], width=width, color=color, title=f"Weight: {data['weight']}")
            net.show_buttons(filter_=['physics'])
            html = net.generate_html()
            import streamlit.components.v1 as components
            components.html(html, height=750, scrolling=True)
