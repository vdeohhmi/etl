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
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Transformer Pro Plus", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.markdown(
    "<h1 style='text-align:center;'>🛠️ Data Transformer Pro Plus</h1>"
    "<p style='text-align:center;color:gray;'>An ETL & Analysis studio — Created by Vishal</p>",
    unsafe_allow_html=True
)
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Datasets", "Transform", "Profile", "Insights", "Export", "History", "Settings", "Pipeline", "Social Graph"],
    index=0
)

# --- Global Session State Initialization ---
for key, default in [('datasets', {}), ('current', None), ('steps', []), ('versions', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helper Functions ---
def get_sf_conn():
    return snowflake.connector.connect(
        user=st.session_state.get('sf_user', ''),
        password=st.session_state.get('sf_password', ''),
        account=st.session_state.get('sf_account', ''),
        warehouse=st.session_state.get('sf_warehouse', ''),
        database=st.session_state.get('sf_database', ''),
        schema=st.session_state.get('sf_schema', '')
    )


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
            except:
                pass
    return df


def load_file(uploader):
    ext = uploader.name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            return parse_dates(pd.read_csv(uploader))
        if ext in ['xls', 'xlsx']:
            sheets = pd.read_excel(uploader, sheet_name=None)
            return {name: parse_dates(df) for name, df in sheets.items()}
        if ext == 'parquet':
            return parse_dates(pd.read_parquet(uploader))
        if ext == 'json':
            return parse_dates(pd.read_json(uploader))
    except Exception as e:
        st.error(f"Failed to load {uploader.name}: {e}")
    return None


def apply_steps(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.versions.append((ts, df.copy()))
    for step in st.session_state.steps:
        t = step['type']
        if t == 'rename':
            df = df.rename(columns={step['old']: step['new']})
        elif t == 'filter' and step.get('expr'):
            try: df = df.query(step['expr'])
            except: pass
        elif t == 'compute' and step.get('expr'):
            expr = step['expr']
            if expr.lower().startswith('sql:'):
                try: df = sqldf(expr[4:], {'df': df})
                except: pass
            else:
                try: df[step['new']] = df.eval(expr)
                except: pass
        elif t == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif t == 'join':
            aux = st.session_state.datasets.get(step['aux'])
            if isinstance(aux, pd.DataFrame):
                df = df.merge(aux, left_on=step['left'], right_on=step['right'], how=step['how'])
        elif t == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = df[c].fillna(
                        df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0]
                    )
    return df

# --- Pages ---
if page == "Datasets":
    st.header("Load Data")
    with st.container():
        col1, col2 = st.columns([2,1])
        with col1:
            files = st.file_uploader("Upload CSV/Excel/Parquet/JSON", type=['csv','xls','xlsx','parquet','json'], accept_multiple_files=True)
            if files:
                for f in files:
                    data = load_file(f)
                    if isinstance(data, dict):
                        for name, df in data.items():
                            st.session_state.datasets[f"{f.name}:{name}"] = df
                    elif isinstance(data, pd.DataFrame):
                        st.session_state.datasets[f.name] = data
                st.success(f"Loaded {len(files)} files.")
        with col2:
            if st.session_state.datasets:
                key = st.selectbox("Select dataset", list(st.session_state.datasets.keys()))
                st.session_state.current = key
                st.markdown(f"**Current:** {key}")
    if st.session_state.current:
        st.data_editor(st.session_state.datasets[st.session_state.current], key=f"datasets_{st.session_state.current}", use_container_width=True)

elif page == "Transform":
    st.header("Transform Data")
    if not st.session_state.current:
        st.info("Load a dataset first under 'Datasets'.")
    else:
        df = st.session_state.datasets[st.session_state.current]
        with st.expander("Current Steps", expanded=False):
            for i,s in enumerate(st.session_state.steps):
                st.write(f"{i+1}. {s['type']} — {s.get('desc','')}")
        op = st.selectbox("Operation", ['rename','filter','compute','drop_const','onehot','join','impute'])
        # Add operation UI blocks dynamically...
        if st.button("Apply Transformations"):
            st.session_state.datasets[st.session_state.current] = apply_steps(df)
            st.success("Transformations applied.")
        st.data_editor(st.session_state.datasets[st.session_state.current], key=f"transform_{st.session_state.current}", use_container_width=True)

elif page == "Profile":
    st.header("Data Profile")
    if st.session_state.current:
        df = st.session_state.datasets[st.session_state.current]
        profile = pd.DataFrame({
            'dtype':df.dtypes,'nulls':df.isna().sum(),'pct_null':df.isna().mean()*100
        })
        st.bar_chart(profile['pct_null'])
        st.dataframe(profile, use_container_width=True)

elif page == "Insights":
    st.header("Auto Insights")
    if st.session_state.current:
        df = st.session_state.datasets[st.session_state.current]
        corr = df.select_dtypes(include=np.number).corr()
        if not corr.empty:
            st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

elif page == "Export":
    st.header("Export & Writeback")
    if st.session_state.current:
        df = st.session_state.datasets[st.session_state.current]
        fmt = st.selectbox("Select format", ['CSV','JSON','Parquet','Excel','Snowflake'])
        if st.button("Export Now"):
            if fmt != 'Snowflake':
                out_map = {'CSV': df.to_csv, 'JSON': df.to_json, 'Parquet': df.to_parquet, 'Excel': df.to_excel}
                buf = BytesIO()
                if fmt == 'Excel': out_map[fmt](buf, index=False, engine='openpyxl')
                else: buf.write(out_map[fmt](index=False).encode() if fmt in ['CSV','JSON'] else df.to_parquet(index=False))
                st.download_button(f"Download {fmt}", buf.getvalue(), f"data.{fmt.lower()}")
            else:
                table = re.sub(r"[^0-9A-Za-z_]+","_",st.session_state.current).upper()
                conn = get_sf_conn(); cur=conn.cursor()
                defs=[]
                for c,dt in df.dtypes.items():
                    if pd.api.types.is_integer_dtype(dt): defs.append(f'"{c}" NUMBER')
                    elif pd.api.types.is_float_dtype(dt): defs.append(f'"{c}" FLOAT')
                    elif pd.api.types.is_datetime64_any_dtype(dt): defs.append(f'"{c}" TIMESTAMP_NTZ')
                    else: defs.append(f'"{c}" VARCHAR({int(df[c].astype(str).map(len).max() or 1)})')
                cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({','.join(defs)})")
                write_pandas(conn, df, table)
                st.success(f"Data loaded into Snowflake: {table}")

elif page == "History":
    st.header("Transformation History")
    for i,(ts,snap) in enumerate(st.session_state.versions):
        cols = st.columns([0.8,0.2])
        cols[0].write(f"**{i+1}. {ts}**")
        if cols[1].button("Revert", key=f"hist_{i}"):
            st.session_state.datasets[st.session_state.current] = snap
            st.experimental_rerun()

elif page == "Settings":
    st.header("Settings — Snowflake Config")
    with st.form("snowflake_form"):
        st.text_input("Account", key='sf_account')
        st.text_input("User", key='sf_user')
        st.text_input("Password", type='password', key='sf_password')
        st.text_input("Warehouse", key='sf_warehouse')
        st.text_input("Database", key='sf_database')
        st.text_input("Schema", key='sf_schema')
        submitted = st.form_submit_button("Save Settings")
        if submitted:
            st.success("Snowflake settings saved.")

elif page == "Pipeline":
    st.header("Pipeline as YAML")
    yaml_str = yaml.dump({'steps': st.session_state.steps}, sort_keys=False)
    st.code(yaml_str, language='yaml')
    st.download_button("Download YAML", yaml_str, "pipeline.yaml")

elif page == "Social Graph":
    st.header("Social Network Graph")
    if st.session_state.current:
        df = st.session_state.datasets[st.session_state.current]
        cols = list(df.columns)
        src = st.selectbox("Source node", cols, key='sg_src')
        tgt = st.selectbox("Target node", cols, key='sg_tgt')
        wt  = st.selectbox("Weight (optional)", [None]+cols, key='sg_wt')
        if st.button("Generate Graph"):
            G = nx.Graph()
            for _,r in df.iterrows():
                u,v = r[src], r[tgt]
                w = float(r[wt]) if wt and pd.notna(r[wt]) else 1
                G.add_edge(u,v,weight=G[u][v]['weight']+w if G.has_edge(u,v) else w)
            top_n = sorted(G.degree(), key=lambda x:-x[1])[:5]
            top_e = sorted(G.edges(data=True), key=lambda x:-x[2]['weight'])[:5]
            net = Network(height='600px', width='100%', bgcolor='#f1f1f1')
            for n in G.nodes():
                net.add_node(n, label=str(n), value=G.degree(n)*1.5)
            for u,v,d in G.edges(data=True):
                width = 3 if (u,v,d) in top_e else 1
                net.add_edge(u, v, value=d['weight'], width=width)
            st.components.v1.html(net.generate_html(), height=600)
