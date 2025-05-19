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

# --- App Configuration ---
st.set_page_config(page_title="Data Transformer Pro Plus", layout="wide")
st.title("🛠️ Data Transformer Pro Plus — Robust ETL Web App")

# --- Session State ---
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
        typ = step['type']
        if typ == 'rename':
            df = df.rename(columns={step['old']: step['new']})
        elif typ == 'filter' and step.get('expr'):
            try:
                df = df.query(step['expr'])
            except:
                pass
        elif typ == 'compute' and step.get('expr'):
            expr = step['expr']
            if expr.lower().startswith('sql:'):
                try:
                    df = sqldf(expr[4:], {'df': df})
                except:
                    pass
            else:
                try:
                    df[step['new']] = df.eval(expr)
                except:
                    pass
        elif typ == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif typ == 'onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif typ == 'join':
            aux = st.session_state.datasets.get(step['aux'])
            if isinstance(aux, pd.DataFrame):
                df = df.merge(aux, left_on=step['left'], right_on=step['right'], how=step['how'])
        elif typ == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = df[c].fillna(df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0])
    return df

# --- UI Tabs Setup ---
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

# 1. Datasets
with tabs[0]:
    st.subheader("1. Load Data")
    files = st.file_uploader(
        "Upload CSV/Excel/Parquet/JSON files", 
        type=['csv','xls','xlsx','parquet','json'],
        accept_multiple_files=True
    )
    if files:
        for f in files:
            data = load_file(f)
            if isinstance(data, dict):
                for name, df in data.items():
                    st.session_state.datasets[f"{f.name}:{name}"] = df
            elif isinstance(data, pd.DataFrame):
                st.session_state.datasets[f.name] = data
        st.success(f"Loaded {len(files)} files.")
    if st.session_state.datasets:
        sel = st.selectbox(
            "Select dataset", 
            list(st.session_state.datasets.keys()), 
            key='load_sel'
        )
        st.session_state.current = sel
        st.data_editor(
            st.session_state.datasets[sel], 
            use_container_width=True
        )

# 2. Transform
with tabs[1]:
    st.subheader("2. Transform Data")
    if not st.session_state.current:
        st.info("Load a dataset first.")
    else:
        df = st.session_state.datasets[st.session_state.current]
        for idx, step in enumerate(st.session_state.steps):
            st.markdown(f"**Step {idx+1}:** {step['type']} - {step.get('desc','')}")
        op = st.selectbox("Operation", ['rename','filter','compute','drop_const','onehot','join','impute'], key='op')
        # Add operations here...
        if st.button("Apply Steps"):
            st.session_state.datasets[st.session_state.current] = apply_steps(df)
            st.success("Transformations applied.")
        st.data_editor(
            st.session_state.datasets[st.session_state.current], 
            use_container_width=True
        )

# 3. Profile
with tabs[2]:
    st.subheader("3. Profile")
    if st.session_state.current:
        df = st.session_state.datasets[st.session_state.current]
        profile = pd.DataFrame({
            'dtype': df.dtypes,
            'nulls': df.isna().sum(),
            'pct_null': df.isna().mean()*100
        })
        st.dataframe(profile, use_container_width=True)

# 4. Insights
with tabs[3]:
    st.subheader("4. Insights")
    if st.session_state.current:
        df = st.session_state.datasets[st.session_state.current]
        corr = df.select_dtypes(include=np.number).corr()
        if not corr.empty:
            st.plotly_chart(
                px.imshow(corr, text_auto=True), 
                use_container_width=True
            )

# 5. Export
with tabs[4]:
    st.subheader("5. Export & Writeback")
    if st.session_state.current:
        df = st.session_state.datasets[st.session_state.current]
        fmt = st.selectbox(
            "Format", 
            ['CSV','JSON','Parquet','Excel','Snowflake'], 
            key='export_fmt'
        )
        if st.button("Export"):
            if fmt == 'CSV':
                st.download_button("Download CSV", df.to_csv(index=False).encode(), "data.csv")
            elif fmt == 'JSON':
                st.download_button("Download JSON", df.to_json(orient='records'), "data.json")
            elif fmt == 'Parquet':
                st.download_button("Download Parquet", df.to_parquet(index=False), "data.parquet")
            elif fmt == 'Excel':
                out = BytesIO()
                df.to_excel(out, index=False, engine='openpyxl')
                st.download_button("Download Excel", out.getvalue(), "data.xlsx")
            else:
                table = re.sub(r"[^0-9A-Za-z_]+","_", st.session_state.current).upper()
                conn = get_sf_conn()
                cur = conn.cursor()
                defs = []
                for c, dt in df.dtypes.items():
                    if pd.api.types.is_integer_dtype(dt):
                        defs.append(f'"{c}" NUMBER')
                    elif pd.api.types.is_float_dtype(dt):
                        defs.append(f'"{c}" FLOAT')
                    elif pd.api.types.is_datetime64_any_dtype(dt):
                        defs.append(f'"{c}" TIMESTAMP_NTZ')
                    else:
                        max_len = int(df[c].astype(str).map(len).max() or 1)
                        defs.append(f'"{c}" VARCHAR({max_len})')
                ddl = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(defs)})"
                cur.execute(ddl)
                write_pandas(conn, df, table)
                st.success(f"Written to Snowflake table {table}")
                cur.close()
                conn.close()

# 6. History
with tabs[5]:
    st.subheader("6. History")
    for i,(ts,snap) in enumerate(st.session_state.versions):
        c1,c2=st.columns([4,1])
        c1.write(f"{i+1}. {ts}")
        if c2.button("Revert", key=f"rev_{i}"):
            st.session_state.datasets[st.session_state.current] = snap
            st.experimental_rerun()

# 7. Snowflake Config
with tabs[6]:
    st.subheader("7. Snowflake Config")
    st.text_input("Account", key='sf_account')
    st.text_input("Username", key='sf_user')
    st.text_input("Password", type='password', key='sf_password')
    st.text_input("Warehouse", key='sf_warehouse')
    st.text_input("Database", key='sf_database')
    st.text_input("Schema", key='sf_schema')

# 8. Pipeline YAML
with tabs[7]:
    st.subheader("8. Pipeline YAML")
    yaml_str = yaml.dump({'steps': st.session_state.steps}, sort_keys=False)
    st.text_area("YAML", yaml_str, height=200)
    st.download_button("Download YAML", yaml_str, "pipeline.yaml")

# 9. Social Graph
with tabs[8]:
    st.subheader("9. Social Graph")
    if st.session_state.current:
        df = st.session_state.datasets[st.session_state.current]
        cols = list(df.columns)
        src = st.selectbox("Source", cols, key='sg_src')
        tgt = st.selectbox("Target", cols, key='sg_tgt')
        wt = st.selectbox("Weight", [None]+cols, key='sg_wt')
        if st.button("Draw Graph"):
            G = nx.Graph()
            for _,r in df.iterrows():
                u,v = r[src], r[tgt]
                w = float(r[wt]) if wt and pd.notna(r[wt]) else 1
                G.add_edge(u, v, weight=G[u][v]['weight']+w if G.has_edge(u,v) else w)
            top_nodes = [n for n,_ in sorted(G.degree(), key=lambda x:-x[1])[:5]]
            top_edges = [(u,v) for u,v,d in sorted(G.edges(data=True), key=lambda x:-x[2]['weight'])[:5]]
            net = Network(height='600px', width='100%')
            for n in G.nodes():
                net.add_node(n, label=str(n), value=G.degree(n)*2 if n in top_nodes else G.degree(n))
            for u,v,d in G.edges(data=True):
                net.add_edge(u, v, value=d['weight'], width=4 if (u,v) in top_edges else 1)
            st.components.v1.html(net.generate_html(), height=600)
