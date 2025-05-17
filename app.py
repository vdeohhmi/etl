import os
import re
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from openai import OpenAI
from sqlalchemy import create_engine
import networkx as nx
from pyvis.network import Network
import plotly.express as px

# --- App Configuration ---
st.set_page_config(page_title="Data Wizard X Pro", layout="wide")
st.title("🔮 Data Wizard X Pro — Direct ETL & AI Integration")

# --- Session State Initialization ---
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
for name, default in [('datasets', {}), ('current_df', None)]:
    init_state(name, default)

# --- OpenAI Client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Please set the OPENAI_API_KEY environment variable for AI features.")
client = OpenAI(api_key=api_key)

# --- Helpers ---
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s]+", "_", regex=True)
        .str.replace(r"[^0-9a-z_]+", "", regex=True)
    )
    return df

@st.cache_data
def ai_pandas_expr(new_col: str, logic: str, sample: pd.DataFrame) -> str:
    prompt = (
        f"You are a Python data engineer. Given sample rows {sample.to_dict('records')} "
        f"and the requirement: {logic}, generate a valid pandas eval expression for '{new_col}'. "
        "Use only snake_case column names. Respond with only the expression."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip().strip('`')

@st.cache_data
def ai_sql_expr(new_col: str, logic: str) -> str:
    prompt = (
        f"You are a SQL expert working with SQLite. Table is named 'df' and columns are snake_case. "
        f"Write a SELECT query that returns *, and computes a new column '{new_col}' as {logic}. "
        "Use SQLite functions like julianday() for date diffs. Return only the SQL query."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    sql = resp.choices[0].message.content.strip().strip('`')
    return normalize_sql(sql, new_col)

# Normalize SQL for SQLite compatibility
def normalize_sql(sql: str, alias: str) -> str:
    safe_alias = re.sub(r"[^0-9a-z_]+", "_", alias.strip().lower())
    sql = sql.strip().lstrip(';')
    sql = re.sub(r'^(sql\s+|select\s+\*)', 'SELECT *', sql, flags=re.IGNORECASE)
    def repl(m):
        a, b = [c.strip() for c in m.group(1).split(',')]
        return f"(julianday({a}) - julianday({b}))"
    sql = re.sub(r'DATEDIFF\s*\(([^)]+)\)', repl, sql, flags=re.IGNORECASE)
    sql = re.sub(r'AS\s+"?[0-9a-z_]+"?', f"AS {safe_alias}", sql, flags=re.IGNORECASE)
    return sql

# Load file and sanitize columns
def load_file(f) -> pd.DataFrame:
    ext = f.name.split('.')[-1].lower()
    try:
        if ext=='csv': df = pd.read_csv(f)
        elif ext in ('xls','xlsx'): df = pd.read_excel(f)
        elif ext=='parquet': df = pd.read_parquet(f)
        elif ext=='json': df = pd.read_json(f)
        else: return None
        return sanitize_columns(df)
    except Exception as e:
        st.error(f"Failed to load {f.name}: {e}")
        return None

# Main UI Tabs
tab_ds, tab_tf, tab_ai, tab_pr, tab_ex, tab_graph, tab_sf = st.tabs([
    "📂 Datasets", "✏️ Transform", "🤖 AI Toolkit", "📈 Profile",
    "⬇️ Export", "🌐 Graph", "⚙️ Snowflake"
])

# 1) Datasets
with tab_ds:
    st.header("1. Datasets")
    uploads = st.file_uploader("Upload CSV/Excel/Parquet/JSON files", accept_multiple_files=True)
    if uploads:
        for f in uploads:
            df = load_file(f)
            if df is not None:
                st.session_state.datasets[f.name] = df
        st.success("Loaded and sanitized datasets.")
    if st.session_state.datasets:
        sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()))
        st.session_state.current_df = sel
        st.dataframe(st.session_state.datasets[sel].head(10), use_container_width=True)

# 2) Transform
with tab_tf:
    st.header("2. Transform")
    key = st.session_state.current_df
    if not key:
        st.info("Load and select a dataset first.")
    else:
        df = st.session_state.datasets[key]
        st.subheader("Preview Before")
        st.dataframe(df.head(5), use_container_width=True)

        op = st.selectbox("Operation", [
            'Rename Column','Filter Rows','Compute Column','SQL Transform',
            'Drop Constant Columns','One-Hot Encode','Impute Missing'
        ], key='op_tf')

        # Rename Column
        if op=='Rename Column':
            old = st.selectbox("Old column", df.columns)
            new = st.text_input("New column name")
            if st.button("Apply Rename"):
                df = df.rename(columns={old:new})
                st.session_state.datasets[key] = df
                st.success(f"Renamed '{old}' to '{new}'.")

        # Filter Rows
        if op=='Filter Rows':
            expr = st.text_input("Filter expression (pandas query)")
            if st.button("Apply Filter") and expr:
                df = df.query(expr)
                st.session_state.datasets[key] = df
                st.success("Filter applied.")

        # Compute Column
        if op=='Compute Column':
            new_col = st.text_input("New column name (snake_case)")
            logic = st.text_area("Describe logic (plain English)")
            manual = st.text_input("Or manual pandas expression")
            if st.button("Generate Expression via AI"):
                expr = ai_pandas_expr(new_col, logic, df.head(3))
                st.code(expr)
            else:
                expr = manual
            if expr and st.button("Apply Compute"):
                try:
                    df[new_col] = df.eval(expr, engine='python')
                    st.session_state.datasets[key] = df
                    st.success(f"Computed '{new_col}'.")
                except Exception as e:
                    st.error(f"Compute failed: {e}")

        # SQL Transform
        if op=='SQL Transform':
            new_col_sql = st.text_input("New column name (snake_case, SQL)")
            logic_sql = st.text_area("Describe SQL logic (plain English)")
            manual_sql = st.text_area("Or manual SQL query (use 'df' as table)")
            if st.button("Generate SQL via AI"):
                sql = ai_sql_expr(new_col_sql, logic_sql)
                st.code(sql)
            else:
                sql = manual_sql
            if sql and st.button("Apply SQL Transform"):
                try:
                    engine = create_engine('sqlite:///:memory:')
                    df.to_sql('df', engine, index=False)
                    df = pd.read_sql(sql, engine)
                    st.session_state.datasets[key] = df
                    st.success("SQL transform applied.")
                except Exception as e:
                    st.error(f"SQL failed: {e}")

        # Drop Constant Columns
        if op=='Drop Constant Columns' and st.button("Apply Drop Constants"):
            df = df.loc[:, df.nunique()>1]
            st.session_state.datasets[key] = df
            st.success("Dropped constant columns.")

        # One-Hot Encode
        if op=='One-Hot Encode':
            cols = st.multiselect("Columns to encode", df.select_dtypes('object').columns)
            if st.button("Apply One-Hot") and cols:
                df = pd.get_dummies(df, columns=cols)
                st.session_state.datasets[key] = df
                st.success("One-hot encoding applied.")

        # Impute Missing
        if op=='Impute Missing' and st.button("Apply Impute"):
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = df[c].fillna(
                        df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode()[0]
                    )
            st.session_state.datasets[key] = df
            st.success("Imputation complete.")

        # Preview After
        st.subheader("Preview After")
        st.dataframe(st.session_state.datasets[key].head(5), use_container_width=True)

# 3) AI Toolkit
with tab_ai:
    st.header("3. AI Toolkit")
    st.write("Use the Transform tab's AI buttons to generate code snippets.")

# 4) Profile
with tab_pr:
    st.header("4. Profile")
    key = st.session_state.current_df
    if key:
        df = st.session_state.datasets[key]
        stats = pd.DataFrame({
            'dtype': df.dtypes,
            'nulls': df.isna().sum(),
            'pct_null': df.isna().mean()*100
        })
        st.dataframe(stats, use_container_width=True)

# 5) Export
with tab_ex:
    st.header("5. Export")
    key = st.session_state.current_df
    if key:
        df = st.session_state.datasets[key]
        fmt = st.selectbox("Format", ['CSV','Excel','Parquet','Snowflake'])
        if fmt!='Snowflake' and st.button("Download File"):
            buf = BytesIO()
            if fmt=='CSV':
                st.download_button("CSV", df.to_csv(index=False).encode(),"data.csv")
            elif fmt=='Excel':
                df.to_excel(buf,index=False,engine='openpyxl')
                st.download_button("Excel",buf.getvalue(),"data.xlsx")
            else:
                st.download_button("Parquet", df.to_parquet(index=False),"data.parquet")
        if fmt=='Snowflake':
            st.info("Configure and write in Snowflake tab.")

# 6) Social Graph
with tab_graph:
    st.header("6. Social Network Graph")
    key = st.session_state.current_df
    if key:
        df = st.session_state.datasets[key]
        src = st.selectbox("Source column", df.columns, key='g_src')
        tgt = st.selectbox("Target column", df.columns, key='g_tgt')
        wt = st.selectbox("Weight column (optional)", [None]+list(df.columns), key='g_wt')
        if st.button("Generate Graph"):
            G = nx.Graph()
            for _, row in df.iterrows():
                u, v = row[src], row[tgt]
                w = float(row[wt]) if wt else 1.0
                if G.has_edge(u, v):
                    G[u][v]['weight'] += w
                else:
                    G.add_edge(u, v, weight=w)
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            net.show_buttons(filter_=['physics'])
            for n in G.nodes():
                net.add_node(n, label=str(n), title=f"Degree: {G.degree(n)}", value=G.degree(n))
            for u, v, d in G.edges(data=True):
                net.add_edge(u, v, value=d['weight'], width=4 if d['weight'] == max([dd['weight'] for _,_,dd in G.edges(data=True)]) else 1)
            html = net.generate_html()
            import streamlit.components.v1 as components
            components.html(html, height=650)

# 7) Snowflake
with tab_sf:
    st.header("7. Snowflake Settings & Export")
    acc = st.text_input("Account")
    usr = st.text_input("User")
    pwd = st.text_input("Password", type='password')
    wh = st.text_input("Warehouse")
    db = st.text_input("Database")
    sc = st.text_input("Schema")
    tbl = st.text_input("Table Name")
    if st.button("Write to Snowflake") and acc and usr and pwd and wh and db and sc and tbl:
        df = st.session_state.datasets[key]
        conn = snowflake.connector.connect(
            user=usr, password=pwd, account=acc,
            warehouse=wh, database=db, schema=sc
        )
        write_pandas(conn, df, tbl)
        conn.close()
        st.success(f"Written to {tbl}.")
