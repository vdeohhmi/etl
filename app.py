import os
import re
import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from io import BytesIO
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from openai import OpenAI
from sqlalchemy import create_engine
import dask.dataframe as dd
# Correct AgGrid import
from st_aggrid import AgGrid, GridOptionsBuilder
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# --- App Configuration ---
st.set_page_config(page_title="Data Wizard X Pro", layout="wide")
st.title("🔮 Data Wizard X Pro — Polars, Dask & Advanced Visuals")

# --- Session State Initialization ---
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
for k, v in [('datasets', {}), ('current', None)]:
    init_state(k, v)

# --- OpenAI Client ---
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error('Please set OPENAI_API_KEY environment variable to enable AI features.')
client = OpenAI(api_key=api_key)

# --- Helpers ---
def sanitize_cols(pl_df: pl.DataFrame) -> pl.DataFrame:
    cols = [re.sub(r'[^0-9a-z_]+', '_', c.strip().lower().replace(' ', '_')) for c in pl_df.columns]
    return pl_df.rename({old: new for old, new in zip(pl_df.columns, cols)})

@st.cache_data
def ai_polars_expr(newcol, logic, sample: pl.DataFrame) -> str:
    prompt = (
        f"You are a Polars data engineer. Given sample rows {sample.to_dicts()} and logic: '{logic}', "
        f"generate a Polars expression to create a new column '{newcol}'. Return only the expression."
    )
    resp = client.chat.completions.create(model='gpt-4o-mini',messages=[{'role':'user','content':prompt}])
    return resp.choices[0].message.content.strip().strip('`')

@st.cache_data
def ai_dask_query(logic, columns):
    prompt = (
        f"You are a Dask data engineer. Given columns {columns} and logic: '{logic}', "
        "generate a Dask dataframe filter expression. Return only the expression."
    )
    resp = client.chat.completions.create(model='gpt-4o-mini',messages=[{'role':'user','content':prompt}])
    return resp.choices[0].message.content.strip().strip('`')

@st.cache_data
def ai_sql_query(newcol, logic):
    prompt = (
        f"You are a SQL expert using SQLite. Table is named 'df'. Write a SELECT *, "
        f"{logic} AS {newcol} FROM df; Return only the SQL query."
    )
    resp = client.chat.completions.create(model='gpt-4o-mini',messages=[{'role':'user','content':prompt}])
    return normalize_sql(resp.choices[0].message.content.strip().strip('`'))

def normalize_sql(sql: str) -> str:
    sql = sql.strip().lstrip('sql ').lstrip(';')
    selects = re.findall(r'(?i)select\s*\*', sql)
    if len(selects) > 1:
        sql = re.sub(r'(?i)select\s*\*', '', sql, count=len(selects)-1)
        sql = 'SELECT * ' + sql
    def repl(m):
        a, b = [c.strip() for c in m.group(1).split(',')]
        return f"(julianday({a}) - julianday({b}))"
    sql = re.sub(r'(?i)DATEDIFF\s*\(([^)]+)\)', repl, sql)
    if 'from df' not in sql.lower():
        if 'select' in sql.lower():
            sql = sql.replace('SELECT *', 'SELECT * FROM df')
        else:
            sql = 'SELECT * FROM df'
    return sql

# --- Load File as Polars DataFrame ---
def load_file(f) -> pl.DataFrame:
    ext = f.name.split('.')[-1].lower()
    try:
        if ext=='csv': df = pl.read_csv(f)
        elif ext in ('xls','xlsx'): df = pl.read_excel(f)
        elif ext=='parquet': df = pl.read_parquet(f)
        elif ext=='json': df = pl.read_json(f)
        else: return None
        return sanitize_cols(df)
    except Exception as e:
        st.error(f"Failed to load {f.name}: {e}")
        return None

# --- Main UI Tabs ---
tabs = st.tabs(['Datasets','Transform','Profile','Export','Graph','Snowflake','AI'])

# 1) Datasets
with tabs[0]:
    st.header('1. Datasets')
    files = st.file_uploader('Upload CSV/Excel/Parquet/JSON', accept_multiple_files=True)
    if files:
        for f in files:
            df = load_file(f)
            if df is not None:
                st.session_state.datasets[f.name] = df
        st.success('Datasets loaded.')
    if st.session_state.datasets:
        key = st.selectbox('Select dataset', list(st.session_state.datasets.keys()))
        st.session_state.current = key
        df = st.session_state.datasets[key]
        gb = GridOptionsBuilder.from_dataframe(df.to_pandas())
        gb.configure_pagination(); gb.configure_side_bar()
        AgGrid(df.to_pandas(), gridOptions=gb.build(), enable_enterprise_modules=False)

# 2) Transform
with tabs[1]:
    st.header('2. Transform')
    key = st.session_state.current
    if not key:
        st.info('Load and select a dataset first')
    else:
        df_pl = st.session_state.datasets[key]
        st.subheader('Preview Before')
        AgGrid(df_pl.to_pandas(), enable_enterprise_modules=False)
        op = st.selectbox('Operation', ['Polars Compute','Dask Filter','SQL Execute','Drop Const','One-Hot','Impute'])

        # Polars Compute
        if op=='Polars Compute':
            newcol = st.text_input('New column name')
            logic = st.text_area('Logic (English)')
            manual = st.text_input('Or manual Polars expression')
            if st.button('Generate Expression'):
                expr = ai_polars_expr(newcol, logic, df_pl.head(5))
                st.code(expr)
            else:
                expr = manual
            if expr and st.button('Apply Polars Compute'):
                try:
                    df_new = df_pl.with_columns(pl.eval(expr).alias(newcol))
                    st.session_state.datasets[key] = df_new
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f'Polars compute failed: {e}')

        # Dask Filter
        if op=='Dask Filter':
            logic = st.text_area('Filter logic (English)')
            manual = st.text_input('Or manual Dask expression')
            if st.button('Generate Filter'):
                expr = ai_dask_query(logic, list(df_pl.columns))
                st.code(expr)
            else:
                expr = manual
            if expr and st.button('Apply Dask Filter'):
                try:
                    df_dd = dd.from_pandas(df_pl.to_pandas(), npartitions=2)
                    df_filtered = df_dd.query(expr).compute()
                    df_new = pl.from_pandas(df_filtered)
                    st.session_state.datasets[key] = df_new
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f'Dask filter failed: {e}')

        # SQL Execute
        if op=='SQL Execute':
            newcol = st.text_input('New column name (SQL)')
            logic = st.text_area('Logic (English for SQL)')
            manual = st.text_area('Or manual SQL (use table df)')
            if st.button('Generate SQL'):
                sql = ai_sql_query(newcol, logic)
                st.code(sql)
            else:
                sql = manual
            if sql and st.button('Apply SQL'):
                try:
                    engine = create_engine('sqlite:///:memory:')
                    df_pl.to_pandas().to_sql('df', engine, index=False)
                    df_sql = pd.read_sql(sql, engine)
                    df_new = pl.from_pandas(df_sql)
                    st.session_state.datasets[key] = df_new
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f'SQL execution failed: {e}')

        # Drop constant columns
        if op=='Drop Const' and st.button('Apply Drop Constants'):
            df_new = df_pl.select(pl.exclude(pl.all().filter(lambda s: df_pl[n := s.name].n_unique() <= 1)))
            st.session_state.datasets[key] = df_new
            st.experimental_rerun()

        # One-Hot Encode
        if op=='One-Hot' and st.button('Apply One-Hot'):
            df_new_pd = pd.get_dummies(df_pl.to_pandas())
            df_new = pl.from_pandas(df_new_pd)
            st.session_state.datasets[key] = df_new
            st.experimental_rerun()

        # Impute Missing
        if op=='Impute' and st.button('Apply Impute'):
            df_new = df_pl.fill_null(strategy='median')
            st.session_state.datasets[key] = df_new
            st.experimental_rerun()

# 3) Profile
with tabs[2]:
    st.header('3. Profile')
    key = st.session_state.current
    if key:
        df_pl = st.session_state.datasets[key]
        stats = df_pl.describe().to_pandas()
        st.dataframe(stats)

# 4) Export
with tabs[3]:
    st.header('4. Export')
    key = st.session_state.current
    if key:
        df_pl = st.session_state.datasets[key]
        fmt = st.selectbox('Format', ['CSV','Parquet','JSON'], key='exp_fmt')
        if st.button('Download'):
            buf = BytesIO()
            if fmt=='CSV':
                st.download_button('CSV', df_pl.write_csv(), 'data.csv')
            elif fmt=='Parquet':
                st.download_button('Parquet', df_pl.write_parquet(), 'data.parquet')
            else:
                st.download_button('JSON', df_pl.write_json(), 'data.json')

# 5) Graph
with tabs[4]:
    st.header('5. Graph')
    key = st.session_state.current
    if key:
        df_pd = st.session_state.datasets[key].to_pandas()
        src = st.selectbox('Source column', df_pd.columns, key='g_src')
        tgt = st.selectbox('Target column', df_pd.columns, key='g_tgt')
        G = nx.from_pandas_edgelist(df_pd, source=src, target=tgt)
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'))
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition='top center', marker=dict(size=10, color='skyblue'))
        fig = go.Figure(data=[edge_trace, node_trace])
        st.plotly_chart(fig, use_container_width=True)

# 6) Snowflake
with tabs[5]:
    st.header('6. Snowflake')
    key = st.session_state.current
    st.write('Configure Snowflake connection and table below')
    acc = st.text_input('Account'); usr = st.text_input('User'); pwd = st.text_input('Password', type='password')
    wh = st.text_input('Warehouse'); db = st.text_input('Database'); sc = st.text_input('Schema'); tbl = st.text_input('Table')
    if st.button('Write to Snowflake') and key:
        df_pl = st.session_state.datasets[key]
        conn = snowflake.connector.connect(user=usr, password=pwd, account=acc, warehouse=wh, database=db, schema=sc)
        write_pandas(conn, df_pl.to_pandas(), tbl)
        conn.close()
        st.success(f'Written to {tbl}')

# 7) AI Toolkit
with tabs[6]:
    st.header('7. AI Toolkit')
    st.write('Use AI buttons in Transform tab to generate Polars, Dask, or SQL code snippets.')
