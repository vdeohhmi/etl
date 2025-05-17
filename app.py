import os
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
    st.error("Please set the OPENAI_API_KEY environment variable to enable AI features.")
client = OpenAI(api_key=api_key)

# --- AI Helpers ---
@st.cache_data
def ai_pandas_expr(new_col: str, logic: str, sample: pd.DataFrame) -> str:
    prompt = (
        f"You are a Python data engineer. Given sample rows {sample.to_dict('records')} "
        f"and the requirement: {logic}, generate a valid pandas expression to compute '{new_col}'. "
        "Respond with only the expression, no extra text."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip().strip('`')

@st.cache_data
def ai_sql_expr(new_col: str, logic: str) -> str:
    prompt = (
        f"You are a SQL expert. Using a table named 'df', write a SELECT query that returns *, and adds a new column '{new_col}' "
        f"calculated as {logic}. Return only the full SQL query, no comments or explanation."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip().strip('`')

# --- File Loading ---
def load_file(f) -> pd.DataFrame:
    ext = f.name.split('.')[-1].lower()
    try:
        if ext=='csv': return pd.read_csv(f)
        if ext in ('xls','xlsx'): return pd.read_excel(f)
        if ext=='parquet': return pd.read_parquet(f)
        if ext=='json': return pd.read_json(f)
    except Exception as e:
        st.error(f"Failed to load {f.name}: {e}")
    return None

# --- Main Tabs ---
tab_ds, tab_tf, tab_ai, tab_pr, tab_ex, tab_graph, tab_sf = st.tabs([
    "📂 Datasets","✏️ Transform","🤖 AI Toolkit","📈 Profile",
    "⬇️ Export","🌐 Graph","⚙️ Snowflake"
])

# 1) Datasets
with tab_ds:
    st.header("1. Datasets")
    files = st.file_uploader("Upload CSV/Excel/Parquet/JSON", accept_multiple_files=True)
    if files:
        for f in files:
            df = load_file(f)
            if df is not None:
                st.session_state.datasets[f.name] = df
        st.success("Datasets loaded.")
    if st.session_state.datasets:
        sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()))
        st.session_state.current_df = sel
        st.dataframe(st.session_state.datasets[sel].head(10), use_container_width=True)

# 2) Transform
with tab_tf:
    st.header("2. Transform")
    key = st.session_state.current_df
    if not key:
        st.info("Please load and select a dataset first.")
    else:
        df = st.session_state.datasets[key]
        st.subheader("Current Preview")
        st.dataframe(df.head(5), use_container_width=True)

        op = st.selectbox("Operation", [
            'Rename Column','Filter Rows','Compute Column','SQL Transform',
            'Drop Constant Columns','One-Hot Encode','Impute Missing'
        ], key='op_tf')

        # Rename
        if op=='Rename Column':
            old = st.selectbox("Old Column", df.columns)
            new = st.text_input("New Column Name")
            if st.button("Apply Rename"):
                df = df.rename(columns={old:new})
                st.session_state.datasets[key] = df
                st.success(f"Renamed '{old}' to '{new}'.")

        # Filter
        if op=='Filter Rows':
            expr = st.text_input("Pandas query (e.g. age > 30)")
            if st.button("Apply Filter") and expr:
                df = df.query(expr)
                st.session_state.datasets[key] = df
                st.success("Filter applied.")

        # Compute
        if op=='Compute Column':
            new_col = st.text_input("New Column Name", key='comp_new')
            logic = st.text_area("Logic (plain English)", key='comp_logic')
            manual = st.text_input("Or manual pandas expression", key='comp_manual')
            if st.button("Generate Expression"):  # key-less single
                expr = ai_pandas_expr(new_col, logic, df.head(3))
                st.code(expr)
            else:
                expr = manual
            if expr and st.button("Apply Compute"):
                try:
                    df[new_col] = df.eval(expr, engine='python')
                    st.session_state.datasets[key] = df
                    st.success(f"Computed column '{new_col}'.")
                except Exception as e:
                    st.error(f"Compute failed: {e}")

        # SQL Transform
        if op=='SQL Transform':
            new_col_sql = st.text_input("New Column Name (SQL)", key='sql_new')
            logic_sql = st.text_area("Logic (plain English for SQL)", key='sql_logic')
            manual_sql = st.text_area("Or manual SQL query (use 'df' as table)", key='sql_manual')
            if st.button("Generate SQL"):  # generate via AI
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

        # Drop constants
        if op=='Drop Constant Columns' and st.button("Apply Drop Constants"):
            df = df.loc[:, df.nunique()>1]
            st.session_state.datasets[key] = df
            st.success("Dropped constant columns.")

        # One-hot
        if op=='One-Hot Encode':
            cols = st.multiselect("Columns to encode", df.select_dtypes('object').columns)
            if st.button("Apply One-Hot") and cols:
                df = pd.get_dummies(df, columns=cols)
                st.session_state.datasets[key] = df
                st.success("One-hot encoding applied.")

        # Impute
        if op=='Impute Missing' and st.button("Apply Impute"):
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = df[c].fillna(
                        df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode()[0]
                    )
            st.session_state.datasets[key] = df
            st.success("Imputation complete.")

        # Updated Preview
        st.subheader("Updated Preview")
        st.dataframe(st.session_state.datasets[key].head(5), use_container_width=True)

# 3) AI Toolkit
with tab_ai:
    st.header("3. AI Toolkit")
    st.write("Generate expressions in Transform tab using AI.")

# 4) Profile
with tab_pr:
    st.header("4. Profile")
    key = st.session_state.current_df
    if key:
        df = st.session_state.datasets[key]
        stats = pd.DataFrame({
            'dtype':df.dtypes,
            'nulls':df.isna().sum(),
            'pct_null':df.isna().mean()*100
        })
        st.dataframe(stats, use_container_width=True)

# 5) Export
with tab_ex:
    st.header("5. Export")
    key = st.session_state.current_df
    if key:
        df = st.session_state.datasets[key]
        fmt = st.selectbox("Format", ['CSV','Excel','Parquet','Snowflake'], key='ex_fmt')
        if fmt!='Snowflake' and st.button("Download File"):
            buf = BytesIO()
            if fmt=='CSV':
                st.download_button("CSV", df.to_csv(index=False).encode(),"data.csv")
            elif fmt=='Excel':
                df.to_excel(buf,index=False,engine='openpyxl')
                st.download_button("Excel", buf.getvalue(),"data.xlsx")
            else:
                st.download_button("Parquet", df.to_parquet(index=False),"data.parquet")
        if fmt=='Snowflake':
            st.info("Configure Snowflake in the last tab and click Write to Snowflake.")

# 6) Graph
with tab_graph:
    st.header("6. Social Graph")
    key = st.session_state.current_df
    if key:
        df = st.session_state.datasets[key]
        src = st.selectbox("Source", df.columns, key='g_src')
        tgt = st.selectbox("Target", df.columns, key='g_tgt')
        wt = st.selectbox("Weight (optional)", [None]+list(df.columns), key='g_wt')
        if st.button("Generate Graph"):
            G = nx.Graph()
            for _,r in df.iterrows():
                u,v = r[src], r[tgt]
                w = float(r[wt]) if wt else 1.0
                G.add_edge(u, v, weight=w)
            net = Network(height="600px",width="100%",bgcolor="#222222",font_color="white")
            net.show_buttons(filter_=['physics'])
            for n in G.nodes(): net.add_node(n,label=str(n),title=f"Degree:{G.degree(n)}",value=G.degree(n))
            for u,v,d in G.edges(data=True): net.add_edge(u,v,value=d['weight'],width=2)
            html = net.generate_html()
            import streamlit.components.v1 as components; components.html(html, height=650)

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
            user=usr,password=pwd,account=acc,
            warehouse=wh,database=db,schema=sc
        )
        write_pandas(conn, df, tbl)
        conn.close()
        st.success(f"Written to {tbl}.")
