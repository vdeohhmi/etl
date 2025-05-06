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
from datetime import datetime
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
        if ext == 'csv':
            return pd.read_csv(u)
        if ext in ['xls', 'xlsx']:
            return pd.read_excel(u, sheet_name=None)
        if ext == 'parquet':
            return pd.read_parquet(u)
        if ext == 'json':
            return pd.read_json(u)
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
                try:
                    df = df.query(expr)
                except Exception:
                    st.warning(f"Invalid filter: {expr}")
        elif t == 'compute':
            expr = step['expr']
            if expr.lower().startswith('sql:'):
                q = expr[4:].strip()
                try:
                    df = sqldf(q, {'df': df})
                except Exception:
                    st.warning(f"Invalid SQL: {q}")
            else:
                try:
                    df[step['new']] = df.eval(expr)
                except Exception:
                    st.warning(f"Invalid formula: {expr}")
        elif t == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif t == 'join' and step.get('aux'):
            aux_df = st.session_state.datasets.get(step['aux'])
            if aux_df is not None:
                df = df.merge(aux_df,
                              left_on=step['left'],
                              right_on=step['right'],
                              how=step['how'])
        elif t == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df[c].fillna(df[c].median(), inplace=True)
                    else:
                        df[c].fillna(df[c].mode().iloc[0], inplace=True)
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
    "📜 Pipeline"
])

# Datasets Tab
def datasets_tab():
    st.header("1. Upload & Manage Datasets")
    files = st.file_uploader("Upload files", 
                             type=['csv','xls','xlsx','parquet','json'], 
                             accept_multiple_files=True)
    if files:
        for u in files:
            df = load_file(u)
            if isinstance(df, dict):
                for sheet, sdf in df.items():
                    key = f"{u.name}:{sheet}"
                    st.session_state.datasets[key] = sdf
            elif df is not None:
                st.session_state.datasets[u.name] = df
        st.success(f"Loaded {len(files)} files into {len(st.session_state.datasets)} datasets.")
    if st.session_state.datasets:
        sel = st.selectbox("Select active dataset", list(st.session_state.datasets.keys()), key='sel_dataset')
        st.session_state.current = sel
        st.data_editor(st.session_state.datasets[sel], key=f"editor_{sel}", use_container_width=True)

def transform_tab():
    st.header("2. Build Transformations")
    key = st.session_state.current
    if not key:
        st.info("Upload and select a dataset first.")
        return
    df = st.session_state.datasets[key]
    # Show existing steps
    for i, s in enumerate(st.session_state.steps):
        st.write(f"{i+1}. {s['type']} — {s.get('desc','')}")
    st.markdown("---")
    op = st.selectbox(
        "Operation",
        ['rename','filter','compute','drop_const','onehot','join','impute'],
        key='op_transform'
    )
    # Rename
    if op == 'rename':
        old = st.selectbox("Old column", df.columns)
        new = st.text_input("New column name")
        if st.button("Add Rename"):
            st.session_state.steps.append({
                'type':'rename','old':old,'new':new,'desc':f"Rename {old}→{new}"}
            )
    # Filter
    elif op == 'filter':
        st.write("Available columns:", list(df.columns))
        expr = st.text_input("Filter expression")
        if st.button("Add Filter"):
            st.session_state.steps.append({
                'type':'filter','expr':expr,'desc':expr}
            )
    # Compute
    elif op == 'compute':
        newc = st.text_input("New column", help="Name for the computed column.")
        st.write("**Available columns:**", list(df.columns))
        st.write("**Functions:** np.log(), np.sqrt(), prefix SQL: for SQL queries")
        expr2 = st.text_input("Formula or SQL:", help="Enter a pandas eval formula or prefix with SQL:")
        if st.button("Add Compute"):
            st.session_state.steps.append({
                'type':'compute','new':newc,'expr':expr2,'desc':f"Compute {newc} = {expr2}"}
            )
    # Drop Constants
    elif op == 'drop_const':
        if st.button("Add Drop Constants"):
            st.session_state.steps.append({'type':'drop_const','desc':'Drop constant columns'})
    # One-Hot Encoding
    elif op == 'onehot':
        cols = st.multiselect("Columns to encode", df.select_dtypes('object').columns)
        if st.button("Add One-Hot"):
            st.session_state.steps.append({'type':'onehot','cols':cols,'desc':f"One-hot {cols}"})
    # Join
    elif op == 'join':
        aux = st.selectbox("Aux dataset", [k for k in st.session_state.datasets if k != key])
        left = st.selectbox("Left key", df.columns)
        right = st.selectbox("Right key", st.session_state.datasets[aux].columns)
        how = st.selectbox("Join type", ['inner','left','right','outer'])
        if st.button("Add Join"):
            st.session_state.steps.append({
                'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':f"Join {aux}"}
            )
    # Impute
    elif op == 'impute':
        if st.button("Add Auto-Impute"):
            st.session_state.steps.append({'type':'impute','desc':'Auto-impute'})
    # Apply transformations after selecting operation
    if st.button("Apply Transformations"):
        st.session_state.datasets[key] = apply_steps(df)
        st.success("Applied steps.")
    # Show transformed data
    st.data_editor(
        st.session_state.datasets[key],
        key=f"transformed_{key}",
        use_container_width=True
    )

def profile_tab():
():
    st.header("3. Data Profile")
    key = st.session_state.current
    if not key:
        st.info("Select a dataset first.")
        return
    df = st.session_state.datasets[key]
    stats = pd.DataFrame({
        'dtype': df.dtypes,
        'nulls': df.isna().sum(),
        'null_pct': df.isna().mean()*100
    })
    st.dataframe(stats, use_container_width=True)

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

def export_tab():
    st.header("5. Export & Writeback")
    key = st.session_state.current
    if not key:
        st.info("Select a dataset first.")
        return
    df = st.session_state.datasets[key]
    fmt = st.selectbox("Format", ['CSV','JSON','Parquet','Excel','Snowflake'], key='fmt')
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
            with st.form("sf_wb"):
                tbl = st.text_input("Table")
                sub = st.form_submit_button("Writeback")
            if sub:
                conn = get_sf_conn()
                write_pandas(conn, df, tbl)
                conn.close()
                st.success(f"Written to {tbl}")

def history_tab():
    st.header("6. History")
    for idx, (ts, snap) in enumerate(st.session_state['versions']):
        cols = st.columns([0.7, 0.3])
        cols[0].write(f"{idx+1}. {ts}")
        if cols[1].button("Revert", key=f"h_{idx}"):
            st.session_state.datasets[st.session_state.current] = snap
            st.experimental_rerun()

def snowflake_tab():
    st.header("7. Snowflake Settings")
    st.text_input("Account", key='sf_account')
    st.text_input("User", key='sf_user')
    st.text_input("Password", type='password', key='sf_password')
    st.text_input("Warehouse", key='sf_warehouse')
    st.text_input("Database", key='sf_database')
    st.text_input("Schema", key='sf_schema')

def pipeline_tab():
    st.header("8. Pipeline Configuration as YAML")
    yaml_steps = yaml.dump({'pipeline_steps': st.session_state.steps}, sort_keys=False)
    st.text_area("Pipeline YAML", yaml_steps, height=300)
    st.download_button("Download YAML", yaml_steps, file_name="pipeline.yaml", mime="text/yaml")

# Render Tabs
dataset_funcs = [datasets_tab, transform_tab, profile_tab, insights_tab, export_tab, history_tab, snowflake_tab, pipeline_tab]
for fn, tab in zip(dataset_funcs, tabs):
    with tab:
        fn()
