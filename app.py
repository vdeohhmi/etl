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
        if ext in ['xls','xlsx']:
            return pd.read_excel(u, sheet_name=None)
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
tabs = st.tabs(["📂 Datasets","✏️ Transform","📈 Profile","💡 Insights","⬇️ Export","🕒 History","⚙️ Snowflake"])

# Datasets Tab
with tabs[0]:
    st.header("1. Upload & Manage Datasets")
    files = st.file_uploader("Upload files", type=['csv','xls','xlsx','parquet','json'], accept_multiple_files=True)
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
        sel = st.selectbox("Select active dataset", list(st.session_state.datasets.keys()))
        st.session_state.current = sel
        st.data_editor(st.session_state.datasets[sel], key=f"editor_{sel}", use_container_width=True)

# Transform Tab
with tabs[1]:
    st.header("2. Build Transformations")
    key = st.session_state.current
    if not key:
        st.info("Upload and select a dataset first.")
    else:
        df = st.session_state.datasets[key]
        for i,s in enumerate(st.session_state.steps): st.write(f"{i+1}. {s['type']} — {s.get('desc','')}")
        st.markdown("---")
        op = st.selectbox("Operation", ['rename','filter','compute','drop_const','onehot','join','impute'], key='op_transform')
        if op=='rename':
            old = st.selectbox("Old col", df.columns)
            new = st.text_input("New col name")
            if st.button("Add Rename"): st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f"Rename {old}→{new}"})
        if op=='filter':
            st.write(df.columns.tolist())
            expr = st.text_input("Filter expression")
            if st.button("Add Filter"): st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
        if op=='compute':
            newc = st.text_input("New column")
            expr2 = st.text_input("Formula or SQL:")
            if st.button("Add Compute"): st.session_state.steps.append({'type':'compute','new':newc,'expr':expr2,'desc':newc})
        if op=='drop_const' and st.button("Add Drop Constants"): st.session_state.steps.append({'type':'drop_const','desc':'Drop constants'})
        if op=='onehot':
            cols = st.multiselect("Cols to encode", df.select_dtypes('object').columns)
            if st.button("Add One-Hot"): st.session_state.steps.append({'type':'onehot','cols':cols,'desc':','.join(cols)})
        if op=='join':
            aux = st.selectbox("Aux dataset", [k for k in st.session_state.datasets if k!=key])
            left=st.selectbox("Left key", df.columns)
            right=st.selectbox("Right key", st.session_state.datasets[aux].columns)
            how=st.selectbox("How",['inner','left','right','outer'])
            if st.button("Add Join"): st.session_state.steps.append({'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':f"Join {aux}"})
        if op=='impute' and st.button("Add Impute"): st.session_state.steps.append({'type':'impute','desc':'Auto-impute'})
        if st.button("Apply"): st.session_state.datasets[key]=apply_steps(df); st.success("Applied steps.")
        st.data_editor(st.session_state.datasets[key], key=f"transformed_{key}", use_container_width=True)

# Profile Tab
with tabs[2]:
    st.header("3. Data Profile")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        stats=pd.DataFrame({'dtype':df.dtypes,'nulls':df.isna().sum(),'null_pct':df.isna().mean()*100})
        st.dataframe(stats)

# Insights Tab
with tabs[3]:
    st.header("4. Auto Insights")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        num=df.select_dtypes('number')
        if not num.empty:
            fig=px.imshow(num.corr(), text_auto=True, title='Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
            miss=df.isna()
            fig2=px.imshow(miss, title='Missingness')
            st.plotly_chart(fig2, use_container_width=True)

# Export Tab
with tabs[4]:
    st.header("5. Export & Writeback")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        fmt=st.selectbox("Format",['CSV','JSON','Parquet','Excel','Snowflake'], key='fmt')
        if st.button("Export"):
            if fmt=='CSV': st.download_button("CSV",df.to_csv(index=False).encode(),"data.csv")
            if fmt=='JSON': st.download_button("JSON",df.to_json(orient='records'),"data.json")
            if fmt=='Parquet': st.download_button("Parquet",df.to_parquet(index=False),"data.parquet")
            if fmt=='Excel': out=BytesIO();df.to_excel(out,index=False,engine='openpyxl');st.download_button("Excel",out.getvalue(),"data.xlsx")
            if fmt=='Snowflake':
                with st.form("sf_wb"): tbl=st.text_input("Table" ); sub=st.form_submit_button("Writeback")
                if sub:
                    conn=get_sf_conn(); write_pandas(conn,df,tbl); conn.close(); st.success(f"Written to {tbl}")

# History Tab
with tabs[5]:
    st.header("6. History")
    for idx,(ts,snap) in enumerate(st.session_state['versions']):
        cols=st.columns([0.7,0.3])
        cols[0].write(f"{idx+1}. {ts}")
        if cols[1].button("Revert",key=f"h_{idx}"):
            st.session_state.datasets[st.session_state.current]=snap
            st.experimental_rerun()

# Snowflake Config Tab
with tabs[6]:
    st.header("7. Snowflake Settings")
    st.text_input("Account", key='sf_account')
    st.text_input("User", key='sf_user')
    st.text_input("Password", type='password', key='sf_password')
    st.text_input("Warehouse", key='sf_warehouse')
    st.text_input("Database", key='sf_database')
    st.text_input("Schema", key='sf_schema')
