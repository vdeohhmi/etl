# app.py
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
import tableauserverclient as TSC

# --- App Configuration ---
st.set_page_config(page_title="Data Transformer Pro Plus", layout="wide")
st.title("🛠️ Data Transformer Pro Plus — Robust ETL Web App")

# --- Session State Initialization ---
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
            except Exception:
                pass
    return df


def load_file(uploader_file):
    ext = uploader_file.name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            return parse_dates(pd.read_csv(uploader_file))
        if ext in ['xls', 'xlsx']:
            sheets = pd.read_excel(uploader_file, sheet_name=None)
            return {name: parse_dates(df) for name, df in sheets.items()}
        if ext == 'parquet':
            return parse_dates(pd.read_parquet(uploader_file))
        if ext == 'json':
            return parse_dates(pd.read_json(uploader_file))
        if ext == 'twbx':
            # save the .twbx for further use or publishing
            path = os.path.join("/tmp", uploader_file.name)
            with open(path, "wb") as f:
                f.write(uploader_file.getbuffer())
            st.success(f"Uploaded {uploader_file.name}")
            return None
    except Exception as e:
        st.error(f"Failed to load {uploader_file.name}: {e}")
    return None


def apply_steps(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.versions.append((timestamp, df.copy()))
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
    "🕸️ Social Graph",
    "🔗 Tableau Writeback"
])

# 1. Datasets
with tabs[0]:
    st.subheader("1. Load Data")
    files = st.file_uploader(
        "Upload files (CSV/Excel/Parquet/JSON/TWBX)",
        type=['csv', 'xls', 'xlsx', 'parquet', 'json', 'twbx'],
        accept_multiple_files=True
    )
    if files:
        for file in files:
            data = load_file(file)
            if isinstance(data, dict):
                for sheet, df in data.items():
                    st.session_state.datasets[f"{file.name}:{sheet}"] = df
            elif isinstance(data, pd.DataFrame):
                st.session_state.datasets[file.name] = data
    if st.session_state.datasets:
        key = st.selectbox("Select dataset", list(st.session_state.datasets.keys()), key='sel')
        st.session_state.current = key
        st.data_editor(st.session_state.datasets[key], key=f"editor_{key}", use_container_width=True)

# 2. Transform
with tabs[1]:
    st.subheader("2. Transform Data")
    key = st.session_state.current
    if not key:
        st.info("Please load a dataset first.")
    else:
        df = st.session_state.datasets[key]
        for i, step in enumerate(st.session_state.steps):
            st.markdown(f"**Step {i+1}:** {step['type']} — {step.get('desc','')}")
        st.markdown("---")
        op = st.selectbox("Operation", ['rename','filter','compute','drop_const','onehot','join','impute'], key='op')
        # [Add operations as before]...
        if st.button("Apply Transformations"):
            st.session_state.datasets[key] = apply_steps(df)
            st.success("Transformations applied.")
        st.data_editor(st.session_state.datasets[key], key=f"transformed_{key}", use_container_width=True)

# 3. Profile
with tabs[2]:
    st.subheader("3. Data Profile")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        profile = pd.DataFrame({
            'dtype': df.dtypes,
            'nulls': df.isna().sum(),
            'null_pct': df.isna().mean()*100
        })
        st.dataframe(profile, use_container_width=True)

# 4. Insights
with tabs[3]:
    st.subheader("4. Auto Insights")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        num = df.select_dtypes(include=np.number)
        if not num.empty:
            st.plotly_chart(px.imshow(num.corr(), text_auto=True), use_container_width=True)

# 5. Export
with tabs[4]:
    st.subheader("5. Export & Writeback")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        fmt = st.selectbox("Export format", ['CSV','JSON','Parquet','Excel','Snowflake'], key='fmt')
        if st.button("Export Now"):
            # [Export logic as before]...
            pass

# 6. History
with tabs[5]:
    st.subheader("6. Transformation History")
    if st.session_state.versions:
        for i,(ts,snap) in enumerate(st.session_state.versions):
            cols=st.columns([0.7,0.3])
            cols[0].write(f"{i+1}. {ts}")
            if cols[1].button("Revert",key=f"rev{i}"):
                st.session_state.datasets[st.session_state.current]=snap
                st.experimental_rerun()

# 7. Snowflake Settings
with tabs[6]:
    st.subheader("7. Snowflake Configuration")
    st.text_input("Account", key='sf_account')
    st.text_input("Username", key='sf_user')
    st.text_input("Password", type='password', key='sf_password')
    st.text_input("Warehouse", key='sf_warehouse')
    st.text_input("Database", key='sf_database')
    st.text_input("Schema", key='sf_schema')

# 8. Pipeline YAML
with tabs[7]:
    st.subheader("8. Pipeline as YAML")
    yaml_str=yaml.dump({'pipeline_steps':st.session_state.steps},sort_keys=False)
    st.text_area("Pipeline YAML",yaml_str,height=300)
    st.download_button("Download YAML",yaml_str,"pipeline.yaml")

# 9. Social Graph
with tabs[8]:
    st.subheader("9. Social Network Graph")
    # [Social graph logic as before]...

# 10. Tableau Writeback
with tabs[9]:
    st.subheader("🔗 Tableau Writeback to Snowflake")
    twbx_file = st.file_uploader("Upload Tableau Packaged Workbook (.twbx)", type=['twbx'])
    workbook_url = st.text_input("Or enter Tableau workbook URL:")
    if workbook_url:
        st.markdown("**Preview:**")
        import streamlit.components.v1 as components
        components.iframe(workbook_url, height=600)
    comment_text = st.text_area("Add your comment on this workbook:")
    if st.button("Submit Comment"):
        if not (workbook_url or twbx_file) or not comment_text:
            st.error("Provide a workbook link/file and a comment.")
        else:
            dfc = pd.DataFrame([{
                'workbook': workbook_url if workbook_url else twbx_file.name,
                'comment': comment_text,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }])
            conn=get_sf_conn(); cur=conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS TABLEAU_COMMENTS (
                  workbook VARCHAR,
                  comment VARCHAR,
                  timestamp TIMESTAMP_NTZ
                )""")
            cur.close()
            success,_,nrows,_=write_pandas(conn,dfc,'TABLEAU_COMMENTS')
            conn.close()
            if success: st.success(f"Comment written ({nrows} rows).")
            else: st.error("Write-back failed.")
